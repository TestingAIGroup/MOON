import xlwt
import numpy as np
from keras import optimizers
from keras.models import load_model
from keras.utils import np_utils
import keras
from tensorflow.keras.optimizers import Adam
from keras.datasets import cifar10
import argparse
from locate_sus_neurons import load_CIFAR

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 70:
        lr *= 1e-3
    if epoch > 60:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def load_compound_cifar_vgg(ratio, random):
    root_path = './dataset/'
    data_path = 'adv_file/cifar_vgg/cifar_vgg_adv_compound_%s_%s.npz' % (str(ratio), random)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = np_utils.to_categorical(compound_data['y_test'], 10)
    return x_dest, y_dest

def load_compound_cifar(ratio, random):
    root_path = './dataset'
    data_path = 'adv_file/cifar_convnet/cifar_adv_compound_%s_%s.npz' % (str(ratio), random)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = np_utils.to_categorical(compound_data['y_test'], 10)
    return x_dest, y_dest


def write_results_to_file(all_result, root_path, dataset, approach):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)

    i = 0
    for index_path in all_result:
        print('cs: ', index_path)
        acc = all_result[index_path]  # {des: [accuracy] }
        sheet1.write(i, 0, index_path)
        j = 1
        for key in acc:
            sheet1.write(i, j, key)
            j += 1
        i = i + 1

    f.save(root_path + '{}_{}.xls'.format(dataset, approach))

def main(args):
    # 载入模型
    if args.dataset == "cifar":
        if args.model == 'convnet':
            model_path = './model/model_cifar.h5'  # [dense_1:20]  有train.py的model文件
            X_train, Y_train, X_test, Y_test = load_CIFAR()  # 在utils中修改
            model_name = 'convnet_cifar'
            saved_path = './results/cifar_convnet/cifar_convnet_total_EI_neurons/'.format(args.approach)


    select_lst = np.load('./select_lst.npy')
    remain_lst = np.load('./remain_lst.npy')

    compound_ratio = [10000, 8000, 6000, 4000, 2000, 0]
    selectsize = [200, 400, 600, 800, 1000]
    randomsize = ['random1' , 'random2',  'random3', 'random4', 'random5']

    all_result = {}

    for ratio in compound_ratio:
        for s in selectsize:
            for random in randomsize:
                # 从6000个样本中筛选，并用于重训练
                compound_x_test, compound_y_test = load_compound_cifar(ratio, random)
                compound_x_test_selection = compound_x_test[select_lst]
                compound_y_test_selection = compound_y_test[select_lst]

                index_path = '{}_{}/{}_{}_total_EI_neurons_compound_{}_{}_{}.npy'.format(str(ratio), random, model_name, approach[0], str(ratio),  random, str(s))
                select_sample_index = np.load(saved_path + index_path)
                print('select_sample_index: ', select_sample_index.shape)
                retrain_acc_sum = 0
                all_retrain_acc = [] # 存储所有重训练后模型的准确率

                # 循环多次取average
                for i in range(5):
                    model = load_model('/model/model_cifar.h5')
                    x = compound_x_test_selection[select_sample_index]
                    y = compound_y_test_selection[select_sample_index]

                    # 得到仿真数据集在模型上的准确率（即重训练之前的准确率）
                    x_remaining = compound_x_test[remain_lst]
                    y_remaining = compound_y_test[remain_lst]
                    _, origin_acc = model.evaluate(x_remaining, y_remaining)

                    # 使用原始的训练集、筛选出的测试子集，共同重训练模型
                    together_x = np.append(X_train, x, axis = 0)
                    together_y = np.append(Y_train, y, axis = 0)

                    # 重训练模型，得到重训练之后的准确率
                    opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
                    model.compile(loss='categorical_crossentropy',
                    optimizer = opt,
                    metrics=['accuracy'])

                    model.fit(together_x, together_y, batch_size = 500, epochs = 5, shuffle = True, verbose = 1, validation_data =(together_x, together_y))
                    _, retrain_acc = model.evaluate(x_remaining, y_remaining)
                    retrain_acc_sum += retrain_acc
                    all_retrain_acc.append(retrain_acc)

                retrain_average_acc = retrain_acc_sum / 5   #重训练后的平均值
                orig_acc_result = round(origin_acc, 4)    # 原始accuracy
                retrain_acc_result = round(retrain_average_acc, 4)
                acc_improvement_result = round(retrain_average_acc - origin_acc, 4)

                all_result[index_path] = [orig_acc_result, all_retrain_acc[0], all_retrain_acc[1], all_retrain_acc[2], all_retrain_acc[3], all_retrain_acc[4],
                                      retrain_acc_result, acc_improvement_result]

    write_results_to_file(all_result, saved_path, args.dataset, args.approach)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-m', '--model',
        help="Model to use; either 'lenet', 'convnet' or 'vgg' ",
        required=False, type=str
    )
    parser.add_argument(
        '-a', '--approach',
        help="appraoch to use; either 'moon', 'dsa', 'mcp', 'ces' or 'deepgini' ",
        required=False, type=str
    )
    args = parser.parse_args()
    main(args)





