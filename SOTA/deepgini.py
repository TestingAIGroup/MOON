import numpy as np
from keras.models import load_model
from utils_spectrum import load_compound_CIFAR10
from locate_sus_neurons import load_CIFAR
import argparse

def seed_deepgini(model, X_test):

    # 取gini不纯度高的
    all_gini=[]

    for idx in range(X_test.shape[0]):
        temp_img = X_test[[idx]]
        logits = model(temp_img)

        pro_sum = 0
        for pro in logits[0]:
            pro_sum = pro_sum + pro*pro
        t_gini = 1 - pro_sum
        all_gini.append(t_gini)
    gini_idx = np.argsort(all_gini)[::-1]

    return gini_idx

def main(args):
    if args.dataset == 'cifar':
        if args.model == 'convnet':
            model_path = 'cifar10/model_cifar_b4.h5'
            model_name = 'cifar_convnet'
            X_train, Y_train, X_test, Y_test = load_CIFAR()  # 在utils中修改

    model = load_model(model_path)
    model.summary()

    compound_ratio = [10000, 8000, 6000, 4000, 2000, 0]
    selectsize = [200, 400, 600, 800, 1000]
    randomsize = ['random1', 'random2', 'random3', 'random4', 'random5']

    # select_lst = np.random.choice(range(10000), replace=False, size= 6000)
    # all_lst = np.array( [l for l in range(10000)])
    # remain_lst = np.array([idx for idx in all_lst if idx not in select_lst])
    # np.save('./select_lst.npy', select_lst)
    # np.save('./remain_lst.npy', remain_lst)

    select_lst = np.load('./select_lst.npy')

    for ratio in compound_ratio:
        print('compound ratio', ratio)

        for random in randomsize:
            # 加载待selection 和 retraining 的数据
            compound_x_test, compound_y_test =  load_compound_CIFAR10(ratio, random)
            compound_x_test_select = compound_x_test[select_lst]
            print('compound_x_test_select: ', compound_x_test_select.shape)

            gini_idx = seed_deepgini(model, compound_x_test_select)

            for s in selectsize:
                    selected_idx = gini_idx[:s]
                    np.save('./sota_result/deepgini/{}/'
                    '{}_{}_deepgini_select_idx_compound_{}_{}.npy'.format(str(ratio), args.dataset, str(ratio) , random, s), selected_idx)


if __name__ == '__main__':
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


