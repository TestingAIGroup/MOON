import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from keras.datasets import cifar10
from retrain_iterate import load_compound_cifar_vgg
import argparse

def find_second(act):
    max_ = 0
    second_max = 0
    sec_index = 0
    max_index = 0
    for i in range(10):
        if act[i] > max_:
            max_ = act[i]
            max_index = i

    for i in range(10):
        if i == max_index:
            continue
        if act[i] > second_max:  # 第2大加一个限制条件，那就是不能和max_一样
            second_max = act[i]
            sec_index = i
    ratio = 1.0 * second_max / max_
    # print 'max:',max_index
    return max_index, sec_index, ratio  # ratio是第二大输出达到最大输出的百分比

def select_my_optimize(model, selectsize, x_target, y_test):

    act_layers =model.predict(x_target)
    dicratio =[[] for i in range(100) ]  # 只用90，闲置10个
    dicindex =[[] for i in range(100)]
    for i in range(len(act_layers)):
        act = act_layers[i]
        max_index, sec_index, ratio =find_second(act)  # max_index
        dicratio[max_index * 10 + sec_index].append(ratio)
        dicindex[max_index * 10 + sec_index].append(i)

    selected_lst = select_from_firstsec_dic(selectsize, dicratio, dicindex)

    return selected_lst


# 输入第一第二大的字典，输出selected_lst。用例的index
def select_from_firstsec_dic(selectsize, dicratio, dicindex):
    selected_lst = []
    tmpsize = selectsize

    noempty = no_empty_number(dicratio)
    # print(selectsize)
    # print(noempty)
    while selectsize >= noempty:
        for i in range(100):
            if len(dicratio[i]) != 0:
                tmp = max(dicratio[i])
                j = dicratio[i].index(tmp)
                if tmp >= 0.1:
                    selected_lst.append(dicindex[i][j])
                dicratio[i].remove(tmp)
                dicindex[i].remove(dicindex[i][j])
        selectsize = tmpsize - len(selected_lst)
        noempty = no_empty_number(dicratio)

    while len(selected_lst) != tmpsize:
        max_tmp = [0 for i in range(selectsize)]
        max_index_tmp = [0 for i in range(selectsize)]
        for i in range(100):
            if len(dicratio[i]) != 0:
                tmp_max = max(dicratio[i])
                if tmp_max > min(max_tmp):
                    index = max_tmp.index(min(max_tmp))
                    max_tmp[index] = tmp_max
                    # selected_lst.append()
                    # if tmp_max>=0.1:
                    max_index_tmp[index] = dicindex[i][dicratio[i].index(tmp_max)]
        if len(max_index_tmp) == 0 and len(selected_lst) != tmpsize:
            print('wrong!!!!!!')
            break
        selected_lst = selected_lst + max_index_tmp
    # print(selected_lst)
    assert len(selected_lst) == tmpsize
    return selected_lst

def no_empty_number(dicratio):
    no_empty=0
    for i in range(len(dicratio)):
        if len(dicratio[i])!=0:
            no_empty+=1
    return no_empty

def load_CIFAR(one_hot=True):
    CLIP_MAX = 0.5
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.astype("float32")
    X_train = (X_train / 255.0) - (1.0 - CLIP_MAX)
    X_test = X_test.astype("float32")
    X_test = (X_test / 255.0) - (1.0 - CLIP_MAX)

    if one_hot:
        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test = np_utils.to_categorical(y_test, num_classes=10)

    return X_train, y_train, X_test, y_test

def main(args):
    if args.dataset == 'cifar':
        if args.model == 'convnet':
            model_path = 'cifar10/model_cifar_b4.h5'
            model_name = 'cifar_convnet'
            X_train, Y_train, X_test, Y_test = load_CIFAR()  # 在utils中修改

    model = load_model(model_path)
    model.summary()

    compound_ratio = [10000, 8000, 6000, 4000, 2000, 0]
    selectsize = [1000]
    randomsize = ['random1', 'random2', 'random3', 'random4', 'random5']

    select_lst = np.load('./select_lst.npy')

    for ratio in compound_ratio:
        print('compound ratio', ratio)
        for random in randomsize:
            compound_x_test, compound_y_test = load_compound_cifar_vgg(ratio, random)
            compound_x_test_select = compound_x_test[select_lst]
            compound_y_test_select = compound_y_test[select_lst]
            print('compound_x_test_select: ', compound_x_test_select.shape)

            for s in selectsize:
                selected_lst = select_my_optimize(model, s, compound_x_test_select, compound_y_test_select)
                np.save('./sota_result/mcp/{}/'
                        '{}_{}_mcp_select_idx_compound_{}_{}.npy'.format(str(ratio), args.dataset, str(ratio), random, s),
                        selected_lst)


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
