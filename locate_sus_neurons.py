
from tensorflow.keras.datasets import cifar10
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from utils_spectrum import filter_correct_classifications, get_layers_ats, construct_spectrum_matrices, obtain_results, get_bin
from utils_spectrum import write_results_to_file, load_compound_CIFAR10
from spectrum_analysis import tarantula_analysis, ochiai_analysis, dstar_analysis
import heapq
import os
import scipy.io as sio
import argparse

PATH_DATA = "./svhn_data/"
def load_svhn():

    train = sio.loadmat(os.path.join(PATH_DATA, 'svhn_train.mat'))
    test = sio.loadmat(os.path.join(PATH_DATA, 'svhn_test.mat'))

    X_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
    X_test = np.transpose(test['X'], axes=[3, 0, 1, 2])

    y_train = np.reshape(train['y'], (-1,)) - 1
    y_test = np.reshape(test['y'], (-1,)) - 1

    X_train = X_train.astype('float64')
    X_test = X_test.astype('float64')
    X_train = (X_train / 255.0)
    X_test = (X_test / 255.0)

    # one-hot-encode the labels
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    return X_train, Y_train, X_test, Y_test

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

def load_MNIST(one_hot=True, channel_first=True):

    # Load data,可以不用下载，在keras.datasets包中直接调用
    mnist_path = 'E:\\githubAwesomeCode\\1DLTesting\\1dataset\\deepimportance_mnist_cifar\\mnist.npz'
    mnist_file = np.load(mnist_path)
    X_train, y_train = mnist_file['x_train'], mnist_file['y_train']
    X_test, y_test = mnist_file['x_test'], mnist_file['y_test']
    mnist_file.close()

    # Preprocess dataset
    # Normalization and reshaping of input.
    if channel_first:
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    else:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    if one_hot:
        # For output, it is important to change number to one-hot vector.
        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test = np_utils.to_categorical(y_test, num_classes=10)

    return X_train, y_train, X_test, y_test


def get_fixed_threshold():
    thres = np.zeros(256)
    for i in range(len(thres)):
        thres[i] = 0.5

    return thres

def main(args):
    assert args.dataset in ['mnist', 'cifar', 'svhn'], \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.model in ['lenet', 'convnet', 'vgg'], \
        "Model parameter must be either 'lenet', 'convnet', 'vgg' "
    assert args.attack in ['fgsm', 'bim-a', 'bim-b', 'jsma'], \
        "Model parameter must be either 'lenet', 'convnet', 'vgg' "

    if args.dataset == 'cifar':
        if args.model == 'convnet':
            model_path = 'cifar10/model_cifar_b4.h5'
            model_name = 'cifar_convnet'
            X_train, Y_train, X_test, Y_test = load_CIFAR()  # 在utils中修改

    model = load_model(model_path)
    model.summary()

    _, acc = model.evaluate(X_test, Y_test)
    print('acc: ', acc)

    X_train_corr, Y_train_corr, X_train_misc, Y_train_misc, train_corr_idx, train_misc_idx = \
        filter_correct_classifications(model, X_train, Y_train)

    all_results_acc = {}; all_results_incon = {}
    adv_name = args.attack
    com_size = [10000, 8000, 6000, 4000, 2000, 0]
    susp_num = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

    trainable_layers = [20]
    layer_names = [model.layers[idx].name for idx in trainable_layers]
    train_ats, train_pred = get_layers_ats(model, X_train, layer_names, args.dataset)

    for cs in com_size:
        compound_x_test, compound_y_test = load_compound_CIFAR10(cs, adv_name)
        activation_threshold = np.array(get_bin(train_ats))
        # activation_threshold = train_ats.mean(axis=0)
        # activation_threshold = get_fixed_threshold()

        neuron_num = activation_threshold.shape[0]
        correct_classifications = train_corr_idx
        misclassifications = train_misc_idx
        #
        spectrum_num = construct_spectrum_matrices(model, trainable_layers, correct_classifications, misclassifications,
                                                   train_ats, activation_threshold, model_name)

        # spectrum_num  = [num_ac, num_uc, num_af, num_uf]
        num_ac = spectrum_num[0]; num_uc = spectrum_num[1];
        num_af = spectrum_num[2]; num_uf = spectrum_num[3]
        spectrum_approach = ['tarantula', 'ochiai', 'dstar']
        results_acc = {}
        results_incon = {}

        # # # locate suspicious neurons
        for app in spectrum_approach:

            if app == 'tarantula':
                suspiciousness = tarantula_analysis(num_ac, num_uc, num_af, num_uf)

            elif app == 'ochiai':
                suspiciousness = ochiai_analysis(num_ac, num_uc, num_af, num_uf)

            elif app == 'dstar':
                suspiciousness = dstar_analysis(num_ac, num_uc, num_af, num_uf)

            total_test_acc = []; total_test_incon = []
            for num in susp_num:
                arr_max = heapq.nlargest(int(num * neuron_num), suspiciousness)
                suspicious_neuron_idx = map(suspiciousness.index, arr_max)

                test_accuracy, test_ratio = \
                    obtain_results(model_path, trainable_layers, suspicious_neuron_idx, compound_x_test,
                                   compound_y_test)
                total_test_acc.append(test_accuracy)
                total_test_incon.append(test_ratio)

            results_acc[app + '_compound' + str(cs)] = total_test_acc
            results_incon[app + '_compound' + str(cs) + '_ratio'] = total_test_incon

        all_results_acc[cs] = results_acc
        all_results_incon[cs] = results_incon

    write_results_to_file(all_results_acc, all_results_incon, model_name, adv_name, susp_num)

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
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim-a', 'bim-b' or 'jsma' "
             "or 'all'",
        required=True, type=str
    )
    args = parser.parse_args()
    main(args)


