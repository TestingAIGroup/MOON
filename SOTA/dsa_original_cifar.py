from multiprocessing import Pool
from keras.models import Model
import os
from scipy.stats import gaussian_kde
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from utils_spectrum import load_compound_CIFAR10
from locate_sus_neurons import load_CIFAR
import argparse


def load_compound_cifar(ratio, random):
    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/'
    data_path = 'adv_file/cifar10/cifar_adv_compound_%s_%s.npz' % (str(ratio), random)
    compound_data = np.load( root_path + data_path)

    x_dest =  compound_data['x_test']
    y_dest = np_utils.to_categorical(compound_data['y_test'], 10)
    return x_dest, y_dest


class SurpriseAdequacy:
   # sa = SurpriseAdequacy(model, X_train, layer_names, upper_bound, dataset, str(ratio), s, random)

    def __init__(self,  model, train_inputs, layer_names, upper_bound, dataset, cs, selected_size, random):

        #self.surprise = surprise
        self.model = model
        self.train_inputs = train_inputs
        self.layer_names = layer_names
        self.upper_bound = upper_bound
        self.n_buckets = 1000
        self.dataset = dataset
        self.save_path='E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/sota_result/dsa/'
        if dataset == 'drive': self.is_classification = False   #处理非分类任务
        else: self.is_classification = True
        self.num_classes = 10
        self.var_threshold = 1e-5
        self.cs = cs
        self.selected_size = selected_size
        self.random = random

    def test(self, test_inputs, dataset_name, instance='lsa'):

        # if instance == 'dsa':
        #     print('dataset_name: ', dataset_name)
        #     target_sa = fetch_dsa(self.model, self.train_inputs, test_inputs,
        #                            dataset_name, self.layer_names,
        #                            self.num_classes, self.is_classification,
        #                            self.save_path, self.dataset)
        #
        #     np.save(self.save_path + '{}/{}_{}_dsa_compound_{}.npy'.format(self.cs, dataset_name, str(self.cs), self.random), target_sa)

        #     # find_sorted_inputs(self.save_path, dataset_name, self.adv_name, str(self.cs), self.selected_size, target_sa)

        # if instance == 'lsa':
        #     print(len(test_inputs))
        #     target_sa = fetch_lsa(self.model, self.train_inputs, test_inputs,
        #                            dataset_name, self.layer_names,
        #                            self.num_classes, self.is_classification,
        #                            self.var_threshold, self.save_path, self.dataset)
        #     np.save(self.save_path + '{}/{}_{}_lsa_compound_{}.npy'.format(self.cs, dataset_name, str(self.cs),
        #                                                                    self.random), target_sa)

        find_sorted_inputs(self.save_path, dataset_name, str(self.cs), self.selected_size, self.random)


def find_sorted_inputs(save_path, dataset_name, cs, selected_size, random):
    # 下面两个文件中结果均一致，一个是topk_neuron_idx(使用的是全部测试样本), 一个是直接用当前代码文件生成的
    # Noting！对原始测试集进行分割，前5000用于selection 和热training;后5000用于评估
    # target_sa = np.load(save_path + '{}_{}_dsa_compound_{}.npy'.format(dataset_name, adv_name, cs))
    target_sa = np.load(save_path + '{}/{}_{}_lsa_compound_{}.npy'.format(cs, dataset_name, cs, random))  #topk_neuron_idx

    print('target_sa: ', target_sa)
    sa_index = np.argsort(target_sa)[-selected_size:]
    print('target_sa[sa_index]:', target_sa[sa_index])
    print('sa_index: ', sa_index)
    np.save(save_path + '{}/{}_lsa_{}_{}_select_idx_compound_{}_{}.npy'.format(cs, dataset_name, cs, 'dsa', random, selected_size), sa_index)

    return target_sa

def fetch_lsa(model, X_train, x_target, target_name, layer_names, num_classes,is_classification, var_threshold, save_path, dataset):

    prefix = "[" + target_name + "] "
    train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(
        model, X_train, x_target, target_name, layer_names, num_classes,
        is_classification, save_path, dataset)

    class_matrix = {}
    if is_classification:
        for i, label in enumerate(train_pred):
            if label.argmax(axis=-1) not in class_matrix:
                class_matrix[label.argmax(axis=-1)] = []
            class_matrix[label.argmax(axis=-1)].append(i)
        print('yes')
    print(class_matrix.keys()) #dict_keys([5, 0, 4, 1, 9, 2, 3, 6, 7, 8])

    kdes, removed_cols = _get_kdes(train_ats, train_pred, class_matrix,
                                   is_classification, num_classes, var_threshold)

    lsa = []
    print(prefix + "Fetching LSA")
    if is_classification:
        for i, at in enumerate(target_ats):
            label = target_pred[i].argmax(axis=-1)
            kde = kdes[label]
            lsa.append(_get_lsa(kde, at, removed_cols))
    else:
        kde = kdes[0]
        for at in target_ats:
            lsa.append(_get_lsa(kde, at, removed_cols))
    return lsa


def _get_kdes(train_ats, train_pred, class_matrix, is_classification, num_classes, var_threshold):

    is_classification =True
    removed_cols = []
    if is_classification:
        for label in range(num_classes):
            col_vectors = np.transpose(train_ats[class_matrix[label]])
            for i in range(col_vectors.shape[0]):
                if ( np.var(col_vectors[i]) < var_threshold and i not in removed_cols ):
                    removed_cols.append(i)

        kdes = {}
        for label in range(num_classes):
            refined_ats = np.transpose(train_ats[class_matrix[label]])
            refined_ats = np.delete(refined_ats, removed_cols, axis=0)

            if refined_ats.shape[0] == 0:
                print("ats were removed by threshold {}".format(var_threshold))
                break
            kdes[label] = gaussian_kde(refined_ats)
    else:
        col_vectors = np.transpose(train_ats)
        for i in range(col_vectors.shape[0]):
            if np.var(col_vectors[i]) < var_threshold:
                removed_cols.append(i)

        refined_ats = np.transpose(train_ats)
        refined_ats = np.delete(refined_ats, removed_cols, axis=0)
        if refined_ats.shape[0] == 0:
            print("ats were removed by threshold {}".format(var_threshold))
        kdes = [gaussian_kde(refined_ats)]

    return kdes, removed_cols

def _get_lsa(kde, at, removed_cols):
    refined_at = np.delete(at, removed_cols, axis=0)
    return np.asscalar(-kde.logpdf(np.transpose(refined_at)))

def fetch_dsa(model, x_train, x_target, target_name, layer_names, num_classes, is_classification, save_path, dataset):

    train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(
        model, x_train, x_target, target_name, layer_names, num_classes,
        is_classification, save_path, dataset)

    print('train_ats: ', train_ats.shape)
    print('target_ats: ',target_ats.shape)
    class_matrix = {}
    all_idx = []
    for i, label in enumerate(train_pred):
        if label.argmax(axis=-1) not in class_matrix:
            class_matrix[label.argmax(axis=-1)] = []
        class_matrix[label.argmax(axis=-1)].append(i)
        all_idx.append(i)

    dsa = []

    for i, at in enumerate(target_ats):
        label = target_pred[i].argmax(axis=-1)
        a_dist, a_dot = find_closest_at(at, train_ats[class_matrix[label]])
        b_dist, _ = find_closest_at(a_dot, train_ats[list(set(all_idx) - set(class_matrix[label]))])
        dsa.append(a_dist / b_dist)

    return dsa

def _get_train_target_ats(model, x_train, x_target, target_name, layer_names,
                          num_classes, is_classification, save_path, dataset):

    saved_train_path = _get_saved_path(save_path, dataset, "train", layer_names)

    if os.path.exists(saved_train_path[0]):
        print("Found saved {} ATs, skip serving".format("train"))
        # In case train_ats is stored in a disk
        train_ats = np.load(saved_train_path[0])  # train_ats:  (60000, 12)
        train_pred = np.load(saved_train_path[1])  # train_pred:  (60000, 10)
        print('train_ats: ', train_ats.shape)
        print('train_pred: ', train_pred.shape)

    else:
        train_ats, train_pred = get_ats(
            model,
            x_train,
            "train",
            layer_names,
            num_classes = num_classes,
            is_classification=is_classification,
            save_path=saved_train_path,
        )

    saved_target_path = _get_saved_path(save_path, dataset, 'cifar10', layer_names)
    if os.path.exists(saved_target_path[0]):
        print("Found saved {} ATs, skip serving").format(target_name)
        # In case target_ats is stored in a disk
        target_ats = np.load(saved_target_path[0])
        target_pred = np.load(saved_target_path[1])
        print('target_ats: ', target_ats.shape)
        print('target_pred: ', target_pred.shape)

    else:
        # target就是X_train
        target_ats, target_pred = get_ats(
            model,
            x_target, #X_test
            target_name,
            layer_names,
            num_classes=num_classes,
            is_classification=is_classification,
            save_path=saved_target_path,
        )

    return train_ats, train_pred, target_ats, target_pred

def get_ats( model, dataset, name, layer_names, save_path=None, batch_size=128, is_classification=True, num_classes=10, num_proc=10,):

    temp_model = Model(
        inputs=model.input, #Tensor("input_1:0", shape=(None, 28, 28, 1), dtype=float32)
        outputs=[model.get_layer(layer_name).output for layer_name in layer_names], #layer_name 层的神经元输出的值
    )

    if is_classification:
        p = Pool(num_proc)  #Pool类可以提供指定数量的进程供用户调用，当有新的请求提交到Pool中时，如果池还没有满，就会创建一个新的进程来执行请求。
        pred = model.predict(dataset, batch_size=batch_size, verbose=1)

        if len(layer_names) == 1:  #计算coverage的只有一层
            layer_outputs = [temp_model.predict(dataset, batch_size=batch_size, verbose=1)]
        else:
            layer_outputs = temp_model.predict(dataset, batch_size=batch_size, verbose=1)

        ats = None

        for layer_name, layer_output in zip(layer_names, layer_outputs):  # (1, 60000, 4, 4, 12)
            if layer_output[0].ndim == 3:
                # For convolutional layers and pooling layers 数据的维数是3维的
                layer_matrix = np.array(p.map(_aggr_output, [layer_output[i] for i in range(len(dataset))]))
                print('layer_matrix_1: ', layer_matrix.shape, layer_matrix)
            else:
                layer_matrix = np.array(layer_output)
                print('layer_matrix_2: ', layer_matrix.shape, layer_matrix)

            if ats is None:
                ats = layer_matrix
            else:
                ats = np.append(ats, layer_matrix, axis=1)
                layer_matrix = None

    # if save_path is not None:
    #     np.save(save_path[0], ats)
    #     np.save(save_path[1], pred)

    return ats, pred


def _get_saved_path(base_path, dataset, dtype, layer_names):
    joined_layer_names = "_".join(layer_names)
    return (
        os.path.join(
            base_path,
            dataset + "_" + dtype + "_" + joined_layer_names + "_ats" + ".npy",
        ),
        os.path.join(base_path, dataset + "_" + dtype + "_pred" + ".npy"),
    )


def find_closest_at(at, train_ats):
    #The closest distance between subject AT and training ATs.

    dist = np.linalg.norm(at - train_ats, axis=1)
    return (min(dist), train_ats[np.argmin(dist)])

#计算1范数的距离，
def find_closest_at_ord1(at, train_ats):

    dist = np.linalg.norm(at - train_ats, ord=1, axis=1)  #二范数
    return (min(dist), train_ats[np.argmin(dist)])

def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]

def main(args):
    if args.dataset == 'cifar':
        if args.model == 'convnet':
            model_path = 'cifar10/model_cifar_b4.h5'
            model_name = 'cifar_convnet'
            X_train, Y_train, X_test, Y_test = load_CIFAR()  # 在utils中修改

    model = load_model(model_path)
    model.summary()
    upper_bound = 2000

    # skip flattern layers和inputlayers
    skip_layers = []
    for idx, lyr in enumerate(model.layers):
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)

    # for lenet4: 除input、flatten和softmax外的所有层
    subject_layer = list(set(range(len(model.layers))) - set(skip_layers))[:-1]

    layer_names = []
    lyr = [20]
    for ly_id in lyr:
        layer_names.append(model.layers[ly_id].name)
    print(layer_names)


    compound_ratio = [10000, 8000, 6000, 4000, 2000, 0]
    selectsize = [200, 400, 600, 800, 1000]
    randomsize = ['random1', 'random2', 'random3', 'random4', 'random5']

    select_lst = np.load('./select_lst.npy')

    for ratio in compound_ratio:
        print('compound ratio', ratio)
        for random in randomsize:
            compound_x_test, compound_y_test = load_compound_CIFAR10(ratio, random)
            compound_x_test_select = compound_x_test[select_lst]
            print('compound_x_test_select: ', compound_x_test_select.shape)

            for s in selectsize:
                sa = SurpriseAdequacy(model, X_train, layer_names, upper_bound, args.dataset, str(ratio), s, random)
                sa.test(compound_x_test_select, args.dataset)


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


