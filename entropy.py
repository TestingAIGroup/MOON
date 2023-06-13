from utils_spectrum import count_num
import numpy as np

def count_num(compound_x_test_ats, EI_neuron_idx, select_sample_index, activation_threshold):

    number = 0
    selectedX_ats = compound_x_test_ats[select_sample_index]   #(100,512)

    for idx in EI_neuron_idx:  #[20,30,22,29,19...]
        for inp in selectedX_ats:
            if inp[idx] > activation_threshold[idx]:
                number = number +1

    return number

def get_random_p(num_EI):

    p_ratio = 0.8
    total_num = round(num_EI * p_ratio)

    ran_p = np.zeros(num_EI)
    s = np.random.choice(range(num_EI), replace=False, size=total_num)
    print(s)
    for i in range(len(s)):
        ran_p[s[i]] = 1

    return ran_p

def get_objectives():
    dataset = 'cifar10_m2'
    compound_ratio = [0]
    randomsize = ['random1']
    spectrum_approach = ['ochiai']

    saved_path = './sota_result/moon/cifar10/'

    f = open(saved_path + 'nsga/' + 'selectsize.txt', 'r')
    line = f.readlines()[0]
    f.close()
    selectsize = int(line)

    # 获得当前种群
    f = open(saved_path + 'nsga/' + 'A.txt', 'r')
    populations = f.readlines()
    f.close()

    EI_neuron_idx = np.load(saved_path + 'EI_neurons/{}_{}_EI_neuron_idx.npy'.format(dataset, spectrum_approach[0]))
    ran_p = get_random_p(len(EI_neuron_idx))
    temp = ran_p * EI_neuron_idx
    final_EI_neuron_idx = temp[temp != 0].astype(int)

    f_score = open(saved_path + 'nsga/' + 'EAobjective.txt', 'w')

    for vector in populations:
        vector = vector[:-1].split("\t")
        vector = np.array([float(val) for val in vector])

        select_sample_index = list(np.argsort(-vector)[:selectsize])
        compound_x_test_ats = np.load(saved_path + 'ats/{}_all_ats_compound_selection_{}_{}.npy'.format(dataset,  str(compound_ratio[0]), randomsize[0]))
        EI_ats_compound = compound_x_test_ats[:, final_EI_neuron_idx]  #[5000, 102]


        total_EI_ats = 0
        for i in select_sample_index:
            selected_EI_ats = np.sum(EI_ats_compound[i])
            total_EI_ats = total_EI_ats + selected_EI_ats

        norm = 0
        for w in select_sample_index:
            for k in range(w + 1, selectsize):
                norm += np.linalg.norm(compound_x_test_ats[w] - compound_x_test_ats[k])

        f_score.write(  str(total_EI_ats)  +  "\t" + str(norm) +  "\n")

    f_score.close()
