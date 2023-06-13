
import EA


if __name__ == "__main__":

    selectsize = [200, 400, 600, 800, 1000]

    saved_path = './sota_result/moon/cifar10/'

    for s in selectsize:
        f_selectisize = open(saved_path + 'nsga/'+ 'selectsize.txt', 'w')
        f_selectisize.write(str(s))
        f_selectisize.close()
        EA.RunEA()
