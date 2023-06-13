import geatpy as ea
import numpy as np
import os

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）

        M = 1 # 初始化M（目标维数）

        maxormins = [-1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 6000  # 决策变量维数 x_select.shape[0]  决策变量维度其实仍然是1000维  但挨着的几个维度是某个聚类中的样本的优先级表示

        varTypes = [0] * Dim  # 这是一个list,初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [i - i for i in range(Dim)]  # 决策变量下界
        ub = [i - i + 100 for i in range(Dim)]  # 决策变量上界
        lbin = [i - i + 1 for i in range(Dim)]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [i - i + 1 for i in range(Dim)]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        # 50 * 10000 的矩阵
        saved_path = 'E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/sota_result/moon/cifar10/nsga/'
        f = open(saved_path + 'A.txt', 'w')
        for i in range(len(Vars)):
            temp = list(Vars[i])
            temp = [str(val) for val in temp]
            writeline = '\t'.join(temp)
            f.write(writeline + "\n")
        f.close()

        import entropy
        entropy.get_objectives()

        f = open(saved_path+ 'EAobjective.txt', 'r')
        lines = f.readlines()
        vector = []

        for line in lines:
            objectives = line[:-1].split("\t")
            objectives = [float(val) for val in objectives]
            vector.append(objectives)

        f.close()

        pop.ObjV = np.array(vector)


def RunEA():

    """================================实例化问题对象============================="""
    problem = MyProblem()  # 生成问题对象
    """==================================种群设置================================"""
    Encoding = 'RI'  # 编码方式
    NIND = 100  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置==============================="""

    myAlgorithm = ea.moea_NSGA2_templet(problem, population)  # 实例化一个算法模板对象`
    myAlgorithm.MAXGEN = 50  # 最大进化代数
    myAlgorithm.logTras = 50  # 设置每多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = True  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """==========================调用算法模板进行种群进化=========================
    调用run执行算法模板，得到帕累托最优解集NDSet以及最后一代种群。NDSet是一个种群类Population的对象。
    NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。
    """
    # prophetChrom = get_init_pop()
    # prophetPop = ea.Population(Encoding, Field, NIND, prophetChrom)  # 实例化种群对象
    # myAlgorithm.call_aimFunc(prophetPop)  # 计算先知种群的目标函数值及约束（假如有约束）
    """===========================调用算法模板进行种群进化========================"""

    [NDSet, population] = myAlgorithm.run()  # 执行算法模板，得到非支配种群以及最后一代种群
    NDSet.save()  # 把非支配种群的信息保存到文件中
    """==================================输出结果=============================="""
    print('用时：%f 秒' % myAlgorithm.passTime)
    print('评价次数：%d 次' % myAlgorithm.evalsNum)
    print('非支配个体数：%d 个' % NDSet.sizes) if NDSet.sizes != 0 else print('没有找到可行解！')

    root_path = 'E:/githubAwesomeCode/1DLTesting/TestSelectionSOTA/sota_result/moon/'

    # 文件夹重命名，根据selectsize保存结果
    f = open( root_path + 'cifar10/nsga/selectsize.txt', 'r')
    line = f.readlines()[0]
    f.close()
    selectsize = int(line)

    writebasepath_name = root_path  + 'results/result'
    os.rename(writebasepath_name, root_path + 'results/' + str(selectsize))






