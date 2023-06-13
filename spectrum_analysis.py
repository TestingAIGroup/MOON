import numpy as np
import math

def tarantula_analysis(num_ac, num_uc, num_af, num_uf):

    result = []
    i = 0
    for j in range(len(num_af[0])):
        result.append(
            float(float(num_af[i][j]) / (num_af[i][j] + num_uf[i][j])) / \
        (float(num_af[i][j]) / (num_af[i][j] + num_uf[i][j]) + float(num_ac[i][j]) / (num_ac[i][j] + num_uc[i][j]))
        )

    return result


def ochiai_analysis( num_ac, num_uc, num_af, num_uf):

    result = []
    i = 0
    for j in range(len(num_af[0])):
        result.append(
            float(num_af[i][j]) / ((num_af[i][j] + num_uf[i][j]) * (num_af[i][j] + num_ac[i][j])) ** (.5)
        )

    return result

def dstar_analysis(num_ac, num_uc, num_af, num_uf):

    star = 3
    result = []
    i = 0
    for j in range(len(num_af[0])):
        result.append(
            float(num_af[i][j] ** star) / (num_ac[i][j] + num_uf[i][j])
        )

    return result
