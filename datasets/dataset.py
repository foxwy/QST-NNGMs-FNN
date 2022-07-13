# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

# -----environment-----
filepath = os.path.abspath(os.path.join(os.getcwd(), '../..'))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# -----external libraries-----
sys.path.append('../..')

from datasets.data_generation import PaState
from Basis.Basic_Function import array_posibility_unique, data_combination
from Basis.Basic_Function import data_combination, qmt


# -----Dataset-----
def Dataset_P(rho_star, M, N, K, p=1, seed=1):   # 
    data_unique = data_combination(N, K, p, seed)
    P = qmt(rho_star, [M] * N)

    if p < 1:
        idxs = data_unique.dot(K**(np.arange(N - 1, -1, -1)))
        P = P[idxs]

    idx_nzero = P!=0
    data_unique = data_unique[idx_nzero]
    P = P[idx_nzero]

    return data_unique, P


def Dataset_sample(povm, state_name, N, sample_num, rho_p, rho_star=0, read_data=False):  # 全测量采样
    if read_data:
        if 'P' in state_name:  # mix state
            trainFileName = filepath + '/datasets/data/' + state_name + \
                '_' + str(rho_p) + '_' + povm + '_data_N' + str(N) + '.txt'
        else:  # pure state
            trainFileName = filepath + '/datasets/data/' + \
                state_name + '_' + povm + '_data_N' + str(N) + '.txt'
        data_all = np.loadtxt(trainFileName)[:sample_num].astype(int)

    else:
        sampler = PaState(povm, N, state_name, rho_p, rho_star)
        data_all, _ = sampler.samples_product(sample_num, save_flag=False)

    data_unique, P = array_posibility_unique(data_all)

    return data_unique, P


def Dataset_sample_P(povm, state_name, N, K, sample_num, rho_p, rho_star=0, read_data=False, p=1, seed=1):  # 随机测量采样：随机选取部分基
    S_choose = data_combination(N, K, p, seed)
    P_choose = np.zeros(len(S_choose))

    data_unique, P = Dataset_sample(povm, state_name, N, sample_num, rho_p, rho_star, read_data)

    for i in range(len(S_choose)):
        if len(np.where((data_unique == S_choose[i]).all(1))[0]) == 0:
            P_choose[i] = 0
        else:
            P_choose[i] = P[np.where(
                (data_unique == S_choose[i]).all(1))[0][0]]

    return S_choose, P_choose
