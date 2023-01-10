# -*- coding: utf-8 -*-
# @Author: foxwy
# @Function: Provide some loss functions
# @Paper: Ultrafast quantum state tomography with feed-forward neural networks

import os
import sys
import numpy as np
import torch
import h5py

filepath = os.path.abspath(os.path.join(os.getcwd(), '../..'))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append('..')
from Basis.Basis_State import Mea_basis, State
from datasets.data_generation import PaState
from Basis.Basic_Function import (array_posibility_unique, 
                                  data_combination, 
                                  num_to_groups)
from Basis.Basic_Function import (data_combination, 
                                  qmt, 
                                  qmt_pure, 
                                  qmt_torch, 
                                  qmt_torch_pure, 
                                  get_default_device)


def Dataset_P(rho_star, M, N, K, ty_state, p=1, seed=1):
    """
    Noise-free quantum measurements are sampled and some of these probabilities 
    are selected proportionally.

    Args:
        rho_star (tensor): The expected density matrix.
        M (tensor): The POVM, size (K, 2, 2).
        N (int): The number of qubits.
        K (int): The number of POVM elements.
        ty_state (str): The type of state, include 'mixed' and 'pure'.
        p (float): Selected measurement base ratio.
        seed (float): Random seed.

    Returns:
        tensor: Index of the sampled measurement base, with the zero removed.
        tensor: Probability distribution of sampling, with the zero removed.
        tensor: Probability distribution of sampling, include all measurement.
    """
    data_unique = data_combination(N, K, p, seed)  # part sample 

    if ty_state == 'pure':
        P = qmt_torch_pure(rho_star, [M] * N)
    else:
        P = qmt_torch(rho_star, [M] * N)
    P_idx = torch.arange(0, len(P), device=P.device)  # index of probability

    if p < 1:
        idxs = data_unique.dot(K**(np.arange(N - 1, -1, -1)))
        P = P[idxs]
        P_idx = P_idx[idxs]

    idx_nzero = P > 0
    return P_idx[idx_nzero], P[idx_nzero], P


def Dataset_sample(povm, state_name, N, sample_num, rho_p, ty_state, rho_star=0, M=None, read_data=False):
    """
    Quantum sampling with noise.

    Args:
        povm (str): The name of measurement, as Mea_basis().
        state_name (str): The name of state, as State().
        N (int): The number of qubits.
        sample_num (int): Number of samples to be sampled.
        rho_p (str): The P of Werner state, pure state when p == 1, identity matrix when p == 0.
        ty_state (str): The type of state, include 'mixed' and 'pure'.
        rho_star (array, tensor): The expect density matrix, assign the value directly if it exists, 
            otherwise regenerate it.
        M (tensor): The POVM, size (K, 2, 2).
        read_data (bool): If true, read the sample data from the ``.txt`` file.

    Returns:
        tensor: Index of the sampled measurement base, with the zero removed.
        tensor: Probability distribution of sampling, with the zero removed.
        tensor: Probability distribution of sampling, include all measurement.
    """
    if read_data:
        if 'P' in state_name:  # mix state
            trainFileName = filepath + '/datasets/data/' + state_name + \
                '_' + str(rho_p) + '_' + povm + '_data_N' + str(N) + '.txt'
        else:  # pure state
            trainFileName = filepath + '/datasets/data/' + \
                state_name + '_' + povm + '_data_N' + str(N) + '.txt'
        data_all = np.loadtxt(trainFileName)[:sample_num].astype(int)
    else:
        sampler = PaState(povm, N, state_name, rho_p, ty_state, M, rho_star) 
        #data_all, _ = sampler.samples_product(sample_num, save_flag=False)  # low
        P_idxs, P, P_all = sampler.sample_torch(sample_num, save_flag=False)  # faster

    #data_unique, P = array_posibility_unique(data_all)
    #return data_unique, P
    return P_idxs, P, P_all


def Dataset_sample_P(povm, state_name, N, K, sample_num, rho_p, ty_state, rho_star=0, read_data=False, p=1, seed=1):
    """The combination of ``Dataset_P`` and ``Dataset_sample``"""
    S_choose = data_combination(N, K, p, seed)
    P_choose = np.zeros(len(S_choose))

    data_unique, P, P_all = Dataset_sample(povm, state_name, N, sample_num, rho_p, ty_state, rho_star, read_data)

    for i in range(len(S_choose)):
        if len(np.where((data_unique == S_choose[i]).all(1))[0]) == 0:
            P_choose[i] = 0
        else:
            P_choose[i] = P[np.where(
                (data_unique == S_choose[i]).all(1))[0][0]]

    return S_choose, P_choose, P_all


def Dataset_train(data_num, name, M, N, K, ty_state, device='cpu'):
    """
    The data needed to generate the network pre-training, including probability and 
    density matrix reverse mapping data, see paper ``Ultrafast quantum state tomography 
    with feed-forward neural networks``.

    Args:
        data_num (int): The number of dataset.
        name (str): The name of saved file.
        M (tensor): The POVM, size (K, 2, 2).
        N (int): The number of qubits.
        K (int): The number of POVM elements.
        ty_state (str): The type of state, include 'mixed' and 'pure'.
        device (torch.device): GPU or CPU.

    Save:
        The dataset is saved using h5py at ``datasets/``.
    """
    num_group = num_to_groups(data_num, 5000)  # 5000 is the size of each packet
    for idx, num in enumerate(num_group):
        X = np.zeros((num, K**N))
        Y = np.zeros((num, 4**N))

        for i in range(num):
            _, rho = State().Get_real_random_state(N)
            rho = torch.from_numpy(rho).to(torch.complex64).to(device)

            if ty_state == 'pure':
                P = qmt_torch_pure(rho, [M] * N)
            else:
                P = qmt_torch(rho, [M] * N)

            # reverse state-mapping precess
            # chol
            #L, _ = torch.linalg.cholesky_ex(rho)
            #Y_i = (L.real + L.imag.T).reshape(-1)

            # chol_h
            eigenvalues, eigenvecs = torch.linalg.eigh(rho)
            eigenvalues_sqrt = torch.sqrt(torch.abs(eigenvalues))
            A = eigenvecs * eigenvalues_sqrt
            H = torch.matmul(A, eigenvecs.T.conj())
            L = torch.tril(H)
            Y_i = (L.real + L.imag.T).reshape(-1)
            #print(torch.dist(L @ L.mT, rho))

            # data
            X[i] = P.cpu().numpy()
            Y[i] = Y_i.cpu().numpy()

            if i % 1000 == 0:
                print('---Generating data {:d}th:{:>5d} / {:d}'.format(idx + 1, idx * 5000 + i + 1000, data_num))

        # save dataset
        h5f = h5py.File('datasets/'+name+str(idx)+'.h5', 'w')
        h5f['data'] = X
        h5f['target'] = Y
        h5f.close()

    return 


class GetLoader(torch.utils.data.Dataset):
    """Dataset loading"""
    def __init__(self, name='datasets/data_', N=2):

        self.name = name  # saved file path
        self.N = N  # number of qubits

    def __getitem__(self, index):
        num = int(index / 5000)  # 5000 is the size of each packet

        f = h5py.File(self.name+str(num)+'.h5', 'r')
        index -= 5000 * num

        data = f['data'][index]
        target = f['target'][index]

        return data, target

    def __len__(self):
        return self.N * 5000


if __name__ == '__main__':
    n_qubits = 6
    POVM = 'Tetra4'
    ty_state = 'mixed'
    device = get_default_device()

    M = Mea_basis(POVM).M
    M = torch.from_numpy(M).to(device)
    Dataset_train(100000, 'data_', M, n_qubits, 4, ty_state, device=device)  # trainset
    Dataset_train(20000, 'data_test_', M, n_qubits, 4, ty_state, device=device)  # testset
