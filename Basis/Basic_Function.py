# -*- coding: utf-8 -*-
# @Author: foxwy
# @Date:   2021-04-30 09:48:23
# @Last Modified by:   yong
# @Last Modified time: 2022-12-06 10:13:28

#--------------------libraries--------------------
# internal libraries
import numpy as np
import torch
from torch.nn.functional import softmax
import random
from scipy.linalg import eigh
import time

# external libraries
import sys
sys.path.append('..')

from evaluation.ncon import ncon

#--------------------function--------------------

def clamp(n, minn, maxn):  # limit num to [minn, maxn]
    return max(min(maxn, n), minn)


# ten_to_k
def ten_to_k(num, k, N):  # 10进制转为k进制, N为总的位数，不足位数的补0
    transfer_num = []
    if num > k**N - 1:  # 大于设置的位数
        print('please consider the right N!')
    else:
        while num != 0:
            num, a = divmod(num, k)
            transfer_num.append(a)
        transfer_num = transfer_num[::-1]
        if len(transfer_num) != N:
            transfer_num = [0] * (N - len(transfer_num)) + transfer_num

    return transfer_num


# data_combination
def data_combination(N, k, p=1, seed=1):  # 从所有可能的组合中随机选取一定比例的组合，N位k
    samples_unique = []
    N_choice = k**N

    if p < 1:
        N_choice = int(N_choice * p)
        random.seed(seed)
        if N_choice % 2 == 1:  # single
            N_choice += 1
        num_choice = random.sample(range(k**N), N_choice)
    else:
        num_choice = range(N_choice)

    for num in num_choice:
        samples_unique.append(ten_to_k(num, k, N))

    return np.array(samples_unique)


# data_combination_M
def data_combination_M(M, N, k, p=1, seed=1):  # 列出所有可能的组合，N位k，增加M矩阵的tensor乘积组合
    samples_unique = []
    M_all = []
    random.seed(seed)
    N_choice = int(k**N * p)
    '''
    if N_choice % 2 == 1:  # single
        N_choice += 1'''
    num_choice = random.sample(range(k**N), N_choice)
    # print(num_choice)
    for num in num_choice:
        sample = ten_to_k(num, k, N)
        samples_unique.append(sample)
        M_temp = M[sample[0]]
        for i in sample[1:]:
            M_temp = np.kron(M_temp, M[i])
        M_all.append(M_temp)

    return samples_unique, np.array(M_all)


# 从采的样本中计算M
def data_combination_M2_single(M, samples_unique_i):  # M=[M1, M2, M3..], samples_unique_i=[1, 2..]
    M_temp = M[samples_unique_i[0]]
    for i in samples_unique_i[1:]:
        M_temp = np.kron(M_temp, M[i])

    return np.array(M_temp)


def data_combination_M2(M, samples_unique):
    M_all = []
    for samples_unique_i in samples_unique:
        M_temp = data_combination_M2_single(M, samples_unique_i)
        M_all.append(M_temp)

    return np.array(M_all)


# onehot
def onehot(data, k):
    data_onehot = []
    N = len(data[0])
    for i in range(len(data)):
        one_hot = np.squeeze(np.reshape(np.eye(k)[data[i]], [1, N * k]).astype(np.uint8)).tolist()
        data_onehot.append(one_hot)

    return np.array(data_onehot)


# ati_onehot
def ati_onehot(data_onehot, k):
    N = int(data_onehot.shape[1] // k)
    data_onehot_reshape = np.reshape(data_onehot, [data_onehot.shape[0] * N, k])
    data_ati_onehot = np.argmax(data_onehot_reshape, axis=1)
    data_ati_onehot = np.reshape(np.array(data_ati_onehot), [data_onehot.shape[0], N])

    return data_ati_onehot


# data_smaple_p
def data_sample_p(data_sample, Prob, Ns):
    data_samples = []
    for i, P in enumerate(Prob):
        for j in range(int(round(Ns * P))):
            data_samples.append(data_sample[i])

    return np.array(data_samples)


# data_sample_p_noise
def data_sample_p_noise(data_sample, Prob, Ns, noise):
    data_samples = []
    sample_num = []
    sample_num_temp = []
    for i, P in enumerate(Prob):
        N = int(Ns * P)
        if N * noise >= 1:
            N += np.random.randint(-N * noise, N * noise)
        if N < 0:
            N = 0
        sample_num_temp.append(N)

    for i, N in enumerate(sample_num_temp):
        num = int(N / sum(sample_num_temp) * Ns)
        sample_num.append(num)
        for j in range(num):
            data_samples.append(data_sample[i])

    return np.array(data_samples), sample_num


# array_posibility: 统计array中元素出现的频率
def array_posibility(a):
    x, cnts = np.unique(a, axis=0, return_counts=True)
    #print(x, cnts)

    b = np.zeros((len(a), 1))
    for i in range(len(a)):
        b[i, 0] = cnts[np.where((x == a[i]).all(1))[0][0]]
    b = b / len(a)
    # print(b)
    return b


def array_posibility_unique(a):
    x, cnts = np.unique(a, axis=0, return_counts=True)
    return x, cnts / len(a)


# semidefinite_adjust: 判断矩阵是否为半正定矩阵
def semidefinite_adjust(M):
    M_vals, M_vecs = eigh(M)
    if np.all(M_vals > -1e-4):
        return True
    else:
        return False


# Cal_cond: 计算矩阵广义逆和条件数
def Cal_cond(M_all):  # 矩阵条件数
    t = ncon((M_all, M_all), ([-1, 1, 2], [-2, 2, 1]))
    #it = np.linalg.inv(t)
    t_T_con = t.T.conjugate()
    it = (np.linalg.inv(t_T_con.dot(t))).dot(t_T_con)
    it_cond = np.linalg.cond(it)

    return it, it_cond


# 对一个数进行因式分解
def factorization(num):
    factor = []
    while num > 1:
        for i in range(num - 1):
            k = i + 2
            if num % k == 0:
                factor.append(k)
                num = int(num / k)
                break
    return factor


# 分解数为最相近的两个值的乘积
def crack(integer):
    start = int(np.sqrt(integer))
    factor = integer / start
    while not is_integer(factor):
        start += 1
        factor = integer / start
    return int(factor), start


def is_integer(number):
    if int(number) == number:
        return True
    else:
        return False

# 找到x的前n的和大于b，返回n，相对直接法，二分法在数字大的时候优势明显
def Find_x(x, b):
    if len(x) == 1:
        return 0
    else:
        mid_idx = len(x) // 2
        if sum(x[:mid_idx]) > b:
            x = x[:mid_idx]
            return Find_x(x, b)
        else:
            y = x[mid_idx:]
            y[0] += sum(x[:mid_idx])
            return mid_idx + Find_x(y, b)


#-----numpy version-----
def shuffle_forward(rho, dims):
    N = len(dims)
    rho = rho.reshape(np.concatenate((dims, dims), 0))
    ordering = np.reshape(np.arange(2*N).reshape(2, -1).T, -1)
    rho = np.transpose(rho, ordering)

    return rho


#-----numpy version-----
# 利用乘积结构简化计算复杂度
def qmt(X, operators):  # operators = [M1, M2, ....]
    #time_b = time.perf_counter()
    if not isinstance(operators, list):
        operators = [operators]

    X = np.array(X)
    N = len(operators)  # qubits number
    Ks = np.zeros(N, dtype=np.int)
    Ds = np.zeros(N, dtype=np.int)
    Rs = np.zeros(N, dtype=np.int)
    for i in range(N):
        dims = operators[i].shape
        Ks[i] = dims[0]
        Ds[i] = dims[1]
        Rs[i] = dims[2]
        operators[i] = operators[i].reshape((Ks[i], Ds[i]*Ds[i]))

    X = shuffle_forward(X, Ds)
    for i in range(N-1, -1, -1):
        P = operators[i]
        X = X.T
        X = X.reshape(Ds[i]*Ds[i], -1)
        X = P.dot(X)

    P_all = np.maximum(np.real(X.reshape(-1)), 0)
    P_all /= np.sum(P_all)
    #print('cal P time:', time.perf_counter() - time_b)

    return P_all


def qmt_pure(X, operators):  # operators = [M1, M2, ....], two loops
    #time_b = time.perf_counter()
    if not isinstance(operators, list):
        operators = [operators]

    X = np.array(X)
    N = len(operators)  # qubits number
    Ks = np.zeros(N, dtype=np.int)
    Ds = np.zeros(N, dtype=np.int)
    for i in range(N):
        dims = operators[i].shape
        Ks[i] = dims[0]
        Ds[i] = dims[1]
        if i < N - 1:
            operators[i] = operators[i].reshape((Ks[i], Ds[i]**2))

    P_all = np.zeros(np.prod(Ks))
    X = X.reshape(2, -1)
    X_T = X.T.conjugate()
    for k in range(Ks[-1]):
        X_k = X_T.dot(operators[-1][k]).dot(X)

        X_k = shuffle_forward(X_k, Ds[:-1])
        for i in range(N - 2, -1, -1):
            P = operators[i]
            X_k = X_k.reshape(-1, Ds[i]**2).T
            X_k = P.dot(X_k)
        P_all[k * 4**(N - 1): ((k + 1) * 4**(N - 1))] = np.maximum(np.real(X_k.reshape(-1)), 0)

    P_all /= np.sum(P_all)
    #print('cal P time:', time.perf_counter() - time_b)

    return P_all


#-----pytorch version-----
def shuffle_forward_torch(rho, dims):
    N = len(dims)
    rho = rho.reshape(tuple(torch.cat([dims, dims], 0)))
    ordering = torch.reshape(torch.arange(2*N).reshape(2, -1).T, (1, -1))[0]
    rho = rho.permute(tuple(ordering))

    return rho


#-----pytorch version-----
# 利用乘积结构简化计算复杂度
def qmt_torch(X, operators):  # operators = [M1, M2, ....]
    #time_b = time.perf_counter()
    if not isinstance(operators, list):
        operators = [operators]

    N = len(operators)  # qubits number
    Ks = torch.zeros(N, dtype=torch.int)
    Ds = torch.zeros(N, dtype=torch.int)
    for i in range(N):
        dims = operators[i].shape
        Ks[i] = dims[0]
        Ds[i] = dims[1]
        operators[i] = operators[i].reshape((Ks[i], Ds[i]*Ds[i]))

    if N > 12:
        X = X.cpu()
    X = shuffle_forward_torch(X.to(torch.complex64), Ds)
    X = X.permute(*torch.arange(X.ndim - 1, -1, -1))
    X = X.reshape(Ds[i]*Ds[i], -1)
    if N > 12:
        X = X.to(operators[0].device)

    for i in range(N - 1, -1, -1):
        P = operators[i]
        X = torch.matmul(P, X)

        if i > 0:
            X = X.permute(*torch.arange(X.ndim - 1, -1, -1))
            X = X.reshape(Ds[i]*Ds[i], -1)

    P_all = torch.maximum(torch.real(X.reshape(-1)), torch.tensor(0))
    P_all /= torch.sum(P_all)
    #print('cal torch P time:', time.perf_counter() - time_b)

    return P_all


def qmt_torch_pure(X, operators):  # operators = [M1, M2, ....]
    #time_b = time.perf_counter()
    if not isinstance(operators, list):
        operators = [operators]

    N = len(operators)  # qubits number
    Ks = torch.zeros(N, dtype=torch.int)
    Ds = torch.zeros(N, dtype=torch.int)
    for i in range(N):
        dims = operators[i].shape
        Ks[i] = dims[0]
        Ds[i] = dims[1]
        if i < N - 1:
            operators[i] = operators[i].reshape((Ks[i], Ds[i]**2))

    P_all = torch.zeros(torch.prod(Ks)).to(X.device)
    X = X.reshape(2, -1).to(torch.complex64)
    X_T = X.T.conj()
    for k in range(Ks[-1]):
        X_k = torch.matmul(X_T, torch.matmul(operators[-1][k], X))

        if N > 13:
            X_k = X_k.cpu()
        X_k = shuffle_forward_torch(X_k, Ds[:-1])
        X_k = X_k.reshape(-1, Ds[k]**2)
        if N > 13:
            X_k = X_k.to(operators[0].device)

        for i in range(N - 2, -1, -1):
            P = operators[i]
            X_k = X_k.permute(*torch.arange(X_k.ndim - 1, -1, -1))
            X_k = torch.matmul(P, X_k)

            if i > 0:
                X_k = X_k.reshape(-1, Ds[i]**2)

        P_all[k * 4**(N - 1): ((k + 1) * 4**(N - 1))] = torch.maximum(torch.real(X_k.reshape(-1)), torch.tensor(0))

    P_all /= torch.sum(P_all)
    #print('cal torch pure P time:', time.perf_counter() - time_b)

    return P_all


#-----numpy version-----
# 非厄密矩阵转为距离最近的密度矩阵（厄密，单位迹，半正定）
def proj_spectrahedron(rho):
    eigenvalues, eigenvecs = np.linalg.eigh(rho)  #eigenvalues[i], eigenvecs[:, i]
    #print(eigenvalues, eigenvecs)
    eigenvalues = np.real(eigenvalues)
    u = -np.sort(-eigenvalues)
    csu = np.cumsum(u)
    t = (csu - 1) / np.arange(1, len(u) + 1)
    idx_max = np.flatnonzero(u > t)[-1]
    eigenvalues = np.maximum(eigenvalues - t[idx_max], 0)

    print(eigenvecs.shape, eigenvalues.shape)
    A = eigenvecs * np.sqrt(eigenvalues)
    rho = A.dot(A.T.conjugate())

    return rho


#-----pytorch version-----
# 厄密矩阵转为距离最近的密度矩阵
def eigenvalues_trans_S(eigenvalues, device):
    u, _ = torch.sort(eigenvalues)
    csu = torch.cumsum(u, 0)
    t = (csu - 1) / torch.arange(1, len(u) + 1).to(device)
    idx_max = torch.nonzero(u > t)[-1, 0]
    eigenvalues = torch.maximum(eigenvalues - t[idx_max], torch.tensor(0))

    return eigenvalues


def eigenvalues_trans_F(eigenvalues, device):
    eigenvalues = eigenvalues / torch.sum(eigenvalues)
    u, _ = torch.sort(eigenvalues)
    csu = torch.cumsum(u, 0)
    csu0 = torch.zeros_like(csu).to(device)
    csu0[1:] = csu[:-1]
    t = csu0 / torch.arange(len(u), 0, -1).to(device)
    idx = torch.nonzero(u + t > 0)[0, 0]
    eigenvalues = torch.maximum(eigenvalues + t[idx], torch.tensor(0))

    return eigenvalues


def eigenvalues_trans_abs(eigenvalues, P_proj):
    eigenvalues_abs = torch.abs(eigenvalues)

    return eigenvalues_abs**P_proj / torch.sum(eigenvalues_abs**P_proj)


def proj_spectrahedron_torch(rho, device, map_method, P_proj):
    eigenvalues, eigenvecs = torch.linalg.eigh(rho)  # eigenvalues[i], eigenvecs[:, i]

    #eigenvalues = softmax(eigenvalues, 0)
    if map_method == 'proj_F':
        eigenvalues = eigenvalues_trans_F(eigenvalues, device)
    elif map_method == 'proj_S':
        eigenvalues = eigenvalues_trans_S(eigenvalues, device)
    elif map_method == 'proj_A':
        eigenvalues = eigenvalues_trans_abs(eigenvalues, P_proj)
    else:
        print('we have not this map method! please check setting')

    A = eigenvecs * eigenvalues
    rho = torch.matmul(A, eigenvecs.T.conj())
    rho /= torch.trace(rho)  # prevent errors caused by computing accuracy

    return rho


def samples_mp(param):
    P_all = param[0]
    group_N = param[1]
    K = param[2]
    N = param[3]

    counts = np.random.multinomial(1, P_all, group_N)
    idxs = np.argmax(counts, 1)
    S_all = []
    S_one_hot_all = []
    for n in range(group_N):
        ii = idxs[n]
        S = np.array(ten_to_k(ii, K, N))
        S_all.append(S)
        S_one_hot_all.append(np.squeeze(np.reshape(np.eye(K)[S], [1, N * K]).astype(np.uint8)).tolist())

    return [np.array(S_all), np.array(S_one_hot_all)]


#--------------------main--------------------
if __name__ == '__main__':
    #a = np.array([[1, 2], [3, 4], [1, 2], [1, 3], [1, 3], [3, 4], [1, 3], [3, 4], [1, 2]])
    # print(array_posibility(a))

    '''
    data = data_combination(2, 4, 0.5)
    print(data)
    data_onehot = onehot(data, 4)
    print(data_onehot)
    print(ati_onehot(data_onehot, 4))'''

    '''
    a, b = crack(164)
    print(a, b)'''

    '''
    a = np.array([[1, 2], [3, 4], [1, 2], [1, 3], [1, 3], [3, 4], [1, 3], [3, 4], [1, 2]])
    data_unique, p = array_posibility_unique(a)
    print(data_unique, p)'''

    '''
    a = np.array([[1, 2, 0, 0.5], [0, 3, 0, 0], [4, 0, 4, 0], [0.5, 0, 0, 0.5]])
    print(a)
    b = shuffle_forward(a, [2, 2])
    print(b)
    c = b.reshape(-1, 4).T
    print(c)'''

    
    rho = np.array([[1, 3+2j], [3-2j, 2]])
    rho_t = torch.tensor(rho)
    V, W = torch.linalg.eigh(rho_t)
    print(torch.matmul(W, W.T.conj()))
    print(proj_spectrahedron(rho))

    '''
    rho_t = torch.tensor(rho)
    print(proj_spectrahedron_torch(rho_t, 'imag'))'''