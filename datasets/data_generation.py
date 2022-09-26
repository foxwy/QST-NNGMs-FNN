#--------------------libraries--------------------
# internal libraries
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
import time

import sys
sys.path.append('..')

# external libraries
from Basis.Basis_State import Mea_basis
from Basis.Basic_Function import data_combination, data_combination_M2_single, qmt, samples_mp


#--------------------class--------------------
#-----PaState-----
class PaState(Mea_basis):
    def __init__(self, basis='Tetra', n_qubits=2, State_name='GHZ', P_state=0.0, rho_star=0):
        super().__init__(basis)
        self.N = n_qubits
        self.State_name = State_name
        self.p = P_state
        if type(rho_star) is np.ndarray:
            self.rho = rho_star
        else:
            _, self.rho = self.Get_state_rho(State_name, n_qubits, P_state)

    #-----adapting now-----
    def samples_product(self, Ns=1000000, filename='N2', group_N=5000, save_flag=True):  # faster using product construction and multiprocessing for batch processing
        if save_flag:
            if 'P' in self.State_name:  # mix state
                f_name = 'data/' + self.State_name + '_' + str(self.p) + '_' + self.basis + '_train_' + filename + '.txt'
                f2_name = 'data/' + self.State_name + '_' + str(self.p) + '_' + self.basis + '_data_' + filename + '.txt'
            else:  # pure state
                f_name = 'data/' + self.State_name + '_' + self.basis + '_train_' + filename + '.txt'
                f2_name = 'data/' + self.State_name + '_' + self.basis + '_data_' + filename + '.txt'

        P_all = qmt(self.rho, [self.M] * self.N)  # probs of all operators in product construction
        #counts = np.random.multinomial(1, P_all, Ns)  # larger memory

        if Ns < 5000:
            group_N = Ns
        params = [[P_all, group_N, self.K, self.N]] * int(Ns / group_N)
        cpu_counts = mp.cpu_count()
        if len(params) < cpu_counts:
            cpu_counts = len(params)

        print('---begin multiprocessing---')
        with mp.Pool(cpu_counts) as pool:  # long time!!!
            results = pool.map(samples_mp, params)
            pool.close()
            pool.join()
        print('---end multiprocessing---')

        S_all = results[0][0]
        S_one_hot_all = results[0][1]
        for num in range(1, len(results)):
            print('num:', group_N * (num + 1))
            S_all = np.vstack((S_all, results[num][0]))
            S_one_hot_all = np.vstack((S_one_hot_all, results[num][1]))
        print('---finished generating samples---')

        if save_flag:
            print('---begin write data to text---')
            np.savetxt(f_name, S_one_hot_all, '%d')
            np.savetxt(f2_name, S_all, '%d')
            print('---end write data to text---')

        return S_all, S_one_hot_all


#--------------------test--------------------
def Para_input():  # python data_generation.py Tetra 4 GHZ 0 1000
    print("basis", sys.argv[1])
    print("Number_qubits", int(sys.argv[2]))
    print("MPS", sys.argv[3])
    print("noise p ", float(sys.argv[4]))
    print("Nsamples", int(sys.argv[5]))
    sampler = PaState(basis=sys.argv[1], n_qubits=int(sys.argv[2]), MPS_name=sys.argv[3], p=float(sys.argv[4]))
    sampler.samples(Ns=int(sys.argv[5]))


#--------------------main--------------------
if __name__ == '__main__':
    '''
    sampler = PaMPS(basis='Pauli_normal')
    print(sampler.basis)
    sampler.Basis_info()
    '''

    # Para_input()

    '''
    num_qubits = 3
    sample_num = 20000
    sampler = PaMPS(basis='Tetra4', n_qubits=num_qubits)
    sampler.samples(sample_num, 'N'+str(num_qubits))'''

    num_qubits = 8
    sample_num = 1000000
    sampler = PaState(basis='Tetra4', n_qubits=num_qubits, State_name='random_P', P_state=0.4)

    t11 = time.perf_counter()
    sampler.samples_product(sample_num, 'N'+str(num_qubits), save_flag=False)
    t21 = time.perf_counter()

    print(t21-t11)

    '''
    B = Mea_basis('Tetra4')
    #rho = (B.I + 0.5*B.X + np.sqrt(3)/2*B.Y)/2
    s, rho = B.Get_state_rho('W', 10)
    #print('rho:', rho)
    #print(B.M)
    t1 = time.perf_counter()
    P = qmt(rho, [B.M]*10)
    t2 = time.perf_counter()
    print(t2-t1)
    print(P, sum(P))
    #B.Basis_info()'''
