# -*- coding: utf-8 -*-
# @Author: foxwy
# @Date:   2021-01-19 15:38:06
# @Last Modified by:   WY
# @Last Modified time: 2021-09-30 14:44:09

#--------------------libraries--------------------
# internal libraries
import numpy as np


#--------------------class--------------------
#-----state-----
class State():
    def __init__(self):
        # Pauli matrices
        self.I = np.array([[1, 0], [0, 1]])
        self.X = np.array([[0, 1], [1, 0]])
        self.s1 = self.X
        self.Y = np.array([[0, -1j], [1j, 0]])
        self.s2 = self.Y
        self.Z = np.array([[1, 0], [0, -1]])
        self.s3 = self.Z

        # state
        self.state0 = np.matrix([[1], [0]])
        self.state1 = np.matrix([[0], [1]])
        self.state01 = 1 / np.sqrt(2) * (self.state0 + self.state1)

    # ----------state, rho----------
    def Get_state_rho(self, state_name, N, p=1):
        # mix state with real part
        if state_name == 'GHZ_P':
            state, rho = self.Get_GHZ_P(N, p)
        elif state_name == 'GHZi_P':
            state, rho = self.Get_GHZi_P(N, p)
        elif state_name == 'Product_P':
            state, rho = self.Get_Product_P(N, p)
        elif state_name == 'W_P':
            state, rho = self.Get_W_P(N, p)
        elif state_name == 'random_P':
            state, rho = self.Get_random_state(N, p)

        return state, rho

    def Get_GHZ_P(self, N, p):  # mix state
        state0 = self.state0
        state1 = self.state1
        for _ in range(N - 1):
            state0 = np.kron(state0, self.state0)
            state1 = np.kron(state1, self.state1)
        GHZ_state = 1 / np.sqrt(2) * (state0 + state1)
        GHZ_rho = GHZ_state.dot(GHZ_state.T.conjugate())

        # mix
        GHZ_P_rho = p * GHZ_rho + (1 - p) / 2**N * np.matrix(np.eye(2**N))

        return GHZ_state, np.array(GHZ_P_rho)

    def Get_GHZi_P(self, N, p):  # mix state
        state0 = self.state0
        state1 = self.state1
        for _ in range(N - 1):
            state0 = np.kron(state0, self.state0)
            state1 = np.kron(state1, self.state1)
        GHZi_state = 1 / np.sqrt(2) * (state0 + 1j * state1)
        GHZi_rho = GHZi_state.dot(GHZi_state.T.conjugate())

        # mix
        GHZi_P_rho = p * GHZi_rho + (1 - p) / 2**N * np.matrix(np.eye(2**N))

        return GHZi_state, np.array(GHZi_P_rho)

    def Get_Product_P(self, N, p):
        Product_state = self.state01
        for _ in range(N - 1):
            Product_state = np.kron(Product_state, self.state01)
        Product_rho = Product_state.dot(Product_state.T.conjugate())

        # mix
        Product_P_rho = p * Product_rho + (1 - p) / 2**N * np.matrix(np.eye(2**N))

        return Product_state, np.array(Product_P_rho)

    def Get_state_from_array(self, array):
        st = {0: self.state0, 1: self.state1}
        State = st[array[0]]
        for i in array[1:]:
            State = np.kron(State, st[i])

        return State

    def Get_W_P(self, N, p):
        I_array = np.identity(N)
        W_state = 0
        for row in I_array:
            W_state += self.Get_state_from_array(row)
        W_state = 1 / np.sqrt(N) * W_state
        W_rho = W_state.dot(W_state.T.conjugate())

        # mix
        W_P_rho = p * W_rho + (1 - p) / 2**N * np.matrix(np.eye(2**N))

        return W_state, np.array(W_P_rho)

    def Get_random_state(self, N, purity):  # PGD paper
        lambda_t = 0
        purity_t = 0
        x = np.arange(1, 2**N + 1)

        while purity_t < purity:
            lambda_t += 0.001
            lam = np.exp(-lambda_t * x)
            lamb = lam / np.sum(lam)
            purity_t = np.sum(lamb**2)

        randM = np.random.uniform(size=(2**N, 2**N)) * \
                np.exp(1j * 2 * np.pi * np.random.uniform(size=(2**N, 2**N)))

        Q, _ = np.linalg.qr(randM)
        rho = (Q * lamb).dot(Q.T.conj())

        return rho, rho

    def Get_uniform_state(self, N, Ns):  # PGD paper
        rho_all = []
        purity = []
        Ns_t = 0
        print('----begin obtain uniform state----')
        while Ns_t < Ns:
            purity_p = np.random.uniform(0, 1, 1)[0]
            _, rho = self.Get_random_state(N, purity_p)
            rho_all.append(rho)

            pur = np.trace(rho.dot(rho)).real
            purity.append(pur)
            Ns_t += 1
        print('----end obtain uniform state, number of state: {}----'.format(len(purity)))

        '''
        plt.hist(purity, 20)
        plt.show()'''

        return rho_all, purity