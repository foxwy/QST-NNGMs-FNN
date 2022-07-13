# -*- coding: utf-8 -*-
# @Author: foxwy
# @Date:   2021-01-19 15:38:06
# @Last Modified by:   WY
# @Last Modified time: 2021-09-30 14:44:09

#--------------------libraries--------------------
# internal libraries
import numpy as np
import matplotlib.pyplot as plt

# external libraries
import sys
sys.path.append('..')

from Basis.Basic_Function import semidefinite_adjust


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

    @staticmethod
    def Rho_info(rho):
        print('rho:\n', rho)
        print('rho+:\n', rho.T.conjugate())
        print('rho+=rho\n', rho.T.conjugate() == rho)
        print('Tr(rho)\n', np.trace(rho))

        semidefinite_adjust(rho, 'rho')

    @staticmethod
    def Is_rho(rho):
        if abs(np.trace(rho) - 1) < 1e-6 and np.all(np.abs(rho - rho.T.conjugate())) < 1e-4 and semidefinite_adjust(rho):
            return 1
        else:
            if abs(np.trace(rho) - 1) > 1e-6:
                print('trace of rho is not 1')
            if np.all(np.abs(rho - rho.T.conjugate())) > 1e-4:
                print('rho is not Hermitian')
            if not semidefinite_adjust(rho):
                print('rho is not positive semidefine')
            return 0


#-----Mea_basis-----
class Mea_basis(State):
    def __init__(self, basis='Tetra'):
        super().__init__()
        self.basis = basis
        self.Get_basis()

    def Get_basis(self):
        # operators
        if self.basis == 'Tetra':
            self.K = 4

            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)

            v1 = np.array([0, 0, 1.0])
            self.M[0, :, :] = 1.0 / 4.0 * \
                (self.I + v1[0] * self.s1 + v1[1] * self.s2 + v1[2] * self.s3);

            v2 = np.array([2.0 * np.sqrt(2.0) / 3.0, 0.0, -1.0 / 3.0])
            self.M[1, :, :] = 1.0 / 4.0 * \
                (self.I + v2[0] * self.s1 + v2[1] * self.s2 + v2[2] * self.s3);

            v3 = np.array(
                [-np.sqrt(2.0) / 3.0, np.sqrt(2.0 / 3.0), -1.0 / 3.0])
            self.M[2, :, :] = 1.0 / 4.0 * \
                (self.I + v3[0] * self.s1 + v3[1] * self.s2 + v3[2] * self.s3);

            v4 = np.array(
                [-np.sqrt(2.0) / 3.0, -np.sqrt(2.0 / 3.0), -1.0 / 3.0])
            self.M[3, :, :] = 1.0 / 4.0 * \
                (self.I + v4[0] * self.s1 + v4[1] * self.s2 + v4[2] * self.s3);

        elif self.basis == 'Tetra4':
            self.K = 4

            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)

            self.M[0, :, :] = 0.25 * (self.I + (self.X + self.Y + self.Z) / np.sqrt(3))
            self.M[1, :, :] = 0.25 * (self.I + (-self.X - self.Y + self.Z) / np.sqrt(3))
            self.M[2, :, :] = 0.25 * (self.I + (-self.X + self.Y - self.Z) / np.sqrt(3))
            self.M[3, :, :] = 0.25 * (self.I + (self.X - self.Y - self.Z) / np.sqrt(3))

        elif self.basis == '6Pauli':
            self.K = 6

            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)

            self.M[0, :, :] = (self.I + self.X) / 6
            self.M[1, :, :] = (self.I - self.X) / 6
            self.M[2, :, :] = (self.I + self.Y) / 6
            self.M[3, :, :] = (self.I - self.Y) / 6
            self.M[4, :, :] = (self.I + self.Z) / 6
            self.M[5, :, :] = (self.I - self.Z) / 6

        elif self.basis == '4Pauli':  # different from paper
            self.K = 4

            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)

            self.M[0, :, :] = 1.0 / 3.0 * np.array([[1, 0], [0, 0]])
            self.M[1, :, :] = 1.0 / 6.0 * np.array([[1, 1], [1, 1]])
            self.M[2, :, :] = 1.0 / 6.0 * np.array([[1, -1j], [1j, 1]])
            self.M[3, :, :] = 1.0 / 3.0 * (np.array([[0, 0], [0, 1]]) +
                                           0.5 * np.array([[1, -1], [-1, 1]])
                                           + 0.5 * np.array([[1, 1j], [-1j, 1]]))

        elif self.basis == 'Pauli':
            self.K = 6
            Ps = np.array([1. / 3., 1. / 3., 1. / 3.,
                           1. / 3., 1. / 3., 1. / 3.])
            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)
            theta = np.pi / 2.0
            self.M[0, :, :] = Ps[0] * self.pXp(theta, 0.0)
            self.M[1, :, :] = Ps[1] * self.mXm(theta, 0.0)
            self.M[2, :, :] = Ps[2] * self.pXp(theta, np.pi / 2.0)
            self.M[3, :, :] = Ps[3] * self.mXm(theta, np.pi / 2.0)
            self.M[4, :, :] = Ps[4] * self.pXp(0.0, 0.0)
            self.M[5, :, :] = Ps[5] * self.mXm(0, 0.0)

        elif self.basis == 'Pauli_rebit':  # X
            self.K = 4
            Ps = np.array([1. / 2., 1. / 2., 1. / 2., 1. / 2.])
            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)
            theta = np.pi / 2.0
            self.M[0, :, :] = Ps[0] * self.pXp(theta, 0.0)
            self.M[1, :, :] = Ps[1] * self.mXm(theta, 0.0)
            self.M[2, :, :] = Ps[2] * self.pXp(0.0, 0.0)
            self.M[3, :, :] = Ps[3] * self.mXm(0, 0.0)
            self.M = self.M.real

        elif self.basis == 'Pauli_6':
            self.K = 6
            Ps = np.array([1. / 3., 1. / 6., 1. / 2.])
            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)
            self.M[0, :, :] = Ps[0] * np.array([[1, 0], [0, 0]])
            self.M[1, :, :] = Ps[0] * np.array([[0, 0], [0, 1]])
            self.M[2, :, :] = Ps[1] / 2 * np.array([[1, 1], [1, 1]])
            self.M[3, :, :] = Ps[1] / 2 * np.array([[1, -1], [-1, 1]])
            self.M[4, :, :] = Ps[2] / 2 * np.array([[1, -1j], [1j, 1]])
            self.M[5, :, :] = Ps[2] / 2 * np.array([[1, 1j], [-1j, 1]])

        elif self.basis == 'Trine':
            self.K = 3
            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)
            phi0 = 0.0
            for k in range(self.K):
                phi = phi0 + (k) * 2 * np.pi / 3.0
                self.M[k, :, :] = 0.5 * (self.I + np.cos(phi)
                                         * self.Z + np.sin(phi) * self.X) * 2 / 3.0
            self.M = self.M.real

        elif self.basis == 'Psi2':
            self.K = 2
            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)
            self.M[0, 0, 0] = 1
            self.M[1, 1, 1] = 1

        elif self.basis == 'Pauli_normal':  # projective measurement, not POVM
            self.K = 4
            self.M = np.zeros((self.K, 2, 2), dtype=np.complex64)
            self.M[0, :, :] = np.array([[1.0, 0.0], [0.0, 1.0]])
            self.M[1, :, :] = np.array([[0.0, 1.0], [1.0, 0.0]])
            self.M[2, :, :] = np.array([[0.0, -1j], [1j, 0.0]])
            self.M[3, :, :] = np.array([[1.0, 0.0], [0.0, -1.0]])

    def Basis_info(self):
        print('K:\n', self.K)
        print('M:\n', self.M)
        print('M+:\n', [self.M[i].T.conjugate() for i in range(len(self.M))])
        print('M+=M:\n', [self.M[i].T.conjugate() == self.M[i]
                          for i in range(len(self.M))])
        print('sum(M):\n', sum([self.M[i] for i in range(len(self.M))]))

        for M in self.M:
            semidefinite_adjust(M, 'M')

        '''
        it, it_cond = Cal_cond(self.M)
        print('it:\n', it)
        semidefinite_adjust(it, 'it')'''

    @staticmethod
    def pXp(theta, phi):

        return np.array([[np.cos(theta / 2.0)**2, np.cos(theta / 2.0) * np.sin(theta / 2.0) * np.exp(-1j * phi)],
                         [np.cos(theta / 2.0) * np.sin(theta / 2.0) * np.exp(1j * phi), np.sin(theta / 2.0)**2]])

    @staticmethod
    def mXm(theta, phi):

        return np.array([[np.sin(theta / 2.0)**2, -np.cos(theta / 2.0) * np.sin(theta / 2.0) * np.exp(-1j * phi)],
                         [-np.cos(theta / 2.0) * np.sin(theta / 2.0) * np.exp(1j * phi), np.cos(theta / 2.0)**2]])


#--------------------main--------------------
if __name__ == '__main__':
    '''
    B = Mea_basis('Tetra')  # Tetra, 4Pauli, Pauli, Trine, Psi2
    #print(B.K, B.M)
    B.Basis_info()'''

    #print(State().Get_state_rho('GHZi_P', 2, 0.5))

    #MPS_state().MPS_info()

    for n in range(2, 11):
        rho_all, _ = State().Get_uniform_state(n, 200)
        np.save('../datasets/uniform_state/N'+str(n)+'.npy', rho_all)