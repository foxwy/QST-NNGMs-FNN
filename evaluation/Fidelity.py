# --------------------libraries--------------------
# internal libraries
import numpy as np
import torch
#import tensorly as tl
#from tensorly.decomposition import matrix_product_state

# external libraries
import sys
sys.path.append('..')

from Basis.Basic_Function import qmt, qmt_pure, qmt_torch, qmt_torch_pure
from Basis.Basis_State import Mea_basis


# -----Fid-----
class Fid(Mea_basis):
    def __init__(self, basis, n_qubits, ty_state, rho_star, M=None, device='cpu'):
        super().__init__(basis)
        self.N = n_qubits
        self.ty_state = ty_state

        if M is not None:
            self.M = M

        self.rho_star = rho_star

        self.P_all = self.get_real_p(self.rho_star)

    # ----------fidelity----------
    def Fidelity(self, rho):
        Fq = torch.tensor(0)
        if self.ty_state == 'pure':
            tmp = torch.matmul(self.rho_star.T.conj(), rho)[0, 0]
            Fq = (tmp * tmp.conj()).real
        else:
            #try:
            eigenvalues, eigenvecs = torch.linalg.eigh(self.rho_star)
            eigenvalues = torch.abs(eigenvalues)
            sqrt_rho = torch.matmul(eigenvecs * torch.sqrt(eigenvalues), eigenvecs.T.conj())  # sqrtm(self.rho_star)
            rho_tmp = torch.matmul(torch.matmul(sqrt_rho, rho), sqrt_rho)  # sqrtm(self.rho_star).dot(rho).dot(sqrtm(self.rho_star))

            eigenvalues = torch.linalg.eigvalsh(rho_tmp)
            sqrt_eigvals = torch.sqrt(torch.abs(eigenvalues))
            Fq = torch.sum(sqrt_eigvals)**2  # trace(sqrtm(sqrtm(self.rho_star).dot(rho).dot(sqrtm(self.rho_star))))**2

        if Fq > 1:
            Fq = 1  # precision error

        #except Exception:
        #    print('error:', rho)

        return Fq

    def get_real_p(self, rho):
        if type(rho) is np.ndarray:
            if self.ty_state == 'pure':
                P_all = qmt_pure(rho, [self.M] * self.N)
            else:
                P_all = qmt(rho, [self.M] * self.N)

        elif type(rho) is torch.Tensor:
            if self.ty_state == 'pure':
                P_all = qmt_torch_pure(rho, [self.M] * self.N)
            else:
                P_all = qmt_torch(rho, [self.M] * self.N)

        return P_all

    def cFidelity_S_product(self, P_idxs, P_f, unique_flag=1):  # fast!!!
        P_real = self.P_all[P_idxs]
        Fc = self.cFidelity(P_f, P_real, unique_flag)

        return Fc

    @staticmethod
    def cFidelity(P_f, P_real, unique_flag=1):
        if type(P_f) is np.ndarray:
            if unique_flag == 1:  # has not repeat elements
                Fc = np.sqrt(P_real).dot(np.sqrt(P_f))
            else:
                Fc = np.sqrt(P_real).dot(1 / np.sqrt(P_f)) / len(P_real)

            return Fc**2
            
        elif type(P_f) is torch.Tensor:
            Fc = torch.sqrt(P_f).dot(torch.sqrt(P_real))

            return Fc.item()**2

    def cFidelity_rho(self, rho):
        P_f = self.get_real_p(rho)
        Fc = self.cFidelity(P_f, self.P_all)

        return Fc


# --------------------main--------------------
if __name__ == '__main__':
    pass
