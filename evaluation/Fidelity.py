# --------------------libraries--------------------
# internal libraries
import numpy as np
import torch
#import tensorly as tl
#from tensorly.decomposition import matrix_product_state

# external libraries
import sys
sys.path.append('..')

from Basis.Basic_Function import qmt
from Basis.Basis_State import Mea_basis


# -----Fid-----
class Fid(Mea_basis):
    def __init__(self, basis, n_qubits, rho_star, device='cpu', torch_flag=1):
        super().__init__(basis)
        self.N = n_qubits
        if torch_flag == 1:
            self.rho_star = torch.from_numpy(rho_star).to(device).to(torch.complex64)
        else:
            self.rho_star = rho_star

    # ----------fidelity----------
    def Fidelity(self, rho):
        Fq = torch.tensor(0)
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

    def get_real_p(self, S):
        idxs = S.dot(self.K**(np.arange(self.N - 1, -1, -1)))
        if type(self.rho_star) is np.ndarray:
            P_all = qmt(self.rho_star, [self.M] * self.N)
        elif type(self.rho_star) is torch.Tensor:
            P_all = qmt(self.rho_star.cpu().numpy(), [self.M] * self.N)
        P_real = P_all[idxs]

        return P_real

    def cFidelity_S_product(self, S, P_f, unique_flag=1):  # fast!!!
        P_real = self.get_real_p(S)
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
        if type(self.rho_star) is np.ndarray:
            P_real = qmt(self.rho_star, [self.M] * self.N)
            P_f = qmt(rho, [self.M] * self.N)
        elif type(self.rho_star) is torch.Tensor:
            P_real = qmt(self.rho_star.cpu().numpy(), [self.M] * self.N)
            P_f = qmt(rho.cpu().numpy(), [self.M] * self.N)
        Fc = self.cFidelity(P_f, P_real)

        return Fc


# --------------------main--------------------
if __name__ == '__main__':
    pass
