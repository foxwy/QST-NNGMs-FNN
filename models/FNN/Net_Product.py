# -*- coding: utf-8 -*-
# @Author: foxwy
# @Date:   2021-05-23 14:59:09
# @Last Modified by:   yong
# @Last Modified time: 2022-12-23 21:30:10
# @Function: FNN, CNN, and training
# @Paper: Ultrafast quantum state tomography with feed-forward neural networks

import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter
from tqdm import tqdm

#torch.set_default_dtype(torch.float32)

sys.path.append('../..')
from Basis.Basic_Function import qmt_torch, qmt_torch_pure, proj_spectrahedron_torch, crack
from Basis.Loss_Function import MLE_loss, CF_loss
from Basis.Basis_State import Mea_basis


class generator(nn.Module):
    """
    Feedforward neural networks are used to perform quantum state tomography tasks, 
    mapping measured probability distributions to density matrix and measuring the 
    distance from the probability distribution to optimize the network parameters, 
    see paper ``Ultrafast quantum state tomography with feed-forward neural networks``.

    Examples::
        see ``FNN/FNN_learn``.
    """
    def __init__(self, in_size, 
                       num_qubits, 
                       P_idxs, 
                       M, 
                       type_state='mixed', 
                       map_method='chol_h', 
                       P_proj=1.5, 
                       net_type='learn', 
                       device='cpu'):
        """
        Args:
            in_size (int): Input size of the network.
            num_qubits (int): The number of qubits.
            P_idxs (tensor): Index of the POVM used for measurement, Not all measurements 
                are necessarily used.
            M (tensor): The POVM, size (K, 2, 2).
            type_state (str): The type of state, include 'mixed' and 'pure'.
            map_method (str): State-mapping method, include ['chol', 'chol_h', 'proj_F', 'proj_S', 'proj_A'].
            P_proj (float): P order.
            net_type (str): The network types, here divided into ``train`` and ``learn``, are the same as in the paper.
            device (torch.device): GPU or CPU. 
        """
        super(generator, self).__init__()

        if type_state == 'pure':
            self.out_size = 2**(num_qubits + 1)  # pure state
        elif type_state == 'mixed':  # mixed state
            self.out_size = 4**num_qubits
        print('out size:', self.out_size)           

        self.N = num_qubits
        self.P_idxs = P_idxs
        self.M = M
        self.device = device
        self.type_state = type_state
        self.map_method = map_method
        self.P_proj = P_proj
        self.net_type = net_type

        #---net---
        if net_type == 'learn':
            out_size_log = 2 * num_qubits  # direct learning
        else:
            out_size_log = 200 * num_qubits  # pre-training

        self.net = nn.Sequential(
            nn.Linear(in_size, out_size_log), 
            nn.PReLU(),
            nn.Linear(out_size_log, self.out_size)
            )

    def forward(self, X):
        """
        In case of direct learning, the network output needs to be 
        mapped to a density matrix, or output directly.
        """
        if self.net_type == 'train':    
            d_out = self.net(X)

        elif self.net_type == 'learn':
            out_all = self.net(X)
            if 'chol' in self.map_method:
                self.rho = self.Rho_T(out_all)  # decomposition
            elif 'proj' in self.map_method:
                self.rho = self.Rho_proj(out_all)  # projection
            d_out = self.Measure_rho()  # perfect measurement

        else:
            print('please input right net type!')
        return d_out

    def Rho_T(self, T_array):
        """decomposition"""
        if self.type_state == 'pure':  # pure state
            T = T_array.view(self.out_size, -1)
            T_a = T[:2**self.N].to(torch.complex64)
            T_i = T[2**self.N:]
            T_a += 1j * T_i

            rho = T_a / torch.norm(T_a)

        elif self.type_state == 'mixed':  # mixed state
            T_m = T_array.view(2**self.N, -1)
            T_triu = torch.triu(T_m, 1)
            T = torch.tril(T_m) + 1j * T_triu.T

            if self.map_method == 'chol_h':
                T += torch.tril(T, -1).T.conj()
            T_temp = torch.matmul(T.T.conj(), T)

            rho = T_temp / torch.trace(T_temp)
        return rho.to(torch.complex64)

    def Rho_proj(self, T_array):
        """projection"""
        if self.type_state == 'pure':  # pure state
            T = T_array.view(self.out_size, -1)
            T_a = T[:2**self.N].to(torch.complex64)
            T_i = T[2**self.N:]
            T_a += 1j * T_i

            rho = T_a / torch.norm(T_a)

        elif self.type_state == 'mixed':  # mixed state
            T_m = T_array.view(2**self.N, -1)
            T_triu = torch.triu(T_m, 1)
            T = torch.tril(T_m) + 1j * T_triu.T
            #T += torch.tril(T, -1).T.conj()  # cause torch.linalg.eigh only use the lower triangular part of the matrix
            rho = proj_spectrahedron_torch(T, self.device, self.map_method, self.P_proj)
        return rho.to(torch.complex64)

    def Measure_rho(self):
        """perfect measurement"""
        if self.type_state == 'pure':  # pure state
            P_all = qmt_torch_pure(self.rho, [self.M] * self.N)
        else:  # mixed state
            P_all = qmt_torch(self.rho, [self.M] * self.N)

        P_real = P_all[self.P_idxs]
        return P_real


class Net_MLP():
    """
    For network training for direct learning.

    Examples::
        see ``FNN/FNN_learn``.
    """
    def __init__(self, generator, P_star, learning_rate=0.01):
        """
        Args:
            generator (generator): The network used for training.
            P_star (tensor): Probability distribution data from experimental measurements.
            learning_rate (float): Learning rate of the optimizer.

        Net setups:
            Optimizer: Rpop.
            Loss: CF_loss in ``Basis/Loss_Function``.
        """
        super().__init__

        self.generator = generator  # torch.compile(generator, mode="max-autotune")
        self.P_star = P_star

        self.optim = optim.Rprop(self.generator.parameters(), lr=learning_rate)
        #self.criterion = nn.MSELoss()

    def train(self, epochs, fid, result_save):
        """Net training"""
        print('\n'+'-'*20+'train'+'-'*20)
        #self.sche = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=epochs, eta_min=0.)
        #self.sche = optim.lr_scheduler.LinearLR(self.optim, start_factor=0.1, total_iters=epochs)

        pbar = tqdm(range(epochs))
        epoch = 0
        time_all = 0
        for i in pbar:
            epoch += 1
            time_b = perf_counter()

            self.generator.train()
            data = self.P_star
            self.optim.zero_grad()
            P_out = self.generator(data)
            loss = CF_loss(P_out, data)
            #loss = self.criterion(P_out, data)
            assert torch.isnan(loss) == 0, print('loss is nan', loss)

            '''
            for group in self.optim.param_groups:
                for param in group["params"]:
                    if torch.isnan(param.grad).sum() > 0:
                        print('grad have nan, doing clip grad!')
                        param.grad.data.normal_(0, 1e-10)'''

            loss.backward()
            self.optim.step()
            #self.sche.step()

            time_e = perf_counter()
            time_all += time_e - time_b
           
            # show and save
            if epoch % 2 == 0:
                self.generator.eval()
                with torch.no_grad():
                    Fc = fid.cFidelity_rho(self.generator.rho)
                    Fq = fid.Fidelity(self.generator.rho)

                    result_save['time'].append(time_all)
                    result_save['epoch'].append(epoch)
                    result_save['Fc'].append(Fc)
                    result_save['Fq'].append(Fq)
                    pbar.set_description("NN --loss {:.8f} | Fc {:.8f} | Fq {:.8f} | time {:.4f} | epochs {:d}".format(loss.item(), Fc, Fq, time_all, epoch))

                    if Fq >= 0.99:
                        break

        pbar.close()


class Net_MLP_train():
    """
    For network training for pre-traing.

    Examples::
        see ``FNN/FNN_learn``.
    """
    def __init__(self, generator, learning_rate=0.01):
        """
        Args:
            generator (generator): The network used for pre-training.
            learning_rate (float): Learning rate of the optimizer.

        Net setups:
            Optimizer: Adam.
            Loss: MSE loss.
            Scheduler: CosineAnnealingLR.
        """
        super().__init__

        self.generator = generator  # torch.compile(generator, mode="max-autotune")
        self.criterion = nn.MSELoss()
        self.optim = optim.Adam(self.generator.parameters(), lr=learning_rate)

    def train(self, trainloader, testloader, epochs, device):
        """Net training"""
        print('\n'+'-'*20+'train'+'-'*20)
        self.sche = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=epochs, eta_min=0.)
        #self.sche = optim.lr_scheduler.LinearLR(self.optim, start_factor=0.1, total_iters=epochs)

        pbar = tqdm(range(epochs))
        epoch = 0
        time_all = 0
        test_loss_min = 1e10
        for i in pbar:
            epoch += 1
            time_b = perf_counter()

            # ----train----
            self.generator.train()
            train_loss = 0
            for idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(torch.float32).to(device), targets.to(torch.float32).to(device)
                self.optim.zero_grad()
                outputs = self.generator(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optim.step()
                self.sche.step()

                train_loss += loss.item()

            time_e = perf_counter()
            time_all += time_e - time_b         

            # ----test----
            self.generator.eval()
            test_loss = 0
            with torch.no_grad():
                for idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(torch.float32).to(device), targets.to(torch.float32).to(device)
                    outputs = self.generator(inputs)
                    loss = self.criterion(outputs, targets)

                    test_loss += loss.item()
                
            pbar.set_description("train loss {:.12f} | test loss {:.12f} | time {:.4f} | epochs {:d}".format(train_loss, test_loss, time_all, i))

            if i == 0:
                test_loss_min = test_loss
            else:
                if test_loss_min > test_loss:
                    test_loss_min = test_loss
                    torch.save(self.generator.state_dict(), 'model.pt')

        pbar.close()
            

class Net_Conv(nn.Module):
    """
    Convolutional neural networks are used to perform quantum state tomography tasks, 
    mapping measured probability distributions to density matrix and measuring the 
    distance from the probability distribution to optimize the network parameters, 
    see paper ``Ultrafast quantum state tomography with feed-forward neural networks``.

    Examples::
        see ``FNN/FNN_learn``.
    """
    def __init__(self, in_size, 
                       num_qubits, 
                       P_idxs, 
                       M, 
                       type_state='mixed', 
                       map_method='chol_h', 
                       P_proj=1.5, 
                       device='cpu'):
        """
        Args:
            in_size (int): Input size of the network.
            num_qubits (int): The number of qubits.
            P_idxs (tensor): Index of the POVM used for measurement, Not all measurements 
                are necessarily used.
            M (tensor): The POVM, size (K, 2, 2).
            type_state (str): The type of state, include 'mixed' and 'pure'.
            map_method (str): State-mapping method, include ['chol', 'chol_h', 'proj_F', 'proj_S', 'proj_A'].
            P_proj (float): P order.
            device (torch.device): GPU or CPU. 
        """
        super(Net_Conv, self).__init__()

        if type_state == 'pure':
            self.out_size = 2**(num_qubits + 1)  # pure state
        elif type_state == 'mixed':  # mixed state
            self.out_size = 4**num_qubits
        print('out size:', self.out_size)           

        self.N = num_qubits
        self.P_idxs = P_idxs
        self.M = M
        self.device = device
        self.type_state = type_state
        self.map_method = map_method
        self.P_proj = P_proj

        #---net---
        # change list into matrix (figure)
        out_size_log = 4 * num_qubits  # int(math.log(self.out_size, 2))
        self.in_size_row, self.in_size_column = crack(in_size)
        print('Conv in size, out size:', self.in_size_row, self.in_size_column)

        if num_qubits <= 3:
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
                )
            '''
            self.conv2 = nn.Sequential(
                nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(8)
                )'''

            self.fc = nn.Sequential(
                nn.Linear(10 * int(self.in_size_row/2) * int(self.in_size_column/2), out_size_log),
                nn.ReLU(),
                nn.Linear(out_size_log, self.out_size)
                )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2**(num_qubits-3))
                )
            '''
            self.conv2 = nn.Sequential(
                nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(8)
                )'''

            self.fc = nn.Sequential(
                nn.Linear(10 * int(self.in_size_row/2**(num_qubits-3)) * int(self.in_size_column/2**(num_qubits-3)), out_size_log),
                nn.ReLU(),
                nn.Linear(out_size_log, self.out_size)
                )

    def forward(self, X):
        out = X.view(1, 1, self.in_size_row, self.in_size_column)
        out = self.conv1(out)
        #out = self.conv2(out)
        out = out.view(1, -1)
        out = self.fc(out)

        if 'chol' in self.map_method:
            self.rho = self.Rho_T(out)  # decomposition
        elif 'proj' in self.map_method:
            self.rho = self.Rho_proj(out)  # projection
        P_out = self.Measure_rho()  # perfect measurement
        return P_out

    def Rho_T(self, T_array):
        """decomposition"""
        if self.type_state == 'pure':  # pure state
            T = T_array.view(self.out_size, -1)
            T_a = T[:2**self.N].to(torch.complex64)
            T_i = T[2**self.N:]
            T_a += 1j * T_i

            rho = T_a / torch.norm(T_a)

        elif self.type_state == 'mixed':  # mixed state
            T_m = T_array.view(2**self.N, -1)
            T_triu = torch.triu(T_m, 1)
            T = torch.tril(T_m) + 1j * T_triu.T

            if self.map_method == 'chol_h':
                T += torch.tril(T, -1).T.conj()
            T_temp = torch.matmul(T.T.conj(), T)

            rho = T_temp / torch.trace(T_temp)
        return rho.to(torch.complex64)

    def Rho_proj(self, T_array):
        """projection"""
        if self.type_state == 'pure':  # pure state
            T = T_array.view(self.out_size, -1)
            T_a = T[:2**self.N].to(torch.complex64)
            T_i = T[2**self.N:]
            T_a += 1j * T_i

            rho = T_a / torch.norm(T_a)

        elif self.type_state == 'mixed':  # mixed state
            T_m = T_array.view(2**self.N, -1)
            T_triu = torch.triu(T_m, 1)
            T = torch.tril(T_m) + 1j * T_triu.T
            #T += torch.tril(T, -1).T.conj()  # cause torch.linalg.eigh only use the lower triangular part of the matrix
            rho = proj_spectrahedron_torch(T, self.device, self.map_method, self.P_proj)
        return rho.to(torch.complex64)

    def Measure_rho(self):
        """perfect measurement"""
        if self.type_state == 'pure':  # pure state
            P_all = qmt_torch_pure(self.rho, [self.M] * self.N)
        else:  # mixed state
            P_all = qmt_torch(self.rho, [self.M] * self.N)

        P_real = P_all[self.P_idxs]
        return P_real


if __name__ == "__main__":
    B = Mea_basis('Tetra4')
    #rho = (B.I + 0.5*B.X + np.sqrt(3)/2*B.Y)/2
    s, rho = B.Get_state_rho('W', 10)
    #print('rho:', rho)
    #print(B.M)
    t1 = perf_counter()
    rho = torch.tensor(rho).to(torch.complex64)
    P = qmt_torch(rho, [torch.tensor(B.M).to(torch.complex64)]*10)
    t2 = perf_counter()
    print(t2-t1)
    print(P, sum(P))
