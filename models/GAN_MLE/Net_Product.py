# -*- coding: utf-8 -*-
# @Author: foxwy
# @Date:   2021-05-23 14:59:09
# @Last Modified by:   yong
# @Last Modified time: 2022-12-06 14:15:47

#--------------------libraries--------------------
#-----internal libraries-----
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from time import perf_counter
from tqdm import tqdm

import sys
sys.path.append('../..')

#-----external libraries-----
from Basis.Basic_Function import qmt_torch, qmt_torch_pure, proj_spectrahedron_torch, crack
from Basis.Loss_Function import MLE_loss, CF_loss, MLE_CF_loss
from Basis.Basis_State import Mea_basis


#--------------------class--------------------
#-----Fully Connected Net-----
class generator(nn.Module):
    def __init__(self, in_size, num_qubits, P_idxs, M, type_state='mixed', map_method='chol_h', P_proj=1.5):
        super(generator, self).__init__()

        #---parameter---
        if type_state == 'pure':
            self.out_size = 2**(num_qubits + 1)  # pure state
        elif type_state == 'mixed':  # mixed state
            self.out_size = 4**num_qubits
        print('out size:', self.out_size)           

        self.N = num_qubits
        self.P_idxs = P_idxs
        self.M = M
        self.device = M.device
        self.type_state = type_state
        self.map_method = map_method
        self.P_proj = P_proj

        #---net---
        out_size_log = 2 * num_qubits  # int(math.log(self.out_size, 2))

        self.net = nn.Sequential(
            nn.Linear(in_size, out_size_log), 
            nn.PReLU(),
            nn.Linear(out_size_log, self.out_size),
            nn.PReLU()
            )

    def weight_init(self, mean=0.0, std=0.0):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    def forward(self, X):
        out = self.net(X)
        if 'chol' in self.map_method:
            self.rho = self.Rho_T(out)
        elif 'proj' in self.map_method:
            self.rho = self.Rho_proj(out)
        P_out = self.Measure_rho()

        return P_out

    def Rho_T(self, T_array):  # factored space
        if self.type_state == 'pure':
            T = T_array.view(self.out_size, -1)
            T_a = T[:2**self.N].to(torch.complex64)
            T_i = T[2**self.N:]
            T_a += 1j * T_i

            rho = T_a / torch.norm(T_a)

        elif self.type_state == 'mixed':
            T_m = T_array.view(2**self.N, -1)
            T_triu = torch.triu(T_m, 1)
            T = torch.tril(T_m) + 1j * T_triu.T

            if self.map_method == 'chol_h':
                T += torch.tril(T, -1).T.conj()
            T_temp = torch.matmul(T.T.conj(), T)

            rho = T_temp / torch.trace(T_temp)

        return rho

    def Rho_proj(self, T_array):  # proj space
        if self.type_state == 'pure':
            T = T_array.view(self.out_size, -1)
            T_a = T[:2**self.N].to(torch.complex64)
            T_i = T[2**self.N:]
            T_a += 1j * T_i

            rho = T_a / torch.norm(T_a)

        elif self.type_state == 'mixed':
            T_m = T_array.view(2**self.N, -1)
            T_triu = torch.triu(T_m, 1)
            T = torch.tril(T_m) + 1j * T_triu.T
            #T += torch.tril(T, -1).T.conj()  # cause torch.linalg.eigh only use the lower triangular part of the matrix
            rho = proj_spectrahedron_torch(T, self.device, self.map_method, self.P_proj)

        return rho

    def Measure_rho(self):  # product
        self.rho = self.rho.to(torch.complex64)

        if self.type_state == 'pure':
            P_all = qmt_torch_pure(self.rho, [self.M] * self.N)
        else:
            P_all = qmt_torch(self.rho, [self.M] * self.N)

        P_real = P_all[self.P_idxs]

        return P_real.to(torch.float)


class Net_MLP():
    def __init__(self, generator, P_star, learning_rate=0.01):
        super().__init__

        self.generator = generator  # torch.compile(generator, mode="max-autotune")
        self.P_star = P_star

        self.optim = optim.Rprop(self.generator.parameters(), lr=learning_rate)

    def train(self, epochs, fid, result_save):
        print('\n'+'-'*20+'train'+'-'*20)
        self.sche = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=epochs, eta_min=0.)

        pbar = tqdm(range(epochs))
        epoch = 0
        time_all = 0
        for i in pbar:
            epoch += 1
            time_b = perf_counter()

            self.generator.train()
            self.optim.zero_grad()

            data = self.P_star
            P_out = self.generator(data)

            loss = CF_loss(P_out, data)
            assert torch.isnan(loss) == 0, print('loss is nan', loss)

            '''
            for group in self.optim.param_groups:
                for param in group["params"]:
                    if torch.isnan(param.grad).sum() > 0:
                        print('grad have nan, doing clip grad!')
                        param.grad.data.normal_(0, 1e-10)'''

            loss.backward()
            self.optim.step()
            self.sche.step()

            time_e = perf_counter()
            time_all += time_e - time_b
           
            # output
            if epoch % 2 == 0:
                self.generator.eval()
                with torch.no_grad():
                    Fc = fid.cFidelity_rho(self.generator.rho)
                    Fq = fid.Fidelity(self.generator.rho)

                    result_save['time'].append(time_all)
                    result_save['epoch'].append(epoch)
                    result_save['Fc'].append(Fc)
                    result_save['Fq'].append(Fq)
                    result_save['loss'].append(loss.item())
                    pbar.set_description("loss {:.8f} | Fc {:.8f} | Fq {:.8f}".format(
                                          loss.item(), Fc, Fq))

            '''
            if epoch == epochs:
                self.generator.eval()
                with torch.no_grad():
                    np.save('../../results/fig_M/Re_random_P9.npy', self.generator.rho.cpu().numpy())'''
            '''
            if loss.item() - MLE_loss(data, data).item() < 1e-10:  # finish training
                self.generator.eval()
                with torch.no_grad():
                    Fc = fid.cFidelity(P_out, P_real)

                    if Fc > 1 - 1e-5:
                        Fq = fid.Fidelity(self.generator.rho)
                        loss_df = loss.item() - MLE_loss(data, data).item()

                        result_save['time'].append(time_all)
                        result_save['epoch'].append(epoch)
                        result_save['Fc'].append(Fc)
                        result_save['Fq'].append(Fq)
                        result_save['loss'].append(loss.item())
                        result_save['loss_df'].append(loss_df)
                        pbar.set_description("loss {:.8f} | diff loss {:.8f} | Fc {:.8f} | Fq {:.8f}".format(
                                              loss.item(), loss_df, Fc, Fq))
                        
                        pbar.close()
                        print('final loss diff from optimal: {:.18f}'.format(loss_df))
                        break'''
        pbar.close()
            

#-----Condition GAN Net-----
class discriminator(nn.Module):
    def __init__(self, in_size):
        super(discriminator, self).__init__()

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        out_size = int(math.log(in_size, 2))
        self.fc1_1 = nn.Linear(in_size, out_size)
        self.fc1_2 = nn.Linear(in_size, out_size)

        self.net = nn.Sequential(*block(2 * out_size, out_size),
                                 *block(out_size, out_size),
                                 *block(out_size, out_size),
                                 nn.Linear(out_size, int(out_size / 2)))

    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    def forward(self, P_real, P_gen):
        x = F.leaky_relu(self.fc1_1(P_real), 0.2)
        y = F.leaky_relu(self.fc1_2(P_gen), 0.2)
        x = torch.cat([x, y], 0)
        x = self.net(x)

        return x


class Net_CGAN():
    def __init__(self, generator, discriminator, P_real):
        super(Net_CGAN, self).__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.P_real = P_real
        self.device = P_real.device

        #---optimizer---
        self.G_optim = optim.Adam(self.generator.parameters(), lr=0.02)
        self.G_sche = optim.lr_scheduler.ExponentialLR(self.G_optim, gamma=0.98)
        self.D_optim = optim.Adam(self.discriminator.parameters(), lr=0.05)

    def train(self, epochs, fid, lam=0.0):
        print('\n'+'-'*20+'train'+'-'*20)
        pbar = tqdm(range(epochs))
        epoch = 0
        for i in pbar:
            epoch += 1
            self.generator.train()
            self.discriminator.train()

            P_real = self.P_real.to(self.device)

            P_gen = self.generator(P_real)
            P_gen_de = P_gen.detach()
            disc_real = self.discriminator(P_real, P_real)
            disc_gen = self.discriminator(P_real, P_gen_de)

            D_loss = self.Discriminator_Loss(disc_real, disc_gen)
            self.D_optim.zero_grad()
            D_loss.backward(retain_graph=True)
            self.D_optim.step()

            disc_gen = self.discriminator(P_real, P_gen)
            G_loss = self.Generator_Loss(P_real, P_gen, disc_gen, lam)
            self.G_optim.zero_grad()
            G_loss.backward()
            self.G_optim.step()
            self.G_sche.step()

            # output
            if epoch % 100 == 0 or epoch == 1:
                self.generator.eval()
                self.discriminator.eval()
                with torch.no_grad():
                    rho = self.generator.rho.detach().cpu().numpy()
                    Fc = fid.cFidelity(P_gen, P_real)
                    Fq = fid.Fidelity(rho)
                    pbar.set_description("G loss {:.8f} | D loss {:.8f} | Fc {:.8f} | Fq {:.8f} | Is rho {}".format(
                                          G_loss.item(), D_loss.item(), Fc, Fq, fid.Is_rho(rho)))
        pbar.close()

    @staticmethod
    def Generator_Loss(P_real, P_gen, disc_gen, lam):
        return F.binary_cross_entropy_with_logits(disc_gen, torch.ones_like(disc_gen)) + \
               lam * torch.mean(torch.abs(P_real - P_gen))

    @staticmethod
    def Discriminator_Loss(disc_real, disc_gen):
        return F.binary_cross_entropy_with_logits(disc_real, torch.ones_like(disc_real)) + \
               F.binary_cross_entropy_with_logits(disc_gen, torch.zeros_like(disc_gen))


class Net_Conv(nn.Module):
    def __init__(self, in_size, num_qubits, P_idxs, M, type_state='mixed', map_method='chol_h', P_proj=1.5):
        super(Net_Conv, self).__init__()

        #---parameter---
        if type_state == 'pure':
            self.out_size = 2**(num_qubits + 1)  # pure state
        elif type_state == 'mixed':  # mixed state
            self.out_size = 4**num_qubits
        print('out size:', self.out_size)           

        self.N = num_qubits
        self.P_idxs = P_idxs.to(torch.long)
        self.M = M
        self.device = M.device
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
            self.rho = self.Rho_T(out)
        elif 'proj' in self.map_method:
            self.rho = self.Rho_proj(out)
        P_out = self.Measure_rho()

        return P_out

    def Rho_T(self, T_array):  # factored space
        if self.type_state == 'pure':
            T = T_array.view(self.out_size, -1)
            T_a = T[:2**self.N].to(torch.complex64)
            T_i = T[2**self.N:]
            T_a += 1j * T_i
            T_temp = torch.matmul(T_a, T_a.T.conj())

        elif self.type_state == 'mixed':
            T_m = T_array.view(2**self.N, -1)
            T_triu = torch.triu(T_m, 1)
            T = torch.tril(T_m) + 1j * T_triu.T

            if self.map_method == 'chol_h':
                T += torch.tril(T, -1).T.conj()
            T_temp = torch.matmul(T.T.conj(), T)

        rho = T_temp / torch.trace(T_temp)

        return rho

    def Rho_proj(self, T_array):  # proj space
        if self.type_state == 'pure':
            T = T_array.view(self.out_size, -1)
            T_a = T[:2**self.N].to(torch.complex64)
            T_i = T[2**self.N:]
            T_a += 1j * T_i
            T_temp = torch.matmul(T_a, T_a.T.conj())
            rho = T_temp / torch.trace(T_temp)

        elif self.type_state == 'mixed':
            T_m = T_array.view(2**self.N, -1)
            T_triu = torch.triu(T_m, 1)
            T = torch.tril(T_m) + 1j * T_triu.T
            #T += torch.tril(T, -1).T.conj()  # cause torch.linalg.eigh only use the lower triangular part of the matrix
            rho = proj_spectrahedron_torch(T, self.device, self.map_method, self.P_proj)

        return rho

    def Measure_rho(self):  # product
        self.rho = self.rho.to(torch.complex64)

        P_all = qmt_torch(self.rho, [self.M] * self.N)
        P_real = P_all[self.P_idxs]

        return P_real.to(torch.float)


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
