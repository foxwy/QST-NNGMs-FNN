# -*- coding: utf-8 -*-
# @Author: foxwy
# @Date:   2021-05-20 18:58:08
# @Last Modified by:   yong
# @Last Modified time: 2022-12-06 14:21:12

# --------------------libraries--------------------
# -----internal libraries-----
import os
import sys
import argparse

import torch
import numpy as np

# -----environment-----
filepath = os.path.abspath(os.path.join(os.getcwd(), '../..'))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# -----external libraries-----
sys.path.append('../..')

from models.GAN_MLE.Net_Product import generator, Net_MLP, discriminator, Net_CGAN, Net_Conv
from Basis.Basis_State import Mea_basis, State
from datasets.dataset import Dataset_P, Dataset_sample, Dataset_sample_P
from evaluation.Fidelity import Fid


def filenameGenerate(base, opt):
    args = {"P_state-": opt.P_state,
            "--n_qubits-": opt.n_qubits,
            "--ty_state-": opt.ty_state,
            "--noise-": opt.noise,
            "--n_samples-": opt.n_samples, 
            "--P_povm-": opt.P_povm, 
            "--seed_povm-": opt.seed_povm,
            "--ty_data-": opt.ty_data}

    name = base + "".join([key + str(args[key]) for key in args])

    return name


def Net_train(opt, device):
    torch.cuda.empty_cache()
    print('\nparameter:', opt)

    # ----------file----------
    '''
    r_path = '../../results/FNN/result/' + opt.na_state + '/'
    if os.path.isdir(r_path):
        print('result dir exists, is: ' + r_path)
    else:
        os.makedirs(r_path)
        print('result dir not exists, has been created, is: ' + r_path)

    filename = filenameGenerate("", opt)
    savePath = r_path + filename'''

    # ----------data----------
    print('\n'+'-'*20+'data'+'-'*20)
    state_star, rho_star = State().Get_state_rho(opt.na_state, opt.n_qubits, opt.P_state)
    if opt.ty_state == 'pure':  # pure state
        rho_star = state_star

    if opt.noise == 'depolar_noise':
        _, rho_star = State().Get_state_rho(opt.na_state, opt.n_qubits, 1 - opt.P_state)
        _, rho_o = State().Get_state_rho(opt.na_state, opt.n_qubits, 1)
    #np.save('../../results/fig_M/'+opt.na_state+str(int(opt.P_state*10))+'.npy', rho_star)
    
    rho_star = torch.from_numpy(rho_star).to(torch.complex64).to(device)
    M = Mea_basis(opt.POVM).M
    M = torch.from_numpy(M).to(device)

    print('read original data')
    if opt.noise == "no_noise":
        print('----read ideal data')
        P_idxs, data = Dataset_P(rho_star, M, opt.n_qubits, opt.K, opt.ty_state, opt.P_povm, opt.seed_povm)
    else:
        print('----read sample data')
        if opt.P_povm == 1:
            P_idxs, data = Dataset_sample(opt.POVM, opt.na_state, opt.n_qubits, opt.n_samples, opt.P_state, opt.ty_state, rho_star, M, opt.read_data)  # 全测量采样'''
        else:
            P_idxs, data = Dataset_sample_P(opt.POVM, opt.na_state, opt.n_qubits, opt.K, opt.n_samples, opt.P_state, opt.ty_state, rho_star, opt.read_data, \
                                                 opt.P_povm, opt.seed_povm)
    
    # print(data_unique, data)
    in_size = len(data)
    print('data shape:', in_size)

    # fidelity
    if opt.noise == 'depolar_noise':
        fid = Fid(basis=opt.POVM, n_qubits=opt.n_qubits, ty_state=opt.ty_state, rho_star=rho_o, M=M, device=device)
    else:
        fid = Fid(basis=opt.POVM, n_qubits=opt.n_qubits, ty_state=opt.ty_state, rho_star=rho_star, M=M, device=device)
    #print('data', data)
    CF = fid.cFidelity_S_product(P_idxs, data)
    print('classical fidelity:', CF)


    # ----------Net----------
    print('\n'+'-'*20+'train'+'-'*20)
    #---FNN---
    gen_net = generator(in_size, opt.n_qubits, P_idxs, M, type_state=opt.ty_state, map_method=opt.map_method, P_proj=opt.P_proj).to(device)
    gen_net.weight_init(0, 0.01)

    #---CNN---
    #gen_net = Net_Conv(in_size, opt.n_qubits, P_idxs, M, type_state=opt.ty_state, map_method=opt.map_method, P_proj=opt.P_proj).to(device)

    total_param = sum(param.numel() for param in gen_net.parameters())
    print(gen_net, '\nnumber of params: %0.5fM'%(total_param/1e6))

    net = Net_MLP(gen_net, data, opt.lr)
    result_save = {'parser': opt,
                   'time': [], 
                   'epoch': [],
                   'Fc': [],
                   'Fq': [],
                   'loss': []}
    net.train(opt.n_epochs, fid, result_save)
    #np.save(savePath + '.npy', result_save)

    return result_save

 
    '''
    print('\n'+'-'*20+'net'+'-'*20)
    gen_net = generator(in_size, opt.n_qubits, P_idxs, M, type_state=opt.ty_state).to(device)
    gen_net.weight_init(0, 0.01)
    disc_net = discriminator(in_size).to(device)
    disc_net.weight_init(0, 0.01)
    print(gen_net)
    print(disc_net)
    net = Net_CGAN(gen_net, disc_net, data)
    net.train(opt.n_epochs, fid, 0)'''

    #np.save(state_name+str(opt.n_qubits)+'_'+str(opt.rnn_epochs)+'.npy', net.rho.cpu().detach().numpy())


def get_default_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


# --------------------main--------------------
if __name__ == '__main__':
    # ----------device----------
    print('-'*20+'init'+'-'*20)
    default_device = get_default_device()
    device = torch.device(default_device)
    print('device:', device)

    # ----------parameters----------
    print('-'*20+'set parser'+'-'*20)
    parser = argparse.ArgumentParser()
    parser.add_argument("--POVM", type=str, default="Tetra4", help="type of POVM")
    parser.add_argument("--K", type=int, default=4, help='number of operators in single-qubit POVM')

    parser.add_argument("--na_state", type=str, default="GHZi_P", help="name of state in library")
    parser.add_argument("--P_state", type=float, default=1, help="P of mixed state")
    parser.add_argument("--ty_state", type=str, default="mixed", help="type of state (pure, mixed)")
    parser.add_argument("--n_qubits", type=int, default=11, help="number of qubits")

    parser.add_argument("--noise", type=str, default="noise", help="have or have not sample noise (noise, no_noise, depolar_noise)")
    parser.add_argument("--n_samples", type=int, default=100000, help="number of samples")
    parser.add_argument("--P_povm", type=float, default=1.0, help="possbility of sampling POVM operators")
    parser.add_argument("--seed_povm", type=float, default=1.0, help="seed of sampling POVM operators")
    parser.add_argument("--read_data", type=bool, default=False, help="read data from text in computer")

    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--lr", type=float, default=0.001, help="optim: learning rate")

    parser.add_argument("--map_method", type=str, default="chol_h", help="map method for output vector to density matrix (chol, chol_h, proj_F, proj_S, proj_A)")
    parser.add_argument("--P_proj", type=float, default="1", help="coefficient for proj method")

    opt = parser.parse_args()

    results = Net_train(opt, device)


    #-----ex: 6 (Convergence Experiment of Pure States with Depolarizing Noise)-----
    '''
    opt.noise = 'depolar_noise'
    for na_s in ["W_P", 'Product_P', 'GHZi_P']:
        opt.na_state = na_s
        r_path = '../../results/FNN/result/' + opt.na_state + '/'
        if os.path.isdir(r_path):
            print('result dir exists, is: ' + r_path)
        else:
            os.makedirs(r_path)
            print('result dir not exists, has been created, is: ' + r_path)

        opt.n_samples = 10000
        save_data = {}
        for P_s in np.random.uniform(0, 1, 50):
            opt.P_state = P_s

            results = Net_train(opt, device)
            save_data[str(P_s)] = results
        np.save(r_path+'NN_S_N'+str(int(opt.n_samples/100))+'.npy', save_data)'''


    #-----ex: 5 (Convergence Experiment of Random Mixed States for CNN on different qubits)-----
    '''
    for na_s in ["random_P", "GHZi_P", "W_P", "Product_P"]:
        opt.na_state = na_s
        r_path = '../../results/FNN/result/' + opt.na_state + '/'
        if os.path.isdir(r_path):
            print('result dir exists, is: ' + r_path)
        else:
            os.makedirs(r_path)
            print('result dir not exists, has been created, is: ' + r_path)

        for n_qubit in np.arange(4, 11):
            opt.n_qubits = n_qubit
            save_data = {}
            for P_s in np.random.uniform(0, 1, 50):
                opt.P_state = P_s

                results = Net_train(opt, device)
                save_data[str(P_s)] = results
            np.save(r_path+'CNN_'+str(n_qubit)+'.npy', save_data)'''


    #-----ex: 4 (Convergence Experiment of Random Mixed States for Different Samples)-----
    '''
    for na_s in ["random_P"]:
        opt.na_state = na_s
        r_path = '../../results/FNN/result/' + opt.na_state + '/'
        if os.path.isdir(r_path):
            print('result dir exists, is: ' + r_path)
        else:
            os.makedirs(r_path)
            print('result dir not exists, has been created, is: ' + r_path)

        for sample in [100, 500, 1000, 5000, 10000, 50000, 100000, 1000000]:
            opt.n_samples = sample
            save_data = {}
            for P_s in np.random.uniform(0, 1, 50):
                opt.P_state = P_s

                results = Net_train(opt, device)
                save_data[str(P_s)] = results
            np.save(r_path+'NN_S'+str(int(sample/100))+'.npy', save_data)'''


    '''
    #-----ex: 3 (Convergence Experiment of Random Mixed States for Different Qubits, no noise)-----
    for na_s in ["random_P", "GHZi_P", "W_P", "Product_P"]:
        opt.na_state = na_s
        r_path = '../../results/FNN/result/' + opt.na_state + '/'
        if os.path.isdir(r_path):
            print('result dir exists, is: ' + r_path)
        else:
            os.makedirs(r_path)
            print('result dir not exists, has been created, is: ' + r_path)

        for n_qubit in np.arange(2, 12):
            opt.n_qubits = n_qubit
            save_data = {}
            for P_s in np.random.uniform(0, 1, 50):
                opt.P_state = P_s

                results = Net_train(opt, device)
                save_data[str(P_s)] = results
            np.save(r_path+str(n_qubit)+'.npy', save_data)'''


    '''
    #-----ex: 2 (Convergence Experiments of Random Mixed State for Different Mapping Methods)-----
    for na_s in ["random_P"]:
        opt.na_state = na_s
        r_path = '../../results/FNN/result/' + opt.na_state + '/'
        if os.path.isdir(r_path):
            print('result dir exists, is: ' + r_path)
        else:
            os.makedirs(r_path)
            print('result dir not exists, has been created, is: ' + r_path)

        for m_method in ['chol', 'chol_h', 'proj_F', 'proj_S', 'proj_A']:
            opt.map_method = m_method
            if m_method == 'proj_A':
                for p_proj in [0.5, 1, 1.5, 2, 3, 4]:
                    opt.P_proj = p_proj
                    save_data = {}
                    for P_s in np.random.uniform(0, 1, 50):
                        opt.P_state = P_s

                        results = Net_train(opt, device)
                        save_data[str(P_s)] = results
                    np.save(r_path + m_method + '_' + str(p_proj) + '.npy', save_data)
            else:
                save_data = {}
                for P_s in np.random.uniform(0, 1, 50):
                    opt.P_state = P_s

                    results = Net_train(opt, device)
                    save_data[str(P_s)] = results
                np.save(r_path + m_method + '.npy', save_data)'''


    '''
    #-----ex: 1 (Special State Convergence Experiments for Different Mapping Methods)-----
    for na_s in ["GHZi_P", "W_P", "Product_P"]:
        opt.na_state = na_s
        r_path = '../../results/FNN/result/' + opt.na_state + '/'
        if os.path.isdir(r_path):
            print('result dir exists, is: ' + r_path)
        else:
            os.makedirs(r_path)
            print('result dir not exists, has been created, is: ' + r_path)

        for m_method in ['chol', 'chol_h', 'proj_F', 'proj_S', 'proj_A']:
            opt.map_method = m_method
            if m_method == 'proj_A':
                for p_proj in [0.5, 1, 1.5, 2, 3, 4]:
                    opt.P_proj = p_proj
                    save_data = {}
                    for P_s in np.linspace(0, 1, 10):
                        opt.P_state = P_s

                        results = Net_train(opt, device)
                        save_data[str(P_s)] = results
                    np.save(r_path + m_method + '_' + str(p_proj) + '.npy', save_data)
            else:
                save_data = {}
                for P_s in np.linspace(0, 1, 10):
                    opt.P_state = P_s

                    results = Net_train(opt, device)
                    save_data[str(P_s)] = results
                np.save(r_path + m_method + '.npy', save_data)
    '''
