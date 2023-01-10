# -*- coding: utf-8 -*-
# @Author: foxwy
# @Date:   2021-05-20 18:58:08
# @Last Modified by:   yong
# @Last Modified time: 2022-12-24 10:23:26

"""
-----------------------------------------------------------------------------------------
    The main function of quantum state tomography, used in the experimental 
    part of the paper view, calls other implementations of the QST algorithm,
    paper: ``Ultrafast quantum state tomography with feed-forward neural networks``.
    
    @ARTICLE{2022arXiv220705341W,
       author = {{Wang}, Yong and {Cheng}, Shuming and {Li}, Li and {Chen}, Jie},
        title = "{Ultrafast quantum state tomography with feed-forward neural networks}",
      journal = {arXiv e-prints},
     keywords = {Quantum Physics, Physics - Data Analysis, Statistics and Probability},
         year = 2022,
        month = jul,
        pages = {arXiv:2207.05341},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv220705341W}
    }
-----------------------------------------------------------------------------------------
"""

import os
import sys
import argparse
import torch
import numpy as np

filepath = os.path.abspath(os.path.join(os.getcwd(), '../..'))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append('../..')
from models.FNN.Net_Product import generator, Net_MLP, Net_Conv
from models.FNN.SNN import SNN_nn, SNN
from models.MLE_Pytorch.iMLE import iMLE
from models.MLE_Pytorch.qse_apg import qse_apg
from models.MLE_Pytorch.LRE import LRE
from Basis.Basis_State import Mea_basis, State
from datasets.dataset import Dataset_P, Dataset_sample, Dataset_sample_P
from evaluation.Fidelity import Fid
from Basis.Basic_Function import get_default_device


def Net_train(opt, device):
    """
    *******Main Execution Function*******
    """
    torch.cuda.empty_cache()
    print('\nparameter:', opt)

    # ----------file----------
    r_path = '../../results/FNN/result/' + opt.na_state + '/'
    if os.path.isdir(r_path):
        print('result dir exists, is: ' + r_path)
    else:
        os.makedirs(r_path)
        print('result dir not exists, has been created, is: ' + r_path)

    # ----------rho_star and M----------
    print('\n'+'-'*20+'rho'+'-'*20)
    state_star, rho_star = State().Get_state_rho(opt.na_state, opt.n_qubits, opt.P_state)
    if opt.ty_state == 'pure':  # pure state
        rho_star = state_star

    if opt.noise == 'depolar_noise':
        rho_t, rho_star = State().Get_state_rho(opt.na_state, opt.n_qubits, 1 - opt.P_state)
        rho_o = rho_t.dot(rho_t.T.conjugate())
        rho_t = torch.from_numpy(rho_t).to(torch.complex64).to(device)
        rho_o = torch.from_numpy(rho_o).to(torch.complex64).to(device)
    
    rho_star = torch.from_numpy(rho_star).to(torch.complex64).to(device)
    M = Mea_basis(opt.POVM).M
    M = torch.from_numpy(M).to(device)

    # ----------data----------
    print('\n'+'-'*20+'data'+'-'*20)
    print('read original data')
    if opt.noise == "no_noise":  # perfect measurment
        print('----read ideal data')
        P_idxs, data, data_all = Dataset_P(rho_star, M, opt.n_qubits, opt.K, opt.ty_state, opt.P_povm, opt.seed_povm)
    else:
        print('----read sample data')
        if opt.P_povm == 1:  # measure all POVM
            P_idxs, data, data_all = Dataset_sample(opt.POVM, opt.na_state, opt.n_qubits, 
                                                    opt.n_samples, opt.P_state, opt.ty_state, 
                                                    rho_star, M, opt.read_data)
        else:  # measure partial POVM
            P_idxs, data, data_all = Dataset_sample_P(opt.POVM, opt.na_state, opt.n_qubits, 
                                                      opt.K, opt.n_samples, opt.P_state, 
                                                      opt.ty_state, rho_star, opt.read_data,
                                                      opt.P_povm, opt.seed_povm)
    
    in_size = len(data)
    print('data shape:', in_size)

    # fidelity
    if opt.noise == 'depolar_noise':
        fid = Fid(basis=opt.POVM, n_qubits=opt.n_qubits, ty_state=opt.ty_state, 
                  rho_star=[rho_t, rho_o], M=M, device=device)
    else:
        fid = Fid(basis=opt.POVM, n_qubits=opt.n_qubits, ty_state=opt.ty_state, 
                  rho_star=rho_star, M=M, device=device)
    CF = fid.cFidelity_S_product(P_idxs, data)
    print('classical fidelity:', CF)


    #----------------------------------------------QST algorithms----------------------------------------------------
    #---FNN---
    print('\n'+'-'*20+'FNN'+'-'*20)
    gen_net = generator(in_size, opt.n_qubits, P_idxs, 
                        M, type_state=opt.ty_state, map_method=opt.map_method, 
                        P_proj=opt.P_proj, net_type=opt.net_type, device=device).to(torch.float32).to(device)

    net = Net_MLP(gen_net, data, opt.lr)
    result_save = {'parser': opt,
                   'time': [], 
                   'epoch': [],
                   'Fc': [],
                   'Fq': []}
    net.train(opt.n_epochs, fid, result_save)
    result_saves['NN'] = result_save


    #---CNN---
    print('\n'+'-'*20+'CNN'+'-'*20)
    gen_net = Net_Conv(in_size, opt.n_qubits, P_idxs, 
                       M, type_state=opt.ty_state, map_method=opt.map_method, 
                       P_proj=opt.P_proj, device=device).to(torch.float32).to(device)
    print(gen_net)

    net = Net_MLP(gen_net, data, opt.lr)
    result_save = {'parser': opt,
                   'time': [], 
                   'epoch': [],
                   'Fc': [],
                   'Fq': []}
    net.train(opt.n_epochs, fid, result_save)
    result_saves['CNN'] = result_save


    #---pre-train FNN+SNN---
    print('\n'+'-'*20+'NN SNN'+'-'*20)
    gen_net = generator(in_size, opt.n_qubits, P_idxs, 
                        M, type_state=opt.ty_state, map_method=opt.map_method, 
                        P_proj=opt.P_proj, net_type=opt.net_type, device=device).to(torch.float32).to(device)

    gen_net.load_state_dict(torch.load('model.pt'))
    gen_net(data)
    rho_init = gen_net.rho.detach()
    gen_net = SNN_nn(opt.n_qubits, P_idxs, M, rho_init, 
                     map_method=opt.map_method, P_proj=opt.P_proj).to(torch.float32).to(device)

    net = SNN(gen_net, data, opt.lr)
    result_save = {'parser': opt,
                   'time': [], 
                   'epoch': [],
                   'Fc': [],
                   'Fq': []}
    net.train(opt.n_epochs, fid, result_save)
    result_saves['NN_SNN'] = result_save


    #---SNN---
    print('\n'+'-'*20+'SNN'+'-'*20)
    rho_init = None
    gen_net = SNN_nn(opt.n_qubits, P_idxs, M, 
                     rho_init, map_method=opt.map_method, P_proj=opt.P_proj).to(torch.float32).to(device)

    net = SNN(gen_net, data, opt.lr)
    result_save = {'parser': opt,
                   'time': [], 
                   'epoch': [],
                   'Fc': [],
                   'Fq': []}
    net.train(opt.n_epochs, fid, result_save)
    result_saves['SNN'] = result_save


    #---iMLE---
    print('\n'+'-'*20+'iMLE'+'-'*20)
    result_save = {'parser': opt,
                   'time': [], 
                   'epoch': [],
                   'Fc': [],
                   'Fq': []}
    iMLE(M, opt.n_qubits, data_all, opt.n_epochs, fid, result_save, device)
    result_saves['iMLE'] = result_save


    #---qse_apg---
    print('\n'+'-'*20+'QSE APG'+'-'*20)
    result_save = {'parser': opt,
                   'time': [], 
                   'epoch': [],
                   'Fc': [],
                   'Fq': []}
    qse_apg(M, opt.n_qubits, data_all, opt.n_epochs, fid, 'chol_h', 3, result_save, device)
    result_saves['APG'] = result_save


    print('\n'+'-'*20+'QSE APG Proj'+'-'*20)
    result_save = {'parser': opt,
                   'time': [], 
                   'epoch': [],
                   'Fc': [],
                   'Fq': []}
    qse_apg(M, opt.n_qubits, data_all, opt.n_epochs, fid, 'proj_A', 3, result_save, device)
    result_saves['APG_projA'] = result_save


    #---LRE---
    print('\n'+'-'*20+'LRE'+'-'*20)
    result_save = {'parser': opt,
                   'time': [],
                   'Fc': [],
                   'Fq': []}
    LRE(M, opt.n_qubits, data_all, fid, 'proj_S', 1, result_save, device)
    result_saves['LRE'] = result_save


    print('\n'+'-'*20+'LRE proj'+'-'*20)
    result_save = {'parser': opt,
                   'time': [],
                   'Fc': [],
                   'Fq': []}
    LRE(M, opt.n_qubits, data_all, fid, 'proj_A', 1, result_save, device)
    result_saves['LRE_projA'] = result_save

    return result_saves


if __name__ == '__main__':
    """
    *******Main Function*******
    Given QST perform parameters.
    """
    # ----------device----------
    print('-'*20+'init'+'-'*20)
    device = get_default_device()
    print('device:', device)

    # ----------parameters----------
    print('-'*20+'set parser'+'-'*20)
    parser = argparse.ArgumentParser()
    parser.add_argument("--POVM", type=str, default="Tetra4", help="type of POVM")
    parser.add_argument("--K", type=int, default=4, help='number of operators in single-qubit POVM')

    parser.add_argument("--na_state", type=str, default="GHZ_P", help="name of state in library")
    parser.add_argument("--P_state", type=float, default=1, help="P of mixed state")
    parser.add_argument("--ty_state", type=str, default="mixed", help="type of state (pure, mixed)")
    parser.add_argument("--n_qubits", type=int, default=11, help="number of qubits")

    parser.add_argument("--noise", type=str, default="no_noise", help="have or have not sample noise (noise, no_noise, depolar_noise)")
    parser.add_argument("--n_samples", type=int, default=1000000000, help="number of samples")
    parser.add_argument("--P_povm", type=float, default=1, help="possbility of sampling POVM operators")
    parser.add_argument("--seed_povm", type=float, default=1.0, help="seed of sampling POVM operators")
    parser.add_argument("--read_data", type=bool, default=False, help="read data from text in computer")

    parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
    parser.add_argument("--lr", type=float, default=0.1, help="optim: learning rate")

    parser.add_argument("--map_method", type=str, default="chol_h", help="map method for output vector to density matrix (chol, chol_h, proj_F, proj_S, proj_A)")
    parser.add_argument("--P_proj", type=float, default="3", help="coefficient for proj method")
    parser.add_argument("--net_type", type=str, default="learn", help="type of neural network (train, relearn)")

    opt = parser.parse_args()

    results = Net_train(opt, device)

    #-----ex: 6 (Convergence Experiment of Pure States with Depolarizing Noise)-----
    '''
    opt.n_qubits = 10
    opt.na_state = 'real_random'
    opt.map_method = 'chol_h'
    opt.noise = 'depolar_noise'
    for sample in [10**9]:
        opt.n_samples = sample

        save_data = {}
        for idx, P_s in enumerate(np.linspace(0, 1, 20)):
            print('-'*40, idx, P_s)
            opt.P_state = P_s

            results = Net_train(opt, device)
            save_data[str(P_s)] = results

        np.save(r_path + str(int(np.log10(sample))) + '_depolar.npy', save_data)'''


    #-----ex: 5 (Convergence Experiment of Random Mixed States for Different Qubits, no noise)-----
    '''
    opt.na_state = 'real_random'
    opt.noise = 'no_noise'
    opt.map_method = 'chol_h'

    for n_qubit in [12]:
        opt.n_qubits = n_qubit
        save_data = {}
        for idx, P_s in enumerate(np.random.uniform(0, 1, 20)):
            print('-'*40, idx, P_s)
            opt.P_state = P_s

            results = Net_train(opt, device)
            save_data[str(P_s)] = results

        np.save(r_path + str(n_qubit) + 'q_11NN' + '.npy', save_data)'''


    #-----ex: 4 (Random State Convergence Experiments of pretrained NN and other QST algorithms for Different samples)-----
    '''
    opt.n_qubits = 6
    opt.na_state = 'real_random'
    opt.map_method = 'chol_h'
    for sample in [None]:
        if sample is not None:
            opt.n_samples = sample
            opt.noise = 'noise'
        else:
            opt.noise = 'no_noise'
        
        save_data = {}
        for idx, P_s in enumerate(np.linspace(0.0001, 0.9999, 20)):
            print('-'*40, idx, P_s)
            opt.P_state = P_s

            results = Net_train(opt, device)
            save_data[str(P_s)] = results

        if sample is not None:
            np.save(r_path + str(int(np.log10(sample))) + '_pretrain_6.npy', save_data)
        else:
            np.save(r_path + '11' + '_pretrain_6.npy', save_data)'''


    #-----ex: 3 (Random State Convergence Experiments of QST algorithms for Different samples)-----
    '''
    opt.n_qubits = 10
    opt.na_state = 'real_random'
    opt.map_method = 'chol_h'
    for sample in [10**5, 10**6, 10**7, 10**8, 10**9, None]:
        if sample is not None:
            opt.n_samples = sample
            opt.noise = 'noise'
        else:
            opt.noise = 'no_noise'

        save_data = {}
        for idx, P_s in enumerate(np.random.uniform(0, 1, 5)):
            print('-'*40, idx, P_s)
            opt.P_state = P_s

            results = Net_train(opt, device)
            save_data[str(P_s)] = results

        if sample is not None:
            np.save(r_path + str(int(np.log10(sample))) + '_NN_SNN.npy', save_data)
        else:
            np.save(r_path + '11' + '_NN_SNN.npy', save_data)'''


    #-----ex: 2 (Random State Convergence Experiments of NN-QST for Different Mapping Methods)-----
    '''
    opt.n_qubits = 10
    opt.na_state = 'real_random'
    for sample in [10**5, 10**6, 10**7, 10**8, 10**9, None]:
        if sample is not None:
            opt.n_samples = sample
            opt.noise = 'noise'
        else:
            opt.noise = 'no_noise'
        
        for m_method in ['proj_F']:
            opt.map_method = m_method
            if m_method == 'proj_A':
                for p_proj in [4]:
                    opt.P_proj = p_proj
                    save_data = {}
                    for idx, P_s in enumerate(np.random.uniform(0, 1, 20)):
                        print('-'*40, idx, P_s)
                        opt.P_state = P_s

                        results = Net_train(opt, device)
                        save_data[str(P_s)] = results

                    if sample is not None:
                        np.save(r_path + 'NN_' + m_method + '_'+ str(p_proj) + '_' + str(int(np.log10(sample))) + '.npy', save_data)
                    else:
                        np.save(r_path + 'NN_' + m_method + '_'+ str(p_proj) + '_11' + '.npy', save_data)

            else:
                save_data = {}
                for idx, P_s in enumerate(np.random.uniform(0, 1, 20)):
                    print('-'*40, idx, P_s)
                    opt.P_state = P_s

                    try:
                        results = Net_train(opt, device)
                    except Exception:
                        P_s_1 = np.random.uniform(0, 1, 1)[0]
                        opt.P_state = P_s
                        results = Net_train(opt, device)
                    
                    save_data[str(P_s)] = results

                if sample is not None:
                    np.save(r_path + 'NN_' + m_method + '_' + str(int(np.log10(sample))) + '.npy', save_data)
                else:
                    np.save(r_path + 'NN_' + m_method + '_11' + '.npy', save_data)'''


    #-----ex: 1 (Random State Convergence Experiments of NN-QST for Different loss functions)-----
    '''
    opt.n_qubits = 10
    opt.na_state = 'real_random'
    for sample in [10**5]:
        if sample is not None:
            opt.n_samples = sample
            opt.noise = 'noise'
        else:
            opt.noise = 'no_noise'

        for m_method in ['chol_h']:
            opt.map_method = m_method
            if m_method == 'proj_A':
                for p_proj in [4]:
                    opt.P_proj = p_proj
                    save_data = {}
                    for P_s in np.random.uniform(0, 1, 20):
                        opt.P_state = P_s

                        results = Net_train(opt, device)
                        save_data[str(P_s)] = results

                    if sample is not None:
                        np.save(r_path + 'NN_' + m_method + '_'+ str(p_proj) + '_' + str(int(np.log10(sample))) + '.npy', save_data)
                    else:
                        np.save(r_path + 'NN_' + m_method + '_'+ str(p_proj) + '_inf' + '.npy', save_data)

            else:
                save_data = {}
                for P_s in np.random.uniform(0, 1, 20):
                    opt.P_state = P_s

                    results = Net_train(opt, device)
                    save_data[str(P_s)] = results

                if sample is not None:
                    np.save(r_path + 'NN_' + m_method + '_' + str(int(np.log10(sample))) +'_NLL_H'+ '.npy', save_data)
                else:
                    np.save(r_path + 'NN_' + m_method + '_11_NLL_H' + '.npy', save_data)'''
