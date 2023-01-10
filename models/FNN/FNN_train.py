# -*- coding: utf-8 -*-
# @Author: foxwy
# @Date:   2021-05-20 18:58:08
# @Last Modified by:   yong
# @Last Modified time: 2022-12-23 22:31:48

"""
-----------------------------------------------------------------------------------------
    Used for pre-training of feedforward network.
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

import warnings
warnings.filterwarnings("ignore")

filepath = os.path.abspath(os.path.join(os.getcwd(), '../..'))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append('../..')
from models.FNN.Net_Product import generator, Net_MLP_train
from datasets.dataset import GetLoader
from Basis.Basic_Function import get_default_device


def Net_train(opt, device):
    """
    *******Main Execution Function*******
    """
    torch.cuda.empty_cache()
    print('\nparameter:', opt)

    # ----------data----------
    print('\n'+'-'*20+'data'+'-'*20)
    b_size = 64
    n_workers = 8
    trainset = GetLoader(name=filepath + '/datasets/datasets/data_', N=80)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=b_size, shuffle=True, num_workers=n_workers, pin_memory=True)

    testset = GetLoader(name=filepath + '/datasets/datasets/data_test_', N=10)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=b_size, shuffle=True, num_workers=n_workers, pin_memory=True)

    # ----------net----------
    in_size = opt.K**opt.n_qubits
    gen_net = generator(in_size, opt.n_qubits, type_state=opt.ty_state, map_method=opt.map_method, P_proj=opt.P_proj, net_type=opt.net_type, device=device).to(device)
    #gen_net.load_state_dict(torch.load('model_6q.pt'))

    # ----------train----------
    net = Net_MLP_train(gen_net, opt.lr)
    net.train(trainloader, testloader, opt.n_epochs, device)


# --------------------main--------------------
if __name__ == '__main__':
    """
    *******Main Function*******
    """
    # ----------device----------
    print('-'*20+'init'+'-'*20)
    device = get_default_device()
    print('device:', device)

    # ----------parameters----------
    print('-'*20+'set parser'+'-'*20)
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=4, help='number of operators in single-qubit POVM')

    parser.add_argument("--ty_state", type=str, default="mixed", help="type of state (pure, mixed)")
    parser.add_argument("--n_qubits", type=int, default=6, help="number of qubits")

    parser.add_argument("--n_epochs", type=int, default=1300, help="number of epochs of training")
    parser.add_argument("--lr", type=float, default=0.001, help="optim: learning rate")

    parser.add_argument("--map_method", type=str, default="chol_h", help="map method for output vector to density matrix (chol, chol_h, proj_F, proj_S, proj_A)")
    parser.add_argument("--P_proj", type=float, default="1", help="coefficient for proj method")
    parser.add_argument("--net_type", type=str, default="train", help="type of neural network (train, relearn)")

    opt = parser.parse_args()

    results = Net_train(opt, device)
