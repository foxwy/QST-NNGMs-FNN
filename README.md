# **Ultrafast quantum state tomography with feed-forward neural networks**

The official Pytorch implementation of the paper named [`Ultrafast quantum state tomography with feed-forward neural networks`](https://arxiv.org/abs/2207.05341), under review on Quantum journal.

[![arXiv](https://img.shields.io/badge/arXiv-<2207.05341>-<COLOR>.svg)](https://arxiv.org/abs/2207.05341)

### **Abstract**

Reconstructing the state of many-body quantum systems is of fundamental importance in quantum information tasks, but extremely challenging due to the curse of dimensionality. In this work, we present a quantum tomography approach based on neural networks to achieve the ultrafast reconstruction of multi-qubit states. Particularly, we propose a simple 3-layer feed-forward network to process the experimental data generated from measuring each qubit with a positive operator-valued measure, which is able to reduce the storage cost and computational complexity. Then, the techniques of state decomposition and P-order absolute projection are jointly introduced to ensure the positivity of state matrices learned in the combined loss function and to improve the tomography fidelity and purity robustness of the above network. The proposed state-mapping method also substantially improves the tomography accuracy to other QST algorithms. Finally, it is tested on a large number of states with a wide range of purity to show that the proposed algorithm achieves more accurate tomography with low time and iterations than previous algorithms, and achieves faithfully tomography 11-qubit states within 2 minutes. Our numerical results also demonstrate that the increased depolarizing noise induces a linear decrease in the tomography fidelity and the ability to achieve iteration-free tomography with the help of pre-training.

### **Citation**

If you find our work useful in your research, please cite:

```
@ARTICLE{2022arXiv220705341W,
    author = {{Wang}, Yong and {Cheng}, Shuming and {Li}, Li and {Chen}, Jie},
    title = {Ultrafast quantum state tomography with feed-forward neural networks},
    journal = {arXiv preprint arXiv:2207.05341}
    year = 2022,
    month = jul
}
```

## Getting started

This code was tested on the computer with a single Intel(R) Core(TM) i7-12700KF CPU @ 3.60GHz with 64GB RAM and a single NVIDIA GeForce RTX 3090 Ti GPU with 24.0GB RAM, and requires:

- Python 3.9
- conda3
- torch==1.14.0.dev20221203+cu117
- h5py==3.7.0
- numpy==1.24.0rc1
- openpyxl==3.0.10
- scipy==1.7.3
- tqdm==4.64.1

## Runs direct learning QST algorithms ([`models/FNN`](models/FNN/FNN_learn.py))

```bash
cd "models/FNN"
python FNN_learn.py
```

### 1. Initial Parameters (`main`)

```python
parser = argparse.ArgumentParser()
parser.add_argument("--POVM", type=str, default="Tetra4", help="type of POVM")
parser.add_argument("--K", type=int, default=4, help='number of operators in single-qubit POVM')

parser.add_argument("--na_state", type=str, default="real_random", help="name of state in library")
parser.add_argument("--P_state", type=float, default=0.6, help="P of mixed state")
parser.add_argument("--ty_state", type=str, default="mixed", help="type of state (pure, mixed)")
parser.add_argument("--n_qubits", type=int, default=8, help="number of qubits")

parser.add_argument("--noise", type=str, default="no_noise", help="have or have not sample noise (noise, no_noise, depolar_noise)")
parser.add_argument("--n_samples", type=int, default=1000000, help="number of samples")
parser.add_argument("--P_povm", type=float, default=1, help="possbility of sampling POVM operators")
parser.add_argument("--seed_povm", type=float, default=1.0, help="seed of sampling POVM operators")
parser.add_argument("--read_data", type=bool, default=False, help="read data from text in computer")

parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--lr", type=float, default=0.001, help="optim: learning rate")

parser.add_argument("--map_method", type=str, default="chol_h", help="map method for output vector to density matrix (chol, chol_h, proj_F, proj_S, proj_A)")
parser.add_argument("--P_proj", type=float, default="2", help="coefficient for proj method")
parser.add_argument("--net_type", type=str, default="learn", help="type of neural network (train, relearn)")
```

### 2. Run FNN algorithm (`Net_train`)

```python
print('\\n'+'-'*20+'FNN'+'-'*20)
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
```

### 3. Run CNN algorithm (`Net_train`)

```python
print('\\n'+'-'*20+'CNN'+'-'*20)
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
```

### 4. Run SNN algorithm (`Net_train`)

```python
print('\\n'+'-'*20+'SNN'+'-'*20)
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
```

### 5. Run iMLE algorithm (`Net_train`)

```python
print('\\n'+'-'*20+'iMLE'+'-'*20)
result_save = {'parser': opt,
               'time': [], 
               'epoch': [],
               'Fc': [],
               'Fq': []}
iMLE(M, opt.n_qubits, data_all, opt.n_epochs, fid, result_save, device)
result_saves['iMLE'] = result_save
```

### 6. Run CG-APG algorithm (`Net_train`)

```python
print('\\n'+'-'*20+'QSE APG'+'-'*20)
result_save = {'parser': opt,
               'time': [], 
               'epoch': [],
               'Fc': [],
               'Fq': []}
qse_apg(M, opt.n_qubits, data_all, opt.n_epochs, fid, 'chol_h', 3, result_save, device)
result_saves['APG'] = result_save
```

### 7. Run CG-APG algorithm with ProjA_3 (`Net_train`)

```python
print('\\n'+'-'*20+'QSE APG Proj'+'-'*20)
result_save = {'parser': opt,
               'time': [], 
               'epoch': [],
               'Fc': [],
               'Fq': []}
qse_apg(M, opt.n_qubits, data_all, opt.n_epochs, fid, 'proj_A', 3, result_save, device)
result_saves['APG_projA'] = result_save
```

### 8. Run LRE algorithm (`Net_train`)

```python
print('\\n'+'-'*20+'LRE'+'-'*20)
result_save = {'parser': opt,
               'time': [],
               'Fc': [],
               'Fq': []}
LRE(M, opt.n_qubits, data_all, fid, 'proj_S', 1, result_save, device)
result_saves['LRE'] = result_save
```

### 9. Run LRE algorithm with ProjA_1 (`Net_train`)

```python
print('\\n'+'-'*20+'LRE proj'+'-'*20)
result_save = {'parser': opt,
               'time': [],
               'Fc': [],
               'Fq': []}
LRE(M, opt.n_qubits, data_all, fid, 'proj_A', 1, result_save, device)
result_saves['LRE_projA'] = result_save
```

## Run post-pre-training learning QST algorithm

### 1. Dataset Preparation ([`datasets/dataset.py`](datasets/dataset.py))

```python
n_qubits = 6
POVM = 'Tetra4'
ty_state = 'mixed'
device = get_default_device()

M = Mea_basis(POVM).M
M = torch.from_numpy(M).to(device)
Dataset_train(100000, 'data_', M, n_qubits, 4, ty_state, device=device)  # trainset
Dataset_train(20000, 'data_test_', M, n_qubits, 4, ty_state, device=device)  # testset
```

trainset and testset are saved in [`datasets/datasets`](datasets/datasets).

### 2. Model Training ([`models/FNN/FNN_train.py`](models/FNN/FNN_train.py))

```python
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
```

model is saved in [`models/FNN`](models/FNN)  named `model.pt`.

### 3. Perform NN-QST algorithm ([`models/FNN/FNN_learn.py`](models/FNN/FNN_learn.py))

```python
print('\\n'+'-'*20+'NN SNN'+'-'*20)
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
```

## **Acknowledgments**

This code is standing on the shoulders of giants. We want to thank the following contributors that our code is based on: [POVM_GENMODEL](https://github.com/carrasqu/POVM_GENMODEL), [qMLE](https://github.com/qMLE/qMLE).

## **License**

This code is distributed under an [Mozilla Public License Version 2.0](LICENSE).

Note that our code depends on other libraries, including POVM_GENMODEL, qMLE, and uses algorithms that each have their own respective licenses that must also be followed.