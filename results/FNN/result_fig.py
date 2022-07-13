# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import scipy.io as scio
import openpyxl

import sys
sys.path.append(".")

plt.style.use(['science', 'no-latex'])
plt.rcParams["font.family"] = 'Arial'

font_size = 34  # 34, 28, 38
font = {'size': font_size, 'weight': 'normal'}


def Plt_set(ax, xlabel, ylabel, savepath, log_flag=0, loc=4, ncol=1, f_size=23):
    ax.tick_params(labelsize=font_size)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)
    ax.spines['top'].set_linewidth(5)
    ax.tick_params(width=5)
    font2 = {'size': f_size, 'weight': 'normal'}
    ax.legend(prop=font2, loc=loc, frameon=True, ncol=ncol)
    ax.set_xlabel(xlabel, font)  # fontweight='bold'
    ax.set_ylabel(ylabel, font)

    if log_flag == 1:
        ax.set_xscale('log')
    if log_flag == 2:
        ax.set_yscale('log')
    if log_flag == 3:
        ax.set_xscale('log')
        ax.set_yscale('log')

    if 'Time' in ylabel:
        ax.set_yticks([0.01, 1, 10, 100])
    '''
    if 'Time' in xlabel:
        ax.set_xticks([0.1, 1, 10, 100])'''

    if 'samples' in xlabel:
        ax.set_xticks([100, 1000, 10000, 100000, 1000000])
    
    if 'samples' in ylabel:
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        tx = ax.yaxis.get_offset_text()
        tx.set_fontsize(30)

    plt.savefig(savepath + '.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)


def Get_r(V):
    V_avg = np.mean(V, 0)
    V_std = np.std(V, 0)
    r1 = list(map(lambda x: x[0] - x[1], zip(V_avg, V_std)))
    r2 = list(map(lambda x: x[0] + x[1], zip(V_avg, V_std)))

    return r1, r2


def Pure_ex_fig(colors):
    wb = openpyxl.load_workbook('result/Pure_state_sample_qubit_099.xlsx')
    ws = wb['Sheet1']

    N_q = []
    GHZi = []
    GHZi_RNN = []
    Product = []
    Product_RNN = []
    for N, S1, S1_r, S2, S2_r in zip(ws['A'][2:9], ws['B'][2:9], ws['B'][10:17], ws['B'][18:25], ws['B'][26:33]):
        N_q.append(N.value)
        GHZi.append(S1.value)
        GHZi_RNN.append(S1_r.value)
        Product.append(S2.value)
        Product_RNN.append(S2_r.value)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.plot(N_q, GHZi, label='$\\rm GHZi$', linewidth=5, color=colors[9])
    ax.plot(N_q, GHZi_RNN, label='$\\rm GHZi+RNN$', linewidth=5, color=colors[9], marker='*', markersize=15)
    ax.plot(N_q, Product, label='$\\rm Product$', linewidth=5, color=colors[0])
    ax.plot(N_q, Product_RNN, label='$\\rm Product+RNN$', linewidth=5, color=colors[0], marker='*', markersize=15)

    ax.set_yticks([0, 5000, 10000, 15000, 20000, 25000, 30000])
    ax.set_xticks([2, 3, 4, 5, 6, 7, 8])
    Plt_set(ax, 'Number of qubits', 'Number of samples', 'fig/pure_state_sample_qubit', loc=2)
    #plt.show()


#-----ex: 1 (Convergence Experiments of Mixed State for Different Mapping Methods)-----
def Map_ex_fig(na_state, state, r_path, m_path, colors):
    m_methods = ['chol', 'chol_h', 'proj_F', 'proj_S', 'proj_A_0.5', 'proj_A_1', 'proj_A_1.5', 'proj_A_2', 'proj_A_3', 'proj_A_4']
    #------分图:不同映射的收敛性------
    '''
    for i in range(len(m_methods)):
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        print(m_methods[i])
        savePath = r_path + m_methods[i]

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        P_all = []
        for p in results:
            P_all.append(float(p))
        P_all.sort()
        for j, p in enumerate(P_all):
            p = str(p)
            if na_state == 'random_P':
                ax.plot(results[p]['time'], np.minimum(results[p]['Fq'], 1), label=p[:4], linewidth=5)
            else:
                ax.plot(results[p]['time'], np.minimum(results[p]['Fq'], 1), label=p[:4], linewidth=5, color=colors[j])

        ax.plot([0, max(results[p]['time'])], [1, 1], 'k--', linewidth=2.5)
        if na_state == 'random_P':
            Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/map/'+na_state+'_time_'+m_methods[i], 1, ncol=5, f_size=10)
        else:
            Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/map/'+na_state+'_time_'+m_methods[i], 1, ncol=2)

        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
        for j, p in enumerate(P_all):
            p = str(p)
            if na_state == 'random_P':
                ax2.plot(results[p]['epoch'], np.minimum(results[p]['Fq'], 1), label=p[:4], linewidth=5)
            else:
                ax2.plot(results[p]['epoch'], np.minimum(results[p]['Fq'], 1), label=p[:4], linewidth=5, color=colors[j])

        ax2.plot([0, max(results[p]['epoch'])], [1, 1], 'k--', linewidth=5)
        if na_state == 'random_P':
            Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/map/'+na_state+'_epoch_'+m_methods[i], 1, ncol=5, f_size=10)
        else:
            Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/map/'+na_state+'_epoch_'+m_methods[i], 1, ncol=2)'''

    #plt.show()
    

    #------总图:不同映射的收敛性------
    #-----time, Fq-----
    '''
    map_location=torch.device('cpu')
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    labels = ['$\\rm Chol_\Delta$', '$\\rm Chol_H$', '$\\rm \mathcal{F}[\cdot]$', '$\\rm \mathcal{S}[\cdot]$', \
              '$\\rm \mathcal{A}[\cdot]_{0.5}$', '$\\rm \mathcal{A}[\cdot]_1$', '$\\rm \mathcal{A}[\cdot]_{1.5}$', \
              '$\\rm \mathcal{A}[\cdot]_2$', '$\\rm \mathcal{A}[\cdot]_3$', '$\\rm \mathcal{A}[\cdot]_4$']
    #markers = ['.', '.', '*', '*', '^', '^', '^', '^']
    for i in range(len(m_methods)):
        print(m_methods[i])
        savePath = r_path + m_methods[i]

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        Fq = []
        time = []
        for p in results:
            Fq.append(np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1))
            time.append(results[p]['time'])

        time_avg = np.mean(time, 0)
        Fq_avg = np.mean(Fq, 0)
        #r1, r2 = Get_r(Fq)
        ax.plot(time_avg, Fq_avg, linewidth=5, label=labels[i], color=colors[i])
        #ax.fill_between(time_avg, r1, r2, alpha=0.15, color=colors[i])

    ax.plot([0, max(time_avg)], [1, 1], 'k--', linewidth=5)

    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/map/'+na_state+'_time', 1, ncol=2)'''

    '''
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
    for i in range(len(m_methods)):
        print(m_methods[i])
        savePath = r_path + m_methods[i]

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        Fq = []
        epoch = 0
        for p in results:
            Fq.append(np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1))
            epoch = results[p]['epoch']

        Fq_avg = np.mean(Fq, 0)
        ax2.plot(epoch, Fq_avg, linewidth=5, label=labels[i], color=colors[i])
    ax2.plot([0, max(epoch)], [1, 1], 'k--', linewidth=5)

    Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/map/'+na_state+'_epoch', 1, ncol=2)
    #plt.show()'''

    #-----purity, Fq-----
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    labels = ['$\\rm Chol_\Delta$', '$\\rm Chol_H$', '$\\rm \mathcal{F}[\cdot]$', '$\\rm \mathcal{S}[\cdot]$', \
              '$\\rm \mathcal{A}[\cdot]_{0.5}$', '$\\rm \mathcal{A}[\cdot]_1$', '$\\rm \mathcal{A}[\cdot]_{1.5}$', \
              '$\\rm \mathcal{A}[\cdot]_2$', '$\\rm \mathcal{A}[\cdot]_3$', '$\\rm \mathcal{A}[\cdot]_4$']
    #markers = ['.', '.', '*', '*', '^', '^', '^', '^']
    for i in range(len(m_methods)):
        print(m_methods[i])
        savePath = r_path + m_methods[i]

        results = np.load(savePath + '.npy', allow_pickle=True).item()

        Fq_NN = []
        P_NN = []
        for p in results:
            P_NN.append(float(p))
        P_NN.sort()
        for p in P_NN:
            p = str(p)
            Fq_t = np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1)
            Fq_NN.append(Fq_t[-1])

        if na_state == 'random_P':
            ax.plot(P_NN, Fq_NN, linewidth=5, label=labels[i], color=colors[i])
        else:
            ax.plot(np.array(P_NN)**2*(1-1/2**10)+1/2**10, Fq_NN, linewidth=5, label=labels[i], color=colors[i])

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    Plt_set(ax, "Purity", "Quantum fidelity", 'fig/'+na_state+'/map/'+na_state+'_purity_fq', ncol=2)


#-----ex: 2 (Convergence Experiment of Random Mixed States for Different Qubits)-----
def Conv_qubit_ex_fig(na_state, state, r_path, m_path, colors):
    #--------------------1------------------
    '''
    for n_qubit in np.arange(2, 12):
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        print(n_qubit)
        savePath = r_path + 'NN_' + str(n_qubit)

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        P_all = []
        for p in results:
            P_all.append(float(p))
        P_all.sort()
        for p in P_all:
            p = str(p)
            ax.plot(results[p]['time'], np.minimum(results[p]['Fq'], 1), label=p[:4], linewidth=5)

        ax.plot([0, max(results[p]['time'])], [1, 1], 'k--', linewidth=5)
        Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/qubit/NN/'+na_state+'_time_'+str(n_qubit), 1, ncol=5, f_size=10)

        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
        for p in P_all:
            p = str(p)
            ax2.plot(results[p]['epoch'], np.minimum(results[p]['Fq'], 1), label=p[:4], linewidth=5)

        ax2.plot([0, max(results[p]['epoch'])], [1, 1], 'k--', linewidth=5)
        Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/qubit/NN/'+na_state+'_epoch_'+str(n_qubit), 1, ncol=5, f_size=10)

    #----matlab----
    #--APG--
    for n_qubit in np.arange(2, 10):
        data_mat = scio.loadmat(m_path + state + '/APG_' + str(n_qubit)+'.mat')['save_data'][0]

        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        print(n_qubit)

        max_t = 0
        P_all = []
        for i in range(len(data_mat)):
            P_all.append(data_mat[i][0][0][0])
        P_idx = np.argsort(P_all)
        for i in P_idx:
            P = P_all[i]
            stats = data_mat[i][1][0][0]
   
            ax.plot(stats[9], np.minimum(stats[3], 1), label=round(P, 2), linewidth=5)

            if max_t < np.max(stats[9]):
                max_t = np.max(stats[9])

        ax.plot([0, max_t], [1, 1], 'k--', linewidth=5)
        Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/qubit/APG/'+state+'_APG_time_'+str(n_qubit), 1, ncol=5, f_size=10)

        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
        max_e = 0
        for i in P_idx:
            P = P_all[i]
            stats = data_mat[i][1][0][0]
   
            ax2.plot(np.arange(len(stats[3])), np.minimum(stats[3], 1), label=round(P, 2), linewidth=5)

            if max_e < len(stats[3]):
                max_e = len(stats[3])

        ax2.plot([0, max_e], [1, 1], 'k--', linewidth=5)
        Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/qubit/APG/'+state+'_APG_epoch_'+str(n_qubit), 1, ncol=5, f_size=10)
        #plt.show()

    #--iMLE--
    for n_qubit in np.arange(2, 10):
        data_mat = scio.loadmat(m_path + state + '/iMLE_' + str(n_qubit)+'.mat')['save_data'][0]

        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        print(n_qubit)

        max_t = 0
        P_all = []
        for i in range(len(data_mat)):
            P_all.append(data_mat[i][0][0][0])
        P_idx = np.argsort(P_all)
        for i in P_idx:
            P = P_all[i]
            stats = data_mat[i][1][0][0]
   
            ax.plot(stats[3], np.minimum(stats[2], 1), label=round(P, 2), linewidth=5)

            if max_t < np.max(stats[3]):
                max_t = np.max(stats[3])

        ax.plot([0, max_t], [1, 1], 'k--', linewidth=5)
        Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/qubit/iMLE/'+state+'_iMLE_time_'+str(n_qubit), 1, ncol=5, f_size=10)

        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
        max_e = 0
        for i in P_idx:
            P = P_all[i]
            stats = data_mat[i][1][0][0]
   
            ax2.plot(np.arange(len(stats[2])), np.minimum(stats[2], 1), label=round(P, 2), linewidth=5)

            if max_e < len(stats[2]):
                max_e = len(stats[2])

        ax2.plot([0, max_e], [1, 1], 'k--', linewidth=5)
        Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/qubit/iMLE/'+state+'_iMLE_epoch_'+str(n_qubit), 1, ncol=5, f_size=10)
        #plt.show()


    #--------------------2------------------
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    for n_qubit in np.arange(2, 12):
        print(n_qubit)
        savePath = r_path + 'NN_' + str(n_qubit)

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        Fq = []
        time = []
        for p in results:
            Fq.append(np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1))
            time.append(results[p]['time'])

        time_avg = np.mean(time, 0)
        Fq_avg = np.mean(Fq, 0)
        #r1, r2 = Get_r(Fq)
        ax.plot(time_avg, Fq_avg, linewidth=5, label="$N$"+str(n_qubit), color=colors[n_qubit-2])
        #ax.fill_between(time_avg, r1, r2, alpha=0.2, color=colors[n_qubit-2])

    ax.plot([0, max(time_avg)], [1, 1], 'k--', linewidth=5)
    Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/qubit/NN/'+na_state+'_time', 1, ncol=2)

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
    for n_qubit in np.arange(2, 12):
        print(n_qubit)
        savePath = r_path + 'NN_' + str(n_qubit)

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        Fq = []
        epoch = 0
        for p in results:
            Fq.append(np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1))
            epoch = results[p]['epoch']

        Fq_avg = np.mean(Fq, 0)
        #r1, r2 = Get_r(Fq)
        ax2.plot(epoch, Fq_avg, linewidth=5, label="$N$"+str(n_qubit), color=colors[n_qubit-2])
        #ax2.fill_between(epoch, r1, r2, alpha=0.2, color=colors[n_qubit-2])
    
    ax2.plot([0, max(epoch)], [1, 1], 'k--', linewidth=5)
    Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/qubit/NN/'+na_state+'_epoch', 1, ncol=2)
    #plt.show()

    #----matlab----
    #--APG--
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    for n_qubit in np.arange(2, 10):
        print(n_qubit)
        data_mat = scio.loadmat(m_path + state + '/APG_' + str(n_qubit)+'.mat')['save_data'][0]       

        Fq = []
        time = []
        for i in range(len(data_mat)):
            stats = data_mat[i][1][0][0]
            if len(stats[3].ravel()) == 500:
                Fq.append(np.minimum(stats[3].ravel(), 1))
                time.append(stats[9].ravel())

        time_avg = np.mean(time, 0)
        Fq_avg = np.mean(Fq, 0)
        #r1, r2 = Get_r(Fq)
        ax.plot(time_avg, Fq_avg, linewidth=5, label="$N$"+str(n_qubit), color=colors[n_qubit-2])
        #ax.fill_between(time_avg, r1, r2, alpha=0.2, color=colors[n_qubit-2])

    ax.plot([0, max(time_avg)], [1, 1], 'k--', linewidth=5)
    Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/qubit/APG/'+state+'_APG_time', 1, ncol=2)

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
    for n_qubit in np.arange(2, 10):
        print(n_qubit)
        data_mat = scio.loadmat(m_path + state + '/APG_' + str(n_qubit)+'.mat')['save_data'][0]       

        Fq = []
        for i in range(len(data_mat)):
            stats = data_mat[i][1][0][0]
            if len(stats[3].ravel()) == 500:
                Fq.append(np.minimum(stats[3].ravel(), 1))

        Fq_avg = np.mean(Fq, 0)
        #r1, r2 = Get_r(Fq)
        ax2.plot(np.arange(500), Fq_avg, linewidth=5, label="$N$"+str(n_qubit), color=colors[n_qubit-2])
        #ax2.fill_between(np.arange(500), r1, r2, alpha=0.2, color=colors[n_qubit-2])
                
    ax2.plot([0, 500], [1, 1], 'k--', linewidth=5)
    Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/qubit/APG/'+state+'_APG_epoch', 1, ncol=2)
    #plt.show()

    #--iMLE--
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    for n_qubit in np.arange(2, 10):
        print(n_qubit)
        data_mat = scio.loadmat(m_path + state + '/iMLE_' + str(n_qubit)+'.mat')['save_data'][0]       

        Fq = []
        time = []
        for i in range(len(data_mat)):
            stats = data_mat[i][1][0][0]
            if len(stats[2].ravel()) == 500:
                Fq.append(np.minimum(stats[2].ravel(), 1))
                time.append(stats[3].ravel())

        time_avg = np.mean(time, 0)
        Fq_avg = np.mean(Fq, 0)
        #r1, r2 = Get_r(Fq)
        ax.plot(time_avg, Fq_avg, linewidth=5, label="$N$"+str(n_qubit), color=colors[n_qubit-2])
        #ax.fill_between(time_avg, r1, r2, alpha=0.2, color=colors[n_qubit-2])

    ax.plot([0, max(time_avg)], [1, 1], 'k--', linewidth=5)
    Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/qubit/iMLE/'+state+'_iMLE_time', 1, ncol=2)

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
    for n_qubit in np.arange(2, 10):
        print(n_qubit)
        data_mat = scio.loadmat(m_path + state + '/iMLE_' + str(n_qubit)+'.mat')['save_data'][0]       

        Fq = []
        for i in range(len(data_mat)):
            stats = data_mat[i][1][0][0]
            if len(stats[2].ravel()) == 500:
                Fq.append(np.minimum(stats[2].ravel(), 1))

        Fq_avg = np.mean(Fq, 0)
        #r1, r2 = Get_r(Fq)
        ax2.plot(np.arange(500), Fq_avg, linewidth=5, label="$N$"+str(n_qubit), color=colors[n_qubit-2])
        #ax2.fill_between(np.arange(500), r1, r2, alpha=0.2, color=colors[n_qubit-2])
                
    ax2.plot([0, 500], [1, 1], 'k--', linewidth=5)
    Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/qubit/iMLE/'+state+'_iMLE_epoch', 1, ncol=2)
    #plt.show()'''


    #--------------------3------------------
    Time = []
    Epoch = []
    for n_qubit in np.arange(2, 12):
        print(n_qubit)
        savePath = r_path + 'NN_' + str(n_qubit)

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        Time_99 = []
        Epoch_99 = []
        for p in results:
            Fq = np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1)
            for i in range(len(Fq)):
                if Fq[i] >= 0.99:
                    Time_99.append(results[p]['time'][i])
                    Epoch_99.append(results[p]['epoch'][i])
                    break

                if i == len(Fq) - 1:
                    Time_99.append(results[p]['time'][i])
                    Epoch_99.append(results[p]['epoch'][i])

        Time.append(Time_99)
        Epoch.append(Epoch_99)

    #----APG----
    Time_apg = []
    Epoch_apg = []
    for n_qubit in np.arange(2, 10):
        print(n_qubit)
        data_mat = scio.loadmat(m_path + state + '/APG_' + str(n_qubit)+'.mat')['save_data'][0]       

        Time_apg_99 = []
        Epoch_apg_99 = []
        for i in range(len(data_mat)):
            stats = data_mat[i][1][0][0]
            Fq = np.minimum(stats[3].ravel(), 1)
            for ii in range(len(Fq)):
                if Fq[ii] >= 0.99:
                    Time_apg_99.append(stats[9].ravel()[ii])
                    Epoch_apg_99.append(ii)
                    break

                if ii == len(Fq) - 1:
                    Time_apg_99.append(stats[9].ravel()[ii])
                    Epoch_apg_99.append(ii)

        Time_apg.append(Time_apg_99)
        Epoch_apg.append(Epoch_apg_99)

    #----iMLE----
    Time_imle = []
    Epoch_imle = []
    for n_qubit in np.arange(2, 10):
        print(n_qubit)
        data_mat = scio.loadmat(m_path + state + '/iMLE_' + str(n_qubit)+'.mat')['save_data'][0]       

        Time_imle_99 = []
        Epoch_imle_99 = []
        for i in range(len(data_mat)):
            stats = data_mat[i][1][0][0]
            Fq = np.minimum(stats[2].ravel(), 1)
            for ii in range(len(Fq)):
                if Fq[ii] >= 0.99:
                    Time_imle_99.append(stats[3].ravel()[ii])
                    Epoch_imle_99.append(ii)
                    break

                if ii == len(Fq) - 1:
                    Time_imle_99.append(stats[3].ravel()[ii])
                    Epoch_imle_99.append(ii)

        Time_imle.append(Time_imle_99)
        Epoch_imle.append(Epoch_imle_99)

    time = np.array(Time).T
    epoch = np.array(Epoch).T
    time_apg = np.array(Time_apg).T
    epoch_apg = np.array(Epoch_apg).T
    time_imle = np.array(Time_imle).T
    epoch_imle = np.array(Epoch_imle).T

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    time_avg = np.mean(time, 0)
    r1, r2 = Get_r(time)
    r1 = np.maximum(r1, 0)
    ax.plot(np.arange(2, 12), time_avg, linewidth=5, label="$\\rm Chol_H$", color=colors[9])
    ax.fill_between(np.arange(2, 12), r1, r2, alpha=0.2, color=colors[9])

    time_avg = np.mean(time_apg, 0)
    r1, r2 = Get_r(time_apg)
    r1 = np.maximum(r1, 0)
    ax.plot(np.arange(2, 10), time_avg, linewidth=5, label="$\\rm CG-APG$", color=colors[0])
    ax.fill_between(np.arange(2, 10), r1, r2, alpha=0.2, color=colors[0])

    time_avg = np.mean(time_imle, 0)
    r1, r2 = Get_r(time_imle)
    r1 = np.maximum(r1, 0)
    ax.plot(np.arange(2, 10), time_avg, linewidth=5, label="$\\rm iMLE$", color=colors[6])
    ax.fill_between(np.arange(2, 10), r1, r2, alpha=0.2, color=colors[6])

    ax.set_xticks([2, 3, 5, 7, 9, 11])
    Plt_set(ax, "Number of qubits", "Time ($s$)", 'fig/'+na_state+'/qubit/'+na_state+'_qubit_time', 2, loc=2)

    '''
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
    epoch_avg = np.mean(epoch, 0)
    r1, r2 = Get_r(epoch)
    r1 = np.maximum(r1, 0)
    ax2.plot(np.arange(2, 12), epoch_avg, linewidth=5, label="$Chol_H$", color=colors[9])
    ax2.fill_between(np.arange(2, 12), r1, r2, alpha=0.2, color=colors[9])

    epoch_avg = np.mean(epoch_apg, 0)
    r1, r2 = Get_r(epoch_apg)
    r1 = np.maximum(r1, 0)
    ax2.plot(np.arange(2, 10), epoch_avg, linewidth=5, label="$CG-APG$", color=colors[0])
    ax2.fill_between(np.arange(2, 10), r1, r2, alpha=0.2, color=colors[0])

    epoch_avg = np.mean(epoch_imle, 0)
    r1, r2 = Get_r(epoch_imle)
    r1 = np.maximum(r1, 0)
    ax2.plot(np.arange(2, 10), epoch_avg, linewidth=5, label="$iMLE$", color=colors[2])
    ax2.fill_between(np.arange(2, 10), r1, r2, alpha=0.2, color=colors[2])

    ax2.set_xticks([2, 3, 5, 7, 9, 11])
    Plt_set(ax2, "Number of qubits", "Epoch", 'fig/'+na_state+'/qubit/'+na_state+'_qubit_epoch', 2, loc=2)
    #plt.show()'''


#-----ex: 3 (Convergence Experiment of Random Mixed States for Different Samples)-----
def Conv_sample_ex_fig(na_state, state, r_path, m_path, colors):  # random_P
    #--------------------1------------------
    #--NN--
    '''
    N_Samples = np.array([1, 5, 10, 50, 100, 500, 1000, 10000])
    for N_s in N_Samples:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        print(N_s * 100)
        savePath = r_path + 'NN_S' + str(N_s)

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        P_all = []
        for p in results:
            P_all.append(float(p))
        P_all.sort()
        for p in P_all:
            p = str(p)
            ax.plot(results[p]['time'], np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1), label=p[:4], linewidth=2.5)

        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.plot([0, max(results[p]['time'])], [1, 1], 'k--', linewidth=2.5)
        Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/sample/NN/'+na_state+'_NN_sample_time_'+str(N_s), 1, ncol=5, f_size=10)

        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
        for p in P_all:
            p = str(p)
            ax2.plot(results[p]['epoch'], np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1), label=p[:4], linewidth=2.5)

        ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax2.plot([0, max(results[p]['epoch'])], [1, 1], 'k--', linewidth=2.5)
        Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/sample/NN/'+na_state+'_NN_sample_epoch_'+str(N_s), 1, ncol=5, f_size=10)
        #plt.show()

    #----matlab----
    #--APG--
    N_Samples = np.array([1, 5, 10, 50, 100, 500, 1000, 10000])
    for N_s in N_Samples:
        data_mat = scio.loadmat(m_path + state + '/APG_S' + str(N_s)+'.mat')['save_data'][0]

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        print(N_s * 100)

        max_t = 0
        P_all = []
        for i in range(len(data_mat)):
            P_all.append(data_mat[i][0][0][0])
        P_idx = np.argsort(P_all)
        for i in P_idx:
            P = P_all[i]
            stats = data_mat[i][1][0][0]
   
            ax.plot(stats[9], np.minimum(stats[3], 1), label=round(P, 2), linewidth=2.5)

            if max_t < np.max(stats[9]):
                max_t = np.max(stats[9])

        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.plot([0, max_t], [1, 1], 'k--', linewidth=2.5)
        Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/sample/APG/'+state+'_APG_sample_time_'+str(N_s), 1, ncol=5, f_size=10)

        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
        max_e = 0
        for i in P_idx:
            P = P_all[i]
            stats = data_mat[i][1][0][0]
   
            ax2.plot(np.arange(len(stats[3])), np.minimum(stats[3], 1), label=round(P, 2), linewidth=2.5)

            if max_e < len(stats[3]):
                max_e = len(stats[3])

        ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax2.plot([0, max_e], [1, 1], 'k--', linewidth=2.5)
        Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/sample/APG/'+state+'_APG_sample_epoch_'+str(N_s), 1, ncol=5, f_size=10)
        #plt.show()

    #--iMLE--
    N_Samples = np.array([1, 5, 10, 50, 100, 500, 1000, 10000])
    for N_s in N_Samples:
        data_mat = scio.loadmat(m_path + state + '/iMLE_S' + str(N_s)+'.mat')['save_data'][0]

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        print(N_s * 100)

        max_t = 0
        P_all = []
        for i in range(len(data_mat)):
            P_all.append(data_mat[i][0][0][0])
        P_idx = np.argsort(P_all)
        for i in P_idx:
            P = P_all[i]
            stats = data_mat[i][1][0][0]
   
            ax.plot(stats[3], np.minimum(stats[2], 1), label=round(P, 2), linewidth=2.5)

            if max_t < np.max(stats[3]):
                max_t = np.max(stats[3])

        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.plot([0, max_t], [1, 1], 'k--', linewidth=2.5)
        Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/sample/iMLE/'+state+'_iMLE_sample_time_'+str(N_s), 1, ncol=5, f_size=10)

        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
        max_e = 0
        for i in P_idx:
            P = P_all[i]
            stats = data_mat[i][1][0][0]
   
            ax2.plot(np.arange(len(stats[2])), np.minimum(stats[2], 1), label=round(P, 2), linewidth=2.5)

            if max_e < len(stats[2]):
                max_e = len(stats[2])

        ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax2.plot([0, max_e], [1, 1], 'k--', linewidth=2.5)
        Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/sample/iMLE/'+state+'_iMLE_sample_epoch_'+str(N_s), 1, ncol=5, f_size=10)
        #plt.show()'''


    #--------------------2------------------
    #--NN--
    '''
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    N_Samples = np.array([1, 5, 10, 50, 100, 500, 1000, 10000])
    N_i = -1
    for N_s in N_Samples:
        print(N_s)
        N_i += 1
        savePath = r_path + 'NN_S' + str(N_s)

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        Fq = []
        time = []
        for p in results:
            Fq.append(np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1))
            time.append(results[p]['time'])

        time_avg = np.mean(time, 0)
        Fq_avg = np.mean(Fq, 0)
        r1, r2 = Get_r(Fq)
        r1 = np.maximum(r1, 0)
        r2 = np.minimum(r2, 1)
        ax.plot(time_avg, Fq_avg, linewidth=2.5, label="$S$"+str(N_s * 100), color=colors[N_i])
        #ax.fill_between(time_avg, r1, r2, alpha=0.2, color=colors[N_i])

    ax.plot([0, max(time_avg)], [1, 1], 'k--', linewidth=2.5)
    Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/sample/NN/'+na_state+'_NN_sample_time', 1, ncol=2)

    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
    N_Samples = np.array([1, 5, 10, 50, 100, 500, 1000, 10000])
    N_i = -1
    for N_s in N_Samples:
        print(N_s)
        N_i += 1
        savePath = r_path + 'NN_S' + str(N_s)

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        Fq = []
        epoch = 0
        for p in results:
            Fq.append(np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1))
            epoch = results[p]['epoch']

        Fq_avg = np.mean(Fq, 0)
        r1, r2 = Get_r(Fq)
        r1 = np.maximum(r1, 0)
        r2 = np.minimum(r2, 1)
        ax2.plot(epoch, Fq_avg, linewidth=2.5, label="$S$"+str(N_s*100), color=colors[N_i])
        #ax2.fill_between(epoch, r1, r2, alpha=0.2, color=colors[N_i])
    
    ax2.plot([0, max(epoch)], [1, 1], 'k--', linewidth=2.5)
    Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/sample/NN/'+na_state+'_NN_sample_epoch', 1, ncol=2)
    #plt.show()

    #----matlab----
    #--APG--
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    N_Samples = np.array([1, 5, 10, 50, 100, 500, 1000, 10000])
    N_i = -1
    for N_s in N_Samples:
        print(N_s)
        N_i += 1
        data_mat = scio.loadmat(m_path + state + '/APG_S' + str(N_s)+'.mat')['save_data'][0]       

        Fq = []
        time = []
        for i in range(len(data_mat)):
            stats = data_mat[i][1][0][0]
            if len(stats[3].ravel()) == 500:
                Fq.append(np.minimum(stats[3].ravel(), 1))
                time.append(stats[9].ravel())

        time_avg = np.mean(time, 0)
        Fq_avg = np.mean(Fq, 0)
        r1, r2 = Get_r(Fq)
        r1 = np.maximum(r1, 0)
        r2 = np.minimum(r2, 1)
        ax.plot(time_avg, Fq_avg, linewidth=2.5, label="$S$"+str(N_s*100), color=colors[N_i])
        #ax.fill_between(time_avg, r1, r2, alpha=0.2, color=colors[N_i])

    ax.plot([0, max(time_avg)], [1, 1], 'k--', linewidth=2.5)
    Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/sample/APG/'+state+'_APG_sample_time', 1, ncol=2)

    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
    N_Samples = np.array([1, 5, 10, 50, 100, 500, 1000, 10000])
    N_i = -1
    for N_s in N_Samples:
        print(N_s)
        N_i += 1
        data_mat = scio.loadmat(m_path + state + '/APG_S' + str(N_s)+'.mat')['save_data'][0]       

        Fq = []
        for i in range(len(data_mat)):
            stats = data_mat[i][1][0][0]
            if len(stats[3].ravel()) == 500:
                Fq.append(np.minimum(stats[3].ravel(), 1))

        Fq_avg = np.mean(Fq, 0)
        r1, r2 = Get_r(Fq)
        r1 = np.maximum(r1, 0)
        r2 = np.minimum(r2, 1)
        ax2.plot(np.arange(500), Fq_avg, linewidth=2.5, label="$S$"+str(N_s*100), color=colors[N_i])
        #ax2.fill_between(np.arange(500), r1, r2, alpha=0.2, color=colors[N_i])
                
    ax2.plot([0, 500], [1, 1], 'k--', linewidth=2.5)
    Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/sample/APG/'+state+'_APG_sample_epoch', 1, ncol=2)
    #plt.show()

    #--iMLE--
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    N_Samples = np.array([1, 5, 10, 50, 100, 500, 1000, 10000])
    N_i = -1
    for N_s in N_Samples:
        print(N_s)
        N_i += 1
        data_mat = scio.loadmat(m_path + state + '/iMLE_S' + str(N_s)+'.mat')['save_data'][0]       

        Fq = []
        time = []
        for i in range(len(data_mat)):
            stats = data_mat[i][1][0][0]
            if len(stats[2].ravel()) == 500:
                Fq.append(np.minimum(stats[2].ravel(), 1))
                time.append(stats[3].ravel())

        time_avg = np.mean(time, 0)
        Fq_avg = np.mean(Fq, 0)
        r1, r2 = Get_r(Fq)
        r1 = np.maximum(r1, 0)
        r2 = np.minimum(r2, 1)
        ax.plot(time_avg, Fq_avg, linewidth=2.5, label="$S$"+str(N_s*100), color=colors[N_i])
        #ax.fill_between(time_avg, r1, r2, alpha=0.2, color=colors[N_i])

    ax.plot([0, max(time_avg)], [1, 1], 'k--', linewidth=2.5)
    Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/sample/iMLE/'+state+'_iMLE_sample_time', 1, ncol=2)

    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
    N_Samples = np.array([1, 5, 10, 50, 100, 500, 1000, 10000])
    N_i = -1
    for N_s in N_Samples:
        print(N_s)
        N_i += 1
        data_mat = scio.loadmat(m_path + state + '/APG_S' + str(N_s)+'.mat')['save_data'][0]       

        Fq = []
        for i in range(len(data_mat)):
            stats = data_mat[i][1][0][0]
            if len(stats[2].ravel()) == 500:
                Fq.append(np.minimum(stats[2].ravel(), 1))

        Fq_avg = np.mean(Fq, 0)
        r1, r2 = Get_r(Fq)
        r1 = np.maximum(r1, 0)
        r2 = np.minimum(r2, 1)
        ax2.plot(np.arange(500), Fq_avg, linewidth=2.5, label="$S$"+str(N_s*100), color=colors[N_i])
        #ax2.fill_between(np.arange(500), r1, r2, alpha=0.2, color=colors[N_i])
                
    ax2.plot([0, 500], [1, 1], 'k--', linewidth=2.5)
    Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/sample/iMLE/'+state+'_iMLE_sample_epoch', 1, ncol=2)
    #plt.show()'''

    #--------------------3:sample-fidelity------------------
    #--NN--
    '''
    Fq_NN = []
    N_Samples = np.array([1, 5, 10, 50, 100, 500, 1000, 10000])
    for N_s in N_Samples:
        print(N_s)
        savePath = r_path + 'NN_S' + str(N_s)

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        Fq = []
        for p in results:
            Fq_t = np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1)
            Fq.append(Fq_t[-1])

        Fq_NN.append(Fq)

    #----APG----
    Fq_apg = []
    N_Samples_apg = np.array([1, 5, 10, 50, 100, 500, 1000, 10000])
    for N_s in N_Samples_apg:
        print(N_s)
        data_mat = scio.loadmat(m_path + state + '/APG_S' + str(N_s)+'.mat')['save_data'][0]       

        Fq_apg_t = []
        for i in range(len(data_mat)):
            stats = data_mat[i][1][0][0]
            Fq_apg_t.append(stats[3].ravel()[-1])

        Fq_apg.append(Fq_apg_t)

    #----iMLE----
    Fq_imle = []
    N_Samples_imle = np.array([1, 5, 10, 50, 100, 500, 1000, 10000])
    for N_s in N_Samples_imle:
        print(N_s)
        data_mat = scio.loadmat(m_path + state + '/iMLE_S' + str(N_s)+'.mat')['save_data'][0]       

        Fq_imle_t = []
        for i in range(len(data_mat)):
            stats = data_mat[i][1][0][0]
            Fq_imle_t.append(stats[2].ravel()[-1])

        Fq_imle.append(Fq_imle_t)

    Fq_NN = np.array(Fq_NN).T
    Fq_apg = np.array(Fq_apg).T
    Fq_imle = np.array(Fq_imle).T

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    Fq_avg = np.mean(Fq_NN, 0)
    r1, r2 = Get_r(Fq_NN)
    r1 = np.maximum(r1, 0)
    r2 = np.minimum(r2, 1)
    ax.plot(N_Samples*100, Fq_avg, linewidth=5, label="$\\rm Chol_H$", color=colors[9])
    ax.fill_between(N_Samples*100, r1, r2, alpha=0.2, color=colors[9])

    Fq_avg = np.mean(Fq_apg, 0)
    r1, r2 = Get_r(Fq_apg)
    r1 = np.maximum(r1, 0)
    r2 = np.minimum(r2, 1)
    ax.plot(N_Samples_apg*100, Fq_avg, linewidth=5, label="$\\rm CG-APG$", color=colors[0])
    ax.fill_between(N_Samples_apg*100, r1, r2, alpha=0.2, color=colors[0])

    Fq_avg = np.mean(Fq_imle, 0)
    r1, r2 = Get_r(Fq_imle)
    r1 = np.maximum(r1, 0)
    r2 = np.minimum(r2, 1)
    ax.plot(N_Samples_imle*100, Fq_avg, linewidth=5, label="$\\rm iMLE$", color=colors[6])
    ax.fill_between(N_Samples_imle*100, r1, r2, alpha=0.2, color=colors[6])

    ax.plot([0, N_Samples[-1]*100], [1, 1], 'k--', linewidth=5)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    Plt_set(ax, "Number of samples", "Quantum fidelity", 'fig/'+na_state+'/sample/'+na_state+'_sample_fq', 1, loc=4)
    #plt.show()'''

    #--------------------4:sample-time------------------
    #--NN--
    '''
    Time = []
    N_Samples = np.array([1, 5, 10, 50, 100, 500, 1000, 10000])
    for N_s in N_Samples:
        print(N_s)
        savePath = r_path + 'NN_S' + str(N_s)

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        Time_99 = []
        for p in results:
            Fq = np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1)
            for i in range(len(Fq)):
                if Fq[i] >= 0.99:
                    Time_99.append(results[p]['time'][i])
                    break

                if i == len(Fq) - 1:
                    Time_99.append(results[p]['time'][i])

        Time.append(Time_99)

    #----APG----
    Time_apg = []
    N_Samples_apg = np.array([1, 5, 10, 50, 100, 500, 1000, 10000])
    for N_s in N_Samples_apg:
        print(N_s)
        data_mat = scio.loadmat(m_path + state + '/APG_S' + str(N_s)+'.mat')['save_data'][0]         

        Time_apg_99 = []
        for i in range(len(data_mat)):
            stats = data_mat[i][1][0][0]
            Fq = np.minimum(stats[3].ravel(), 1)
            for ii in range(len(Fq)):
                if Fq[ii] >= 0.99:
                    Time_apg_99.append(stats[9].ravel()[ii])
                    break

                if ii == len(Fq) - 1:
                    Time_apg_99.append(stats[9].ravel()[ii])

        Time_apg.append(Time_apg_99)

    #----iMLE----
    Time_imle = []
    N_Samples_imle = np.array([1, 5, 10, 50, 100, 500, 1000, 10000])
    for N_s in N_Samples_imle:
        print(N_s)
        data_mat = scio.loadmat(m_path + state + '/iMLE_S' + str(N_s)+'.mat')['save_data'][0]   

        Time_imle_99 = []
        for i in range(len(data_mat)):
            stats = data_mat[i][1][0][0]
            Fq = np.minimum(stats[2].ravel(), 1)
            for ii in range(len(Fq)):
                if Fq[ii] >= 0.99:
                    Time_imle_99.append(stats[3].ravel()[ii])
                    break

                if ii == len(Fq) - 1:
                    Time_imle_99.append(stats[3].ravel()[ii])

        Time_imle.append(Time_imle_99)

    time = np.array(Time).T
    time_apg = np.array(Time_apg).T
    time_imle = np.array(Time_imle).T

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    time_avg = np.mean(time, 0)
    r1, r2 = Get_r(time)
    r1 = np.maximum(r1, 0)
    ax.plot(N_Samples*100, time_avg, linewidth=5, label="$\\rm Chol_H$", color=colors[9])
    ax.fill_between(N_Samples*100, r1, r2, alpha=0.2, color=colors[9])

    time_avg = np.mean(time_apg, 0)
    r1, r2 = Get_r(time_apg)
    r1 = np.maximum(r1, 0)
    ax.plot(N_Samples_apg*100, time_avg, linewidth=5, label="$\\rm CG-APG$", color=colors[0])
    ax.fill_between(N_Samples_apg*100, r1, r2, alpha=0.2, color=colors[0])

    time_avg = np.mean(time_imle, 0)
    r1, r2 = Get_r(time_imle)
    r1 = np.maximum(r1, 0)
    ax.plot(N_Samples_imle*100, time_avg, linewidth=5, label="$\\rm iMLE$", color=colors[6])
    ax.fill_between(N_Samples_imle*100, r1, r2, alpha=0.2, color=colors[6])

    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    Plt_set(ax, "Number of samples", "Time ($s$)", 'fig/'+na_state+'/sample/'+na_state+'_sample_time', 3, loc=3)'''

    #--------------------5:Fq-P------------------
    #--NN--
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    labels = [4, 5, 6]
    N_samples = [100, 1000, 10000]
    lines = [':', '--', '-']
    for j in range(len(N_samples)):
        N_s = N_samples[j]
        savePath = r_path + 'NN_S' + str(N_s)
        results = np.load(savePath + '.npy', allow_pickle=True).item()
        Fq_NN = []
        P_NN = []
        for p in results:
            P_NN.append(float(p))
        P_NN.sort()
        for p in P_NN:
            p = str(p)
            Fq_t = np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1)
            Fq_NN.append(Fq_t[-1])

        #----APG----
        data_mat = scio.loadmat(m_path + state + '/APG_S' + str(N_s)+'.mat')['save_data'][0]       
        Fq_apg = []
        P_apg = []
        P_all = []
        for i in range(len(data_mat)):
            P_all.append(data_mat[i][0][0][0])
        P_idx = np.argsort(P_all)
        for i in P_idx:
            P_apg.append(P_all[i])
            stats = data_mat[i][1][0][0]
            Fq_apg.append(stats[3].ravel()[-1])

        #----iMLE----
        data_mat = scio.loadmat(m_path + state + '/iMLE_S' + str(N_s)+'.mat')['save_data'][0]       
        Fq_imle = []
        P_imle = []
        P_all = []
        for i in range(len(data_mat)):
            P_all.append(data_mat[i][0][0][0])
        P_idx = np.argsort(P_all)
        for i in P_idx:
            P_imle.append(P_all[i])
            stats = data_mat[i][1][0][0]
            Fq_imle.append(stats[2].ravel()[-1])

        ax.plot(np.array(P_NN), Fq_NN, linewidth=5, label='$\\rm Chol_H$_'+str(labels[j]), color=colors[9], linestyle=lines[j])
        ax.plot(np.array(P_apg), Fq_apg, linewidth=5, label='$\\rm CG-APG$_'+str(labels[j]), color=colors[0], linestyle=lines[j])  # 6
        ax.plot(np.array(P_imle), Fq_imle, linewidth=5, label='$\\rm iMLE$_'+str(labels[j]), color=colors[6], linestyle=lines[j])  # 4 

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.plot([0, 1], [1, 1], 'k--', linewidth=5)
    Plt_set(ax, "Purity", "Quantum fidelity", 'fig/'+na_state+'/sample/'+na_state+'_sample_purity_fq', 0, loc=4, f_size=18)

#-----ex: 4 (Convergence Experiment of Random Mixed States for FNN and CNN in Different Qubits)-----
def Conv_CNN_qubit_ex_fig(na_state, state, r_path, m_path, colors):
    #--------------------1------------------
    #---CNN---
    for n_qubit in np.arange(4, 11):
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        print(n_qubit)
        savePath = r_path + 'CNN_' + str(n_qubit)

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        P_all = []
        for p in results:
            P_all.append(float(p))
        P_all.sort()
        for p in P_all:
            p = str(p)
            ax.plot(results[p]['time'], np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1), label=p[:4], linewidth=5)

        ax.plot([0, max(results[p]['time'])], [1, 1], 'k--', linewidth=5)
        Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/qubit/CNN/'+na_state+'_time_'+str(n_qubit), 1, ncol=5, f_size=10)

        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
        for p in P_all:
            p = str(p)
            ax2.plot(results[p]['epoch'], np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1), label=p[:4], linewidth=5)

        ax2.plot([0, max(results[p]['epoch'])], [1, 1], 'k--', linewidth=5)
        Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/qubit/CNN/'+na_state+'_epoch_'+str(n_qubit), 1, ncol=5, f_size=10)


    #--------------------2------------------
    #---FNN---
    '''
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    for n_qubit in np.arange(4, 11):
        print(n_qubit)
        savePath = r_path + 'NN_' + str(n_qubit)

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        Fq = []
        time = []
        for p in results:
            Fq.append(np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1))
            time.append(results[p]['time'])

        time_avg = np.mean(time, 0)
        Fq_avg = np.mean(Fq, 0)
        #r1, r2 = Get_r(Fq)
        ax.plot(time_avg, Fq_avg, linewidth=5, label="$N$"+str(n_qubit), color=colors[n_qubit-2])
        #ax.fill_between(time_avg, r1, r2, alpha=0.2, color=colors[n_qubit-2])

    ax.plot([0, max(time_avg)], [1, 1], 'k--', linewidth=5)
    Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/qubit/NN/'+na_state+'_time', 1, ncol=2)

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
    for n_qubit in np.arange(4, 11):
        print(n_qubit)
        savePath = r_path + 'NN_' + str(n_qubit)

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        Fq = []
        epoch = 0
        for p in results:
            Fq.append(np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1))
            epoch = results[p]['epoch']

        Fq_avg = np.mean(Fq, 0)
        #r1, r2 = Get_r(Fq)
        ax2.plot(epoch, Fq_avg, linewidth=5, label="$N$"+str(n_qubit), color=colors[n_qubit-2])
        #ax2.fill_between(epoch, r1, r2, alpha=0.2, color=colors[n_qubit-2])
    
    ax2.plot([0, max(epoch)], [1, 1], 'k--', linewidth=5)
    Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/qubit/NN/'+na_state+'_epoch', 1, ncol=2)'''
    #plt.show()

    #---CNN---
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    for n_qubit in np.arange(4, 11):
        print(n_qubit)
        savePath = r_path + 'CNN_' + str(n_qubit)

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        Fq = []
        time = []
        for p in results:
            Fq.append(np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1))
            time.append(results[p]['time'])

        time_avg = np.mean(time, 0)
        Fq_avg = np.mean(Fq, 0)
        #r1, r2 = Get_r(Fq)
        ax.plot(time_avg, Fq_avg, linewidth=5, label="$N$"+str(n_qubit), color=colors[n_qubit-2])
        #ax.fill_between(time_avg, r1, r2, alpha=0.2, color=colors[n_qubit-2])

    ax.plot([0, max(time_avg)], [1, 1], 'k--', linewidth=5)
    Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/qubit/CNN/'+na_state+'_time', 1, ncol=2)

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))
    for n_qubit in np.arange(4, 11):
        print(n_qubit)
        savePath = r_path + 'CNN_' + str(n_qubit)

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        Fq = []
        epoch = 0
        for p in results:
            Fq.append(np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1))
            epoch = results[p]['epoch']

        Fq_avg = np.mean(Fq, 0)
        #r1, r2 = Get_r(Fq)
        ax2.plot(epoch, Fq_avg, linewidth=5, label="$N$"+str(n_qubit), color=colors[n_qubit-2])
        #ax2.fill_between(epoch, r1, r2, alpha=0.2, color=colors[n_qubit-2])
    
    ax2.plot([0, max(epoch)], [1, 1], 'k--', linewidth=5)
    Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/qubit/CNN/'+na_state+'_epoch', 1, ncol=2)
    #plt.show()


    #--------------------3------------------
    #---FNN---
    Time = []
    Epoch = []
    for n_qubit in np.arange(4, 11):
        print(n_qubit)
        savePath = r_path + 'NN_' + str(n_qubit)

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        Time_99 = []
        Epoch_99 = []
        for p in results:
            Fq = np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1)
            for i in range(len(Fq)):
                if Fq[i] >= 0.99:
                    Time_99.append(results[p]['time'][i])
                    Epoch_99.append(results[p]['epoch'][i])
                    break

                if i == len(Fq) - 1:
                    Time_99.append(results[p]['time'][i])
                    Epoch_99.append(results[p]['epoch'][i])

        Time.append(Time_99)
        Epoch.append(Epoch_99)

    #---CNN---
    Time_C = []
    Epoch_C = []
    for n_qubit in np.arange(4, 11):
        print(n_qubit)
        savePath = r_path + 'CNN_' + str(n_qubit)

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        Time_99 = []
        Epoch_99 = []
        for p in results:
            Fq = np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1)
            for i in range(len(Fq)):
                if Fq[i] >= 0.99:
                    Time_99.append(results[p]['time'][i])
                    Epoch_99.append(results[p]['epoch'][i])
                    break

                if i == len(Fq) - 1:
                    Time_99.append(results[p]['time'][i])
                    Epoch_99.append(results[p]['epoch'][i])

        Time_C.append(Time_99)
        Epoch_C.append(Epoch_99)

    time = np.array(Time).T
    epoch = np.array(Epoch).T
    time_C = np.array(Time_C).T
    epoch_C = np.array(Epoch_C).T

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    time_avg = np.mean(time, 0)
    r1, r2 = Get_r(time)
    r1 = np.maximum(r1, 0)
    ax.plot(np.arange(4, 11), time_avg, linewidth=5, label="$\\rm FNN$", color=colors[9])
    ax.fill_between(np.arange(4, 11), r1, r2, alpha=0.2, color=colors[9])

    time_avg = np.mean(time_C, 0)
    r1, r2 = Get_r(time_C)
    r1 = np.maximum(r1, 0)
    ax.plot(np.arange(4, 11), time_avg, linewidth=5, label="$\\rm CNN$", color=colors[0])
    ax.fill_between(np.arange(4, 11), r1, r2, alpha=0.2, color=colors[0])

    ax.set_xticks([4, 5, 6, 7, 8, 9, 10])
    Plt_set(ax, "Number of qubits", "Time ($s$)", 'fig/'+na_state+'/qubit/'+na_state+'_FNN_CNN_qubit_time', 2, loc=2)

#-----ex: 5 (Convergence Experiment of Random Mixed States for Different Samples)-----
def Conv_depolar_ex_fig(na_state, state, r_path, m_path, colors):
    #--------------------1------------------
    #--NN--
    '''
    N_Samples = np.array([100, 1000, 10000])
    for N_s in N_Samples:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        print(N_s * 100)
        savePath = r_path + 'NN_S_N' + str(N_s)

        results = np.load(savePath + '.npy', allow_pickle=True).item()
        P_all = []
        for p in results:
            P_all.append(float(p))
        P_all.sort()
        for p in P_all:
            p = str(p)
            ax.plot(results[p]['time'], np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1), label=p[:4], linewidth=2.5)

        ax.plot([0, max(results[p]['time'])], [1, 1], 'k--', linewidth=2.5)
        Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/depolar/NN/'+na_state+'_NN_depolar_time_'+str(N_s), 1, ncol=5, f_size=10)

        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
        for p in P_all:
            p = str(p)
            ax2.plot(results[p]['epoch'], np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1), label=p[:4], linewidth=2.5)

        ax2.plot([0, max(results[p]['epoch'])], [1, 1], 'k--', linewidth=2.5)
        Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/depolar/NN/'+na_state+'_NN_depolar_epoch_'+str(N_s), 1, ncol=5, f_size=10)
        #plt.show()

    #----matlab----
    #--APG--
    N_Samples = np.array([100, 1000, 10000])
    for N_s in N_Samples:
        data_mat = scio.loadmat(m_path + state + '/APG_S_N' + str(N_s)+'.mat')['save_data'][0]

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        print(N_s * 100)

        max_t = 0
        P_all = []
        for i in range(len(data_mat)):
            P_all.append(data_mat[i][0][0][0])
        P_idx = np.argsort(P_all)
        for i in P_idx:
            P = P_all[i]
            stats = data_mat[i][1][0][0]
   
            ax.plot(stats[9], np.minimum(stats[3], 1), label=round(P, 2), linewidth=2.5)

            if max_t < np.max(stats[9]):
                max_t = np.max(stats[9])

        ax.plot([0, max_t], [1, 1], 'k--', linewidth=2.5)
        Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/depolar/APG/'+state+'_APG_depolar_time_'+str(N_s), 1, ncol=5, f_size=10)

        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
        max_e = 0
        for i in P_idx:
            P = P_all[i]
            stats = data_mat[i][1][0][0]
   
            ax2.plot(np.arange(len(stats[3])), np.minimum(stats[3], 1), label=round(P, 2), linewidth=2.5)

            if max_e < len(stats[3]):
                max_e = len(stats[3])

        ax2.plot([0, max_e], [1, 1], 'k--', linewidth=2.5)
        Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/depolar/APG/'+state+'_APG_depolar_epoch_'+str(N_s), 1, ncol=5, f_size=10)
        #plt.show()

    #--iMLE--
    N_Samples = np.array([100, 1000, 10000])
    for N_s in N_Samples:
        data_mat = scio.loadmat(m_path + state + '/iMLE_S_N' + str(N_s)+'.mat')['save_data'][0]

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        print(N_s * 100)

        max_t = 0
        P_all = []
        for i in range(len(data_mat)):
            P_all.append(data_mat[i][0][0][0])
        P_idx = np.argsort(P_all)
        for i in P_idx:
            P = P_all[i]
            stats = data_mat[i][1][0][0]
   
            ax.plot(stats[3], np.minimum(stats[2], 1), label=round(P, 2), linewidth=2.5)

            if max_t < np.max(stats[3]):
                max_t = np.max(stats[3])

        ax.plot([0, max_t], [1, 1], 'k--', linewidth=2.5)
        Plt_set(ax, "Time ($s$)", "Quantum fidelity", 'fig/'+na_state+'/depolar/iMLE/'+state+'_iMLE_depolar_time_'+str(N_s), 1, ncol=5, f_size=10)

        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
        max_e = 0
        for i in P_idx:
            P = P_all[i]
            stats = data_mat[i][1][0][0]
   
            ax2.plot(np.arange(len(stats[2])), np.minimum(stats[2], 1), label=round(P, 2), linewidth=2.5)

            if max_e < len(stats[2]):
                max_e = len(stats[2])

        ax2.plot([0, max_e], [1, 1], 'k--', linewidth=2.5)
        Plt_set(ax2, "Epoch", "Quantum fidelity", 'fig/'+na_state+'/depolar/iMLE/'+state+'_iMLE_depolar_epoch_'+str(N_s), 1, ncol=5, f_size=10)
        #plt.show()'''

    #--------------------3:Fq-P------------------
    #--NN--
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    labels = [4, 5, 6]
    N_samples = [100, 1000, 10000]
    lines = [':', '--', '-']
    for j in range(len(N_samples)):
        N_s = N_samples[j]
        savePath = r_path + 'NN_S_N' + str(N_s)
        results = np.load(savePath + '.npy', allow_pickle=True).item()

        Fq_NN = []
        P_NN = []
        for p in results:
            P_NN.append(float(p))
        P_NN.sort()
        for p in P_NN:
            p = str(p)
            Fq_t = np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1)
            Fq_NN.append(Fq_t[-1])

        #----APG----
        data_mat = scio.loadmat(m_path + state + '/APG_S_N' + str(N_s)+'.mat')['save_data'][0]       
        Fq_apg = []
        P_apg = []
        P_all = []
        for i in range(len(data_mat)):
            P_all.append(data_mat[i][0][0][0])
        P_idx = np.argsort(P_all)
        for i in P_idx:
            P_apg.append(P_all[i])
            stats = data_mat[i][1][0][0]
            Fq_apg.append(stats[3].ravel()[-1])

        #----iMLE----
        data_mat = scio.loadmat(m_path + state + '/iMLE_S_N' + str(N_s)+'.mat')['save_data'][0]       
        Fq_imle = []
        P_imle = []
        P_all = []
        for i in range(len(data_mat)):
            P_all.append(data_mat[i][0][0][0])
        P_idx = np.argsort(P_all)
        for i in P_idx:
            P_imle.append(P_all[i])
            stats = data_mat[i][1][0][0]
            Fq_imle.append(stats[2].ravel()[-1])

        ax.plot(np.array(P_NN), Fq_NN, linewidth=5, label='$\\rm Chol_H$_'+str(labels[j]), color=colors[9], linestyle=lines[j])
        ax.plot(np.array(P_apg), Fq_apg, linewidth=5, label='$\\rm CG-APG$_'+str(labels[j]), color=colors[0], linestyle=lines[j])  # 6
        ax.plot(np.array(P_imle), Fq_imle, linewidth=5, label='$\\rm iMLE$_'+str(labels[j]), color=colors[6], linestyle=lines[j])  # 4 

    ax.plot(np.array(P_NN), 1 - (1 - 1 / 2**8) * np.array(P_NN), 'k--', linewidth=5)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    Plt_set(ax, "$\lambda$", "Quantum fidelity", 'fig/'+na_state+'/depolar/'+na_state+'_depolar_lambda_fq', 0, loc=1, f_size=18)

    #--------------------4:Time-P------------------
    #--NN--
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    labels = [4, 5, 6]
    N_samples = [100, 1000, 10000]
    lines = [':', '--', '-']
    for j in range(len(N_samples)):
        N_s = N_samples[j]
        savePath = r_path + 'NN_S_N' + str(N_s)
        results = np.load(savePath + '.npy', allow_pickle=True).item()

        Time_NN = []
        P_NN = []
        for p in results:
            P_NN.append(float(p))
        P_NN.sort()
        for p in P_NN:
            p = str(p)
            Fq_t = np.minimum(torch.tensor(results[p]['Fq']).numpy(), 1)
            for i in range(len(Fq_t)):
                if abs((1 - (1 - 1 / 2**8) * float(p)) - Fq_t[i]) <= 0.005:
                    Time_NN.append(results[p]['time'][i])
                    break

                if i == len(Fq_t) - 1:
                    Time_NN.append(results[p]['time'][i])

        #----APG----
        data_mat = scio.loadmat(m_path + state + '/APG_S_N' + str(N_s)+'.mat')['save_data'][0]       
        P_apg = []
        Time_apg = []
        P_all = []
        for i in range(len(data_mat)):
            P_all.append(data_mat[i][0][0][0])
        P_idx = np.argsort(P_all)
        for i in P_idx:
            P_apg.append(P_all[i])
            stats = data_mat[i][1][0][0]
            Fq = np.minimum(stats[3].ravel(), 1)
            for ii in range(len(Fq)):
                if abs((1 - (1 - 1 / 2**8) * P_all[i]) - Fq[ii]) <= 0.005:
                    Time_apg.append(stats[9].ravel()[ii])
                    break

                if ii == len(Fq) - 1:
                    Time_apg.append(stats[9].ravel()[ii])

        #----iMLE----
        data_mat = scio.loadmat(m_path + state + '/iMLE_S_N' + str(N_s)+'.mat')['save_data'][0]       
        Time_imle = []
        P_imle = []
        P_all = []
        for i in range(len(data_mat)):
            P_all.append(data_mat[i][0][0][0])
        P_idx = np.argsort(P_all)
        for i in P_idx:
            P_imle.append(P_all[i])
            stats = data_mat[i][1][0][0]
            Fq = np.minimum(stats[2].ravel(), 1)
            for ii in range(len(Fq)):
                if abs((1 - (1 - 1 / 2**8) * P_all[i]) - Fq[ii]) <= 0.005:
                    Time_imle.append(stats[3].ravel()[ii])
                    break

                if ii == len(Fq) - 1:
                    Time_imle.append(stats[3].ravel()[ii])

        ax.plot(np.array(P_NN), Time_NN, linewidth=5, label='$\\rm Chol_H$_'+str(labels[j]), color=colors[9], linestyle=lines[j])
        ax.plot(np.array(P_apg), Time_apg, linewidth=5, label='$\\rm CG-APG$_'+str(labels[j]), color=colors[0], linestyle=lines[j])  # 6
        ax.plot(np.array(P_imle), Time_imle, linewidth=5, label='$\\rm iMLE$_'+str(labels[j]), color=colors[6], linestyle=lines[j])  # 4 

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    Plt_set(ax, "$\lambda$", "Time ($s$)", 'fig/'+na_state+'/depolar/'+na_state+'_depolar_lambda_time', 2, loc=4, f_size=18)


if __name__ == '__main__':
    colors = ['#75bbfd', 'magenta', '#658b38', '#c79fef', '#06c2ac', 'cyan', '#ff9408', '#430541', 'blue', '#fb2943']
    na_states = ['random_P'] #, 'W_P', 'Product_P', 'GHZi_P']  # , 'random_P'
    states = ['R'] #, 'W', 'P', 'I']  # , 'R'

    for i in range(len(na_states)):
        na_state = na_states[i]
        state = states[i]

        print('-----state:', na_state)
        r_path = 'result/' + na_state + '/'
        m_path = '../../models/MLE/result/'
        Conv_sample_ex_fig(na_state, state, r_path, m_path, colors)

    #Pure_ex_fig(colors)