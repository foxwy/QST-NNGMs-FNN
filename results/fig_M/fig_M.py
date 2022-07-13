# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from Basis_State import State

'''
_, rho_mat = State().Get_state_rho('GHZi_P', 5)  # expected
npzfile = np.load('GHZi_5.npy')  # reconstructed'''

rho_mat = np.load('random_P9.npy')  # reconstructed
print(rho_mat)
npzfile = np.load('Re_random_P9.npy')  # reconstructed
print('---', npzfile)


fig, ax = plt.subplots(1, 2, figsize=(8, 3))
v_img = ax[0].imshow(rho_mat.real, cmap='OrRd')
ax[0].set_xticks([0, 5, 10, 15, 20, 25, 30], fontproperties='Arial', size=15)
ax[0].set_yticks([0, 5, 10, 15, 20, 25, 30], fontproperties='Arial', size=15)

vi_img = ax[1].imshow(rho_mat.imag, cmap='Blues')
ax[1].set_xticks([0, 5, 10, 15, 20, 25, 30], fontproperties='Arial', size=15)
ax[1].set_yticks([0, 5, 10, 15, 20, 25, 30], fontproperties='Arial', size=15)

'''
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
v_img = ax[0].imshow(npzfile.real, cmap='OrRd')
ax[0].set_xticks([0, 5, 10, 15, 20, 25, 30], fontproperties='Arial', size=15)
ax[0].set_yticks([0, 5, 10, 15, 20, 25, 30], fontproperties='Arial', size=15)

vi_img = ax[1].imshow(npzfile.imag, cmap='Blues')
ax[1].set_xticks([0, 5, 10, 15, 20, 25, 30], fontproperties='Arial', size=15)
ax[1].set_yticks([0, 5, 10, 15, 20, 25, 30], fontproperties='Arial', size=15)'''

cb0 = plt.colorbar(v_img, ax=ax[0])
cb0.ax.tick_params(labelsize=10)
cb0.set_ticks([-0.07, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08])
cb1 = plt.colorbar(vi_img, ax=ax[1])
cb1.ax.tick_params(labelsize=10)
cb1.set_ticks([-0.07, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.07])
plt.savefig('random_P9.pdf', dpi=600, bbox_inches='tight', pad_inches=0.0)
plt.show()