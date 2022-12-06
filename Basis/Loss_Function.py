# -*- coding: utf-8 -*-

import torch

# -----loss-----
# von Neumann entropy
def S(rho):
    S_rho_tmp = torch.matmul(rho, torch.log(rho))
    S_rho = -torch.trace(S_rho_tmp)
    return S_rho


def MLE_loss(out, target):
    #out = out / torch.sum(out)
    #target = target / torch.sum(target)
    #target = target * torch.sum(out)
    #loss = torch.sum((out - target)**2 * (torch.sqrt(target)))
    out_idx = out > 1e-12
    loss = -target[out_idx].dot(torch.log(out[out_idx]))
    return loss


def CF_loss(out, target):  # classical fidelity
    #out = out / torch.sum(out)
    #target = target / torch.sum(target)
    #target = target * torch.sum(out)
    p = 1 / 2
    loss = 1 - (target**p).dot(out**p)
    return loss


def KL_loss(out, target):
    #target = target * torch.sum(out)
    loss = target.dot(torch.log(target / out))
    return loss


def MLE_CF_loss(out, target, a=0.01, b=0.01):
    loss = a * MLE_loss(out, target) + b * CF_loss(out, target)
    return loss


def MLE_CF_loss_2(out, target):
    loss = MLE_loss(out, target) / - CF_loss(out, target)
    return loss


def KL_CF_loss(out, target, a=1, b=0.01):
    loss = a * KL_loss(out, target) + b * CF_loss(out, target)
    return loss


def MLE_CF_KL_loss(out, target, a=0.01, b=0.01):
    loss = MLE_loss(out, target) + a * CF_loss(out, target) + \
        b * KL_loss(out, target)
    return loss


def MLME_loss(out, target, rho, a=0.001):
    loss = -a * S(rho) + MLE_CF_loss(out, target, 1, 0)
    return loss
