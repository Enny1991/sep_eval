import numpy as np
from itertools import permutations
from torch.autograd import Variable

import torch

def calc_sdr(estimation, origin):
    """
    batch-wise SDR caculation for one audio file.
    estimation: (batch, nsample)
    origin: (batch, nsample)
    """
    
    origin_power = np.sum(origin**2, 1, keepdims=True)  # (batch, 1)
    
    scale = np.sum(origin*estimation, 1, keepdims=True) / origin_power  # (batch, 1)
    
    est_true = scale * origin  # (batch, nsample)
    est_res = estimation - est_true  # (batch, nsample)
    
    true_power = np.sum(est_true**2, 1)
    res_power = np.sum(est_res**2, 1)
    
    return 10*np.log10(true_power) - 10*np.log10(res_power)  # (batch, 1)

def calc_sdr_torch(estimation, origin):
    """
    batch-wise SDR caculation for one audio file on pytorch Variables.
    estimation: (batch, nsample)
    origin: (batch, nsample)
    """
    
    origin_power = torch.pow(origin, 2).sum(1, keepdim=True)  # (batch, 1)
    
    scale = torch.sum(origin*estimation, 1, keepdim=True) / origin_power  # (batch, 1)
    
    est_true = scale * origin  # (batch, nsample)
    est_res = estimation - est_true  # (batch, nsample)
    
    true_power = torch.pow(est_true, 2).sum(1)
    res_power = torch.pow(est_res, 2).sum(1)
    
    return torch.log(true_power) - torch.log(res_power)  # (batch, 1)


def batch_SDR(estimation, origin):
    """
    batch-wise SDR caculation for multiple audio files.
    estimation: (batch, nsource, nsample)
    origin: (batch, nsource, nsample)
    """
    
    batch_size_est, nsource_est, nsample_est = estimation.shape
    batch_size_ori, nsource_ori, nsample_ori = origin.shape
    
    assert batch_size_est == batch_size_ori, "Estimation and original sources should have same shape."
    assert nsource_est == nsource_ori, "Estimation and original sources should have same shape."
    assert nsample_est == nsample_ori, "Estimation and original sources should have same shape."
    
    assert nsource_est < nsample_est, "Axis 1 should be the number of sources, and axis 2 should be the signal."
    
    batch_size = batch_size_est
    nsource = nsource_est
    nsample = nsample_est
    
    # zero mean signals
    estimation = estimation - np.mean(estimation, 2, keepdims=True)
    origin = origin - np.mean(origin, 2, keepdims=True)
    
    # possible permutations
    perm = list(set(permutations(np.arange(nsource))))
    
    # pair-wise SDR
    SDR = np.zeros((batch_size, nsource, nsource))
    for i in range(nsource):
        for j in range(nsource):
            SDR[:,i,j] = calc_sdr(estimation[:,i], origin[:,j])
    
    # choose the best permutation
    SDR_max = []
    for i in range(batch_size):
        SDR_perm = []
        for permute in perm:
            sdr = 0.
            for idx in range(len(permute)):
                sdr += SDR[i][idx][permute[idx]]
            SDR_perm.append(sdr)
        SDR_max.append(np.max(SDR_perm) / nsource)
    
    return np.asarray(SDR_max)


def batch_SDR_torch(estimation, origin, USE_CUDA):
    """
    batch-wise SDR caculation for multiple audio files.
    estimation: (batch, nsource, nsample)
    origin: (batch, nsource, nsample)
    """
    
    batch_size_est, nsource_est, nsample_est = estimation.size()
    batch_size_ori, nsource_ori, nsample_ori = origin.size()
    
    assert batch_size_est == batch_size_ori, "Estimation and original sources should have same shape."
    assert nsource_est == nsource_ori, "Estimation and original sources should have same shape."
    assert nsample_est == nsample_ori, "Estimation and original sources should have same shape."
    
    assert nsource_est < nsample_est, "Axis 1 should be the number of sources, and axis 2 should be the signal."
    
    batch_size = batch_size_est
    nsource = nsource_est
    nsample = nsample_est
    
    # zero mean signals
    estimation = estimation - torch.mean(estimation, 2, keepdim=True).expand_as(estimation)
    origin = origin - torch.mean(origin, 2, keepdim=True).expand_as(estimation)
    
    # possible permutations
    perm = list(set(permutations(np.arange(nsource))))
    
    # pair-wise SDR
    SDR = Variable(torch.zeros((batch_size, nsource, nsource)))
    if USE_CUDA:
        SDR = SDR.cuda()
    for i in range(nsource):
        for j in range(nsource):
            SDR[:,i,j] = calc_sdr_torch(estimation[:,i], origin[:,j])
    
    # choose the best permutation
    SDR_max = []
    for i in range(batch_size):
        SDR_perm = []
        for permute in perm:
            sdr = []
            for idx in range(len(permute)):
                sdr.append(SDR[i][idx][permute[idx]].view(1,-1))
            sdr = torch.sum(torch.cat(sdr, 0))
            SDR_perm.append(sdr.view(1,-1))
        SDR_perm = torch.cat(SDR_perm, 0)
        sdr_max = torch.max(SDR_perm) / nsource
        SDR_max.append(sdr_max.view(1,-1))
    
    return torch.cat(SDR_max, 0)