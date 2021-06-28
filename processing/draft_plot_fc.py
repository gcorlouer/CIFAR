#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 10:38:47 2021

@author: guime
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import HFB_process as hf

from scipy.io import loadmat
from config import args
from pathlib import Path, PurePath
#%% Read ROI and functional connectivity data

ecog = hf.Ecog(args.cohort_path, subject=args.subject, proc=args.proc, 
                       stage = args.stage, epoch=args.epoch)
# Read visual channels 
df_visual = ecog.read_channels_info(fname=args.channels)
# Read roi
roi_idx = hf.read_roi(df_visual, roi=args.roi)
# List conditions
conditions = ['Rest', 'Face', 'Place']
# Load functional connectivity matrix
result_path = Path('~','projects', 'CIFAR','CIFAR_data', 'results').expanduser()
fname = args.subject + '_FC.mat'
functional_connectivity_path = result_path.joinpath(fname)

fc = loadmat(functional_connectivity_path)
#%% Plot functional connectivity

hf.plot_functional_connectivity(fc, df_visual, sfreq=args.sfreq, rotation=90, 
                                tau_x=0.5, tau_y=0.8, font_scale=1.6)
plt.show()

#%% Return lower diagonal of an array

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
n = a.shape[0]
low_indices= np.tril_indices(n)
print(a[low_indices])

#%% Return flatten array of MI and sig_MI along pairs
mi = fc['MI']
sig_mi = fc['sig_MI']
nchan = mi.shape[0]
n_cdt = mi.shape[2]
low_indices= np.tril_indices(nchan)
mi_flat = mi[low_indices]
sig_flat = sig_mi[low_indices]
# Return minimal value at which MI is significant
sig_idx = np.where(sig_flat == 1)
min_sig =np.amin(mi_flat[sig_idx])
print(mi_flat.shape)

#%% Plot MI along pairs of channels

for i in range(n_cdt):
    plt.plot(mi_flat[:,i], label=conditions[i])
    plt.axhline(y=min_sig)

plt.legend()
plt.show()

#%% Compare GC between R and F accross conditions
# For each pairs (R,F) returns indices of the pairs in the GC array then
# return a list of value of GC correspondig to that pair
# Not working yet.
gc = fc['F']
sig_gc = fc['sig_GC']
# gc_pop = {'RR':[], 'RF': [], 'FR':[], 'FF':[]}

pop = ['R','F']
for pop_target in pop:
    for pop_source in pop:
        gc_pop = {}
        key = pop_target + pop_source
        for i in roi_idx[pop_target]:
            for j in roi_idx[pop_source]:
               gc_pop[key] = gc[i,j,:]
