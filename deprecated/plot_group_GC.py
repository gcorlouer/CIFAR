#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 20:51:51 2021

@author: guime
"""


import cifar_load_subject as cf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import helper_functions as fun
import HFB_process as hf

from scipy.io import loadmat

#%%

sub_id = 'DiAs'
visual_chan_table = 'visual_channels_BP_montage.csv'
proc = 'preproc' 
sfreq = 250;
suffix = 'preprocessed_raw'
ext = '.fif'
categories = ['Rest', 'Face', 'Place']

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
fname = sub_id + 'pcgc.mat'
fpath = datadir.joinpath(fname)
visual_chan = subject.pick_visual_chan()
GC = loadmat(fpath)

#%%

F = GC['F']
sig = GC['sig']

#%% Sort indices

F_sorted = np.zeros_like(F)
sig_sorted = np.zeros_like(F)
ch_idx_sorted = visual_chan.index.tolist()
for isort, i in enumerate(ch_idx_sorted):
    for jsort, j in enumerate(ch_idx_sorted):
        F_sorted[isort,jsort,:] = F[i, j, :]
        sig_sorted[isort,jsort,:] = sig[i, j, :]

# %% Plot groupwise GC
plt.rcParams['font.size'] = '19'

ncat = F.shape[2]
TE = np.zeros_like(F)

for icat in range(ncat):
    TE[:,:,icat] = fun.GC_to_TE(F[:,:,icat])

TE_max = np.max(TE)
ROI_indices = hf.parcellation_to_indices(visual_chan, parcellation='DK')
ROI = list(ROI_indices.keys())
nROI = len(ROI_indices.keys())
TE_group = np.zeros((nROI, nROI, ncat))
for icat in range(ncat):
    for i in range(nROI):
        for j in range(nROI):
            iROI = ROI_indices[ROI[i]]
            jROI = ROI_indices[ROI[j]]
            T = np.take(TE, indices=iROI, axis=0)
            T = np.take(T, indices=jROI, axis=1)
            TE_group[i, j, icat] = np.mean(T[:, :, icat])

TE_group_max = np.max(TE_group)

ROI = [ROI[i][7:10] for i in range(nROI)]
for icat in range(ncat):
    populations = ROI
    plt.subplot(2,2, icat+1)
    sns.heatmap(TE_group[:,:,icat], vmin=0, vmax=TE_group_max, xticklabels=populations,
                    yticklabels=populations, cmap='YlOrBr')
    plt.title('TE ' + categories[icat] + ' (bits/s)')