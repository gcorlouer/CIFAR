#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:38:00 2021

@author: guime
"""
import cifar_load_subject as cf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import helper_functions as fun
import HFB_process as hf
import pandas as pd

from scipy.io import loadmat

#%%

sub_id = 'DiAs'
proc = 'preproc' 
categories = ['Rest', 'Face', 'Place']

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
fname = sub_id + 'pcgc.mat'
fpath = datadir.joinpath(fname)
GC = loadmat(fpath)

brainpath  = subject.brain_path()
fname = 'BP_channels.csv'
fpath = brainpath.joinpath(fname)
df_BP = pd.read_csv(fpath)
df_BP = df_BP.sort_values(by='Y')
BP_chan = df_BP['electrode_name']
#%%

F = GC['F']
sig = GC['sig']

#%% Sort indices

F_sorted = np.zeros_like(F)
sig_sorted = np.zeros_like(F)
ch_idx_sorted = BP_chan.index.tolist()
for isort, i in enumerate(ch_idx_sorted):
    for jsort, j in enumerate(ch_idx_sorted):
        F_sorted[isort,jsort] = F[i, j]
        sig_sorted[isort,jsort] = sig[i, j]

#%%

plt.rcParams['font.size'] = '19'

TE = np.zeros_like(F)

TE = fun.GC_to_TE(F)

TE_max = np.max(TE)

plt.subplot(1,2,1)
sns.heatmap(TE, vmin=0, vmax=TE_max,cmap='YlOrBr')
plt.subplot(1,2,2)
sns.heatmap(sig, vmin=0, vmax=1,cmap='YlOrBr')


