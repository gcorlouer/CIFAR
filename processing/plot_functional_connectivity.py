#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:31:12 2021
This script plot functional connectivity i.e. Mutual information and pairwise
condifional Granger causality.
@author: guime
"""


import cifar_load_subject as cf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import helper_functions as fun
import HFB_process as hf

from scipy.io import loadmat
from config import args
from pathlib import Path, PurePath


#%% Read ROI and functional connectivity data

ecog = hf.Ecog(args.cohort_path, subject=args.subject, proc=args.proc, 
                       stage = args.stage, epoch=args.epoch)
# Read visual channels 
df_visual = ecog.read_channels_info(fname=args.channels)
# Read electrodes infos
df_electrodes = ecog.read_channels_info(fname='electrodes_info.csv')
# Read functional and anatomical indices
functional_indices = hf.parcellation_to_indices(df_visual, 'group', matlab=False)
ROI_indices = hf.parcellation_to_indices(df_visual, 'DK', matlab=False)
# Restrict anatomical areas to lateral occipital and fusiform
ROI_indices = {'LO': ROI_indices['ctx-lh-lateraloccipital'], 
               'Fus': ROI_indices['ctx-lh-fusiform'] }
# List visual chans
visual_chan = df_visual['chan_name'].to_list()
# List conditions
conditions = ['Rest', 'Face', 'Place']

# Load functional connectivity matrix
cohort_path = args.cohort_path
fname = args.subject + 'FC.mat'
functional_connectivity_path = cohort_path.joinpath(args.subject, 'EEGLAB_datasets',
                                                    args.proc, fname)
fc = loadmat(functional_connectivity_path)
# Granger causality matrix
f = fc['F']
sig_gc = fc['sig_GC']
# Mutual information matrix
mi = fc['MI']
sig_mi = fc['sig_MI']

#%%

sns.set(font_scale=1.6)
n_cdt = len(conditions)
te = np.zeros_like(f)
# Convert to transfer entropy
for icat in range(n_cdt):
    te[:,:,icat] = fun.GC_to_TE(f[:,:,icat])
# Compute maximum value for scaling
te_max = np.max(te)
mi_max = np.max(mi)
mi_max = 0.03
fig, ax = plt.subplots(3,2)

for icat in range(n_cdt):
    populations = df_visual['group']
    g = sns.heatmap(mi[:,:,icat], vmin=0, vmax=mi_max, xticklabels=populations,
                    yticklabels=populations, cmap='YlOrBr', ax=ax[icat,0])
    g.set_yticklabels(g.get_yticklabels(), rotation = 90)
    ax[icat, 0].xaxis.tick_top()
    ax[0,0].set_title('Mutual information (bit)')
    ax[icat, 0].set_ylabel(conditions[icat])
    g = sns.heatmap(te[:,:,icat], vmin=0, vmax=te_max, xticklabels=populations,
                    yticklabels=populations, cmap='YlOrBr', ax=ax[icat,1])
    g.set_yticklabels(g.get_yticklabels(), rotation = 90)
    ax[icat, 1].xaxis.tick_top()
    ax[icat, 1].set_ylabel('Target')
    ax[0,1].set_title('Transfer entropy (bit/s)')
    for y in range(f.shape[0]):
        for x in range(f.shape[1]):
            if sig_mi[y,x,icat] == 1:
                ax[icat,0].text(x + 0.5, y + 0.8, '*',
                         horizontalalignment='center', verticalalignment='center',
                         color='k')
            else:
                continue
            if sig_gc[y,x,icat] == 1:
                ax[icat,1].text(x + 0.5, y + 0.8, '*',
                         horizontalalignment='center', verticalalignment='center',
                         color='k')
            else:
                continue

