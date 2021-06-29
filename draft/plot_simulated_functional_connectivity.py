#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:13:40 2021

@author: guime
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import HFB_process as hf

from scipy.io import loadmat
from config import args
from pathlib import Path, PurePath

#%% Read data

ecog = hf.Ecog(args.cohort_path, subject=args.subject, proc=args.proc, 
                       stage = args.stage, epoch=args.epoch)

# Load functional connectivity matrix
cohort_path = args.cohort_path
fname = args.subject + 'FC.mat'
functional_connectivity_path = cohort_path.joinpath(args.subject, 'EEGLAB_datasets',
                                                    args.proc, fname)

# Read visual channels 
df_visual = ecog.read_channels_info(fname=args.channels)
# Read roi
roi_idx = hf.read_roi(df_visual, roi=args.roi)
# List conditions
conditions = ['Rest', 'Face', 'Place']
# Simulated data
fname = 'simulated_FC.mat'
fpath  = Path('~', 'projects', 'CIFAR','data_fun').expanduser()
functional_connectivity_path = fpath.joinpath(fname)


fc = loadmat(functional_connectivity_path)

#%% Plot functional connectivity

hf.plot_functional_connectivity(fc, df_visual, sfreq=args.sfreq, te_max=2, 
                                 mi_max=0.08,rotation=90, tau_x=0.5, tau_y=0.8, 
                                 font_scale=1.6)

#%% Conpare connectivity matrix with estimated GC

connectivity_matrix = fc['connectivity_matrix']
sig_GC = fc['sig_GC']
conditions = ['Rest', 'Face', 'Place']
rotation = 90
n_cdt = len(conditions)
fig, ax = plt.subplots(3,2)
for icat in range(n_cdt):
    populations = df_visual['group']
    g = sns.heatmap(sig_GC[:,:,icat], vmin=0, vmax=1, xticklabels=populations,
                    yticklabels=populations, cmap='YlOrBr', ax=ax[icat,0])
    g.set_yticklabels(g.get_yticklabels(), rotation = rotation)
    # Position xticks on top of heatmap
    ax[icat, 0].xaxis.tick_top()
    ax[0,0].set_title('Estimated GC')
    ax[icat, 0].set_ylabel(conditions[icat])
    g = sns.heatmap(connectivity_matrix[:,:,icat], vmin=0, vmax=1, xticklabels=populations,
                    yticklabels=populations, cmap='YlOrBr', ax=ax[icat,1])
    g.set_yticklabels(g.get_yticklabels(), rotation = rotation)
    ax[icat, 1].xaxis.tick_top()
    ax[icat, 1].set_ylabel('Target')
    ax[0,1].set_title('True connectivity')