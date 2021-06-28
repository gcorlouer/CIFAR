#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:31:12 2021
This script plot functional connectivity i.e. Mutual information and pairwise
conditional Granger causality.
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
#plt.pause(3)
#plt.close()
#%%
# Simulated data
# fname = 'simulated_FC.mat'
# fpath  = Path('~', 'projects', 'CIFAR','data_fun').expanduser()
# functional_connectivity_path = fpath.joinpath(fname)