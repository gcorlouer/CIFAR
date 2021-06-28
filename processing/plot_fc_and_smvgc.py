#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 19:18:49 2021

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

# Load spectral granger causality

fname = args.subject + '_spectral_GC.mat'
result_path = Path('~','projects', 'CIFAR','CIFAR_data', 'results').expanduser()
spectral_gc_path = result_path.joinpath(fname)

sgc = loadmat(spectral_gc_path)
nfreq = args.nfreq
sfreq = args.sfreq
f = sgc['f']
(nchan, nchan, nfreq, n_cdt) = f.shape

#%% Plot functional connectivity
fig_name =args.subject + "_functional_connectivity.eps"
fig_path = result_path.joinpath(fig_name)
hf.plot_functional_connectivity(fc, df_visual, sfreq=args.sfreq, rotation=90, 
                                tau_x=0.5, tau_y=0.8, font_scale=1.6)
plt.savefig(fig_path)
plt.show(block=False)
plt.pause(3)
plt.close()
#%% Average spgc over ROI

f_roi = hf.spcgc_to_smvgc(f, roi_idx)
(n_roi, n_roi, nfreq, n_cdt) = f_roi.shape

#%% Plot spectral mvgc
fig_name =args.subject + "_smvgc.eps"
fig_path = result_path.joinpath(fig_name)
hf.plot_smvgc(f_roi, roi_idx, sfreq=args.sfreq, x=40, y=0.01, font_scale=1.5)
plt.savefig(fig_path)
plt.show(block=False)
plt.pause(3)
plt.close()