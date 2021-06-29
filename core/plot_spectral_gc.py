#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 12:00:19 2021
This script plot spectral GC on LFP in all conditions averaged over population
-------------------------------------------------------------------------------
Observation: Qualititatively, there does not seem to be differenece in
spectral mvgc between face, retinotopic and place populations, also the spectral
mvgc seem close to zero although we see some significant GC im time domain
functional connectivity.

Next actions: check that average f is correctly extracted.
Plot functional smvgc for HFB and LFP side by side to show to Lionel.
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

#%% Load data

ecog = hf.Ecog(args.cohort_path, subject=args.subject, proc=args.proc, 
                       stage = args.stage, epoch=args.epoch)
# Read visual channels 
df_visual = ecog.read_channels_info(fname=args.channels)
# Read roi
roi_idx = hf.read_roi(df_visual, roi=args.roi)
# List conditions
conditions = ['Rest', 'Face', 'Place']

# Load spectral granger causality

fname = args.subject + '_spectral_GC.mat'
result_path = Path('~','projects', 'CIFAR','CIFAR_data', 'results').expanduser()
spectral_gc_path = result_path.joinpath(fname)

sgc = loadmat(spectral_gc_path)
nfreq = args.nfreq
sfreq = args.sfreq
f = sgc['f']
(nchan, nchan, nfreq, n_cdt) = f.shape

#%% Average spgc over ROI

f_roi = hf.spcgc_to_smvgc(f, roi_idx)
(n_roi, n_roi, nfreq, n_cdt) = f_roi.shape

#%% Plot spectral mvgc

hf.plot_smvgc(f_roi, roi_idx, sfreq=args.sfreq, x=10, y=0.005, font_scale=2)
plt.show()
#plt.pause(3)
#plt.close()

#%%
# Simulated data
# fname = 'simulated_spectral_GC.mat'
# fpath  = Path('~', 'projects', 'CIFAR','data_fun').expanduser()
# spectral_gc_path = fpath.joinpath(fname)