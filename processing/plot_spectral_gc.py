#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 12:00:19 2021
This script plot spectral GC on LFP in all conditions averaged over population
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
cohort_path = args.cohort_path
fname = args.subject + 'spectral_GC.mat'
spectral_gc_path = cohort_path.joinpath(args.subject, 'EEGLAB_datasets',
                                                    args.proc, fname)
sgc = loadmat(spectral_gc_path)
nfreq = args.nfreq
sfreq = args.sfreq
f = sgc['f']
(nchan, nchan, nfreq, n_cdt) = f.shape

#%% Average spgc over ROI

f_roi = hf.spcgc_to_smvgc(f, roi_idx)
(n_roi, n_roi, nfreq, n_cdt) = f_roi.shape

#%% Plot mvgc

hf.plot_smvgc(f_roi, roi_idx, sfreq=args.sfreq, x=40, y=0.01, font_scale=1.5)

#%%
