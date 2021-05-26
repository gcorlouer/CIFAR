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
# Read electrodes infos
df_electrodes = ecog.read_channels_info(fname='electrodes_info.csv')
# Read functional and anatomical indices
functional_indices = hf.parcellation_to_indices(df_visual, 'group', matlab=False)
roi_idx = hf.parcellation_to_indices(df_visual, 'DK', matlab=False)
# Restrict anatomical areas to lateral occipital and fusiform
roi_idx = {'LO': roi_idx['ctx-lh-lateraloccipital'], 
               'Fus': roi_idx['ctx-lh-fusiform'] }
# List visual chans
visual_chan = df_visual['chan_name'].to_list()
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

#%%

roi = list(roi_idx.keys())
n_roi = len(roi)
for i in range(n_roi):
    for j in range(n_roi):
        source_idx = roi_idx[roi[j]]
        target_idx = roi_idx[roi[i]]
        f_roi = np.take(f, indices=target_idx, axis=0)
        f_roi =  np.take(f_roi, indices = source_idx, axis=1)
        f_roi[i, j,:,:] = np.average(f_roi, axis=(0,1))

#%% Plot
sns.set(font_scale=1.5)
freq_step = sfreq/(2*(nfreq+1))
freqs = np.arange(0, sfreq/2, freq_step)
figure, ax =plt.subplots(n_roi, n_roi, sharex=True, sharey=True)
for c in range(n_cdt):
    for i in range(n_roi):
        for j in range(n_roi):
            ax[i,j].plot(freqs, f_roi[i,j,:,c], label = f'{conditions[c]}')
            ax[i,j].set_ylim(top=0.01)
            ax[i,j].text(x=40, y=0.005, s=f'{roi[j]} -> {roi[i]}')

ax[1,0].set_ylabel('Spectral GC')
ax[2,1].set_xlabel('Frequency (Hz)')


handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
