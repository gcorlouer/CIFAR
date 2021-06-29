#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 14:34:30 2021

@author: guime
"""


import mne
import pandas as pd
import HFB_process as hf
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as fun
import seaborn as sns

from pathlib import Path, PurePath
from config import args
from scipy.io import savemat
from mne.time_frequency import psd_array_multitaper

#%%
cohort_path = args.cohort_path
fpath = cohort_path.joinpath('all_visual_channels.csv')
df_all_visual_chans = pd.read_csv(fpath)
#%% Cross subject ts

cross_ts, time = hf.cross_subject_ts(args.cohort_path, args.cohort, proc=args.proc, 
                     channels = args.channels,
                     stage=args.stage, epoch=args.epoch,
                     sfreq=args.sfreq, tmin_crop=args.tmin_crop, 
                     tmax_crop=args.tmax_crop)

#%% Cross population HFB

cross_population_hfb, populations = hf.ts_to_population_hfb(cross_ts, 
                                                            df_all_visual_chans,
                                                            parcellation='group')

(npop, m, N, ncat)= cross_population_hfb.shape
new_shape = (npop, ncat, N, m)
cross_population_hfb = np.transpose(cross_population_hfb, (0,3,2,1))
#%% Compute psd

fmin=0.1
fmax=100
adaptive=True
bandwidth=2

psd, freqs = psd_array_multitaper(cross_population_hfb, args.sfreq, fmin=fmin, fmax=fmax,
                                                 bandwidth=bandwidth, adaptive=adaptive)
average_psd = np.mean(psd, axis=2)
max_psd = np.max(psd)
#%% Plot psd in each conditions for each populations

sns.set(font_scale=1.6)
f, ax = plt.subplots(3,1)
cat = ['Rest', 'Face', 'Place']
bands = [4, 8, 16, 31]
bands_name = [r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', r'$\gamma$']
xbands = [2, 6, 12, 18, 50]
ybands = [12]*5
for i in range(ncat):
    for j in range(npop):
        ax[i].plot(freqs, average_psd[j,i,:], label=populations[j])
        ax[i].set_ylim(top=max_psd)
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        ax[i].set_ylabel(f'Psd {cat[i]} (dB)')
        for k in range(len(bands)):
            ax[i].axvline(x=bands[k], color='k', linestyle='--')
        for k in range(len(xbands)):
            ax[i].text(xbands[k]+1, ybands[k], bands_name[k])
        ax[i].legend()
        
plt.xlabel('Frequency (Hz)')