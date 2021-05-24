#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 13:57:30 2021
This script plot grand average event related HFB
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

#%%

cross_population_hfb, populations = hf.ts_to_population_hfb(cross_ts, 
                                                            df_all_visual_chans,
                                                            parcellation='group')

#%% Plot grand average event related HFB

evok_stat = fun.compute_evok_stat(cross_population_hfb, axis=2)
max_evok = np.max(evok_stat)
#%%

step = 0.1
alpha = 0.5
sns.set(font_scale=1.6)
color = ['k', 'b', 'g']
cat = ['Rest', 'Face', 'Place']
ncat = len(cat)
npop = len(populations)
xticks = np.arange(args.tmin_crop, args.tmax_crop, step)

f, ax = plt.subplots(3,1)
for i in range(ncat):
    for j in range(npop):
        ax[i].plot(time, evok_stat[0][j,:,i], label = populations[j])
        ax[i].fill_between(time, evok_stat[1][j,:,i], evok_stat[2][j,:,i], alpha=alpha)
        ax[i].set_ylim(bottom=-1, top=max_evok+1)
        ax[i].xaxis.set_ticks(xticks)
        ax[i].axvline(x=0, color='k')
        ax[i].axhline(y=0, color='k')
        ax[i].set_ylabel(f'HFB {cat[i]} (dB)')
        ax[i].legend()

plt.xlabel('Time (s)')