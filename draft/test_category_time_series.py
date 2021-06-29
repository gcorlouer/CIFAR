#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:36:50 2021

@author: guime
"""



import mne
import pandas as pd
import HFB_process as hf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import helper_functions as fun

from pathlib import Path, PurePath
from config import args
from scipy.io import savemat


#%%

ecog = hf.Ecog(args.cohort_path, subject=args.subject, proc=args.proc, 
                       stage = args.stage, epoch=args.epoch)
hfb = ecog.read_dataset()
df_visual = ecog. read_channels_info(fname=args.channels)
visual_chan = df_visual['chan_name'].to_list()
ts, time = hf.category_ts(hfb, visual_chan, sfreq=args.sfreq, tmin_crop=args.tmin_crop, 
                          tmax_crop=args.tmax_crop)
#%%

population_hfb, populations = hf.ts_to_population_hfb(ts, df_visual, parcellation='group')

#%%
evok_stat = fun.compute_evok_stat(population_hfb, axis=2)
max_evok = np.max(evok_stat)
#%%

step = 0.1
alpha = 0.5
sns.set(font_scale=1.6)
evok = [0]*3
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