#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:36:50 2021

@author: guime
"""



import HFB_process as hf
import cifar_load_subject as cf
import scipy as sp
import re 
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import helper_functions as fun

from pathlib import Path, PurePath
from mne.viz import plot_filter, plot_ideal_filter
from scipy import signal, fftpack
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection, multipletests

#%%

sub_id= 'DiAs'
proc= 'preproc' 
stage= '_BP_montage_HFB_raw.fif'
sfreq = 250
tmin_crop = 0
tmax_crop = 1.75

#%%
subject = cf.Subject(sub_id)
visual_populations = subject.pick_visual_chan()
hfb, visual_chan = hf.load_visual_hfb(sub_id= sub_id, proc= proc, 
                            stage= stage)

ts, time = hf.ts_all_categories(hfb, visual_chan, sfreq=sfreq, tmin_crop=tmin_crop, tmax_crop=tmax_crop)

#%% Plot category HFB in each condition

def ts_to_population_hfb(ts, visual_populations, parcellation='group'):
    """
    Return hfb of each population from category specific time series.
    """
    (nchan, nobs, ntrials, ncat) = ts.shape
    populations_indices = hf.parcellation_to_indices(visual_populations,
                                                     parcellation=parcellation)
    populations = populations_indices.keys()
    npop = len(populations)
    population_hfb = np.zeros((npop, nobs, ntrials, ncat))
    for ipop, pop in enumerate(populations):
        pop_indices = populations_indices[pop]
        # population hfb is average of hfb over each population-specific channel
        population_hfb[ipop,...] = np.average(ts[pop_indices,...], axis=0)
    # Return list of populations to keep track of population ordering
    populations = list(populations)
    return population_hfb, populations

#%%

population_hfb, populations = ts_to_population_hfb(ts, visual_populations, parcellation='group')
evok_stat = fun.compute_evok_stat(population_hfb, axis=2)
#%%

step = 0.1
alpha = 0.5
sns.set(font_scale=1.6)
evok = [0]*3
color = ['k', 'b', 'g']
cat = ['Rest', 'Face', 'Place']
xticks = np.arange(tmin_crop, tmax_crop, step)

f, ax = plt.subplots(3,1)
for i in range(ncat):
    for j in range(npop):
        ax[i].plot(time, evok_stat[0][j,:,i], label = populations[j])
        ax[i].fill_between(time, evok_stat[1][j,:,i], evok_stat[2][j,:,i], alpha=alpha)
        ax[i].set_ylim(bottom=-1, top=3)
        ax[i].xaxis.set_ticks(xticks)
        ax[i].axvline(x=0, color='k')
        ax[i].axhline(y=0, color='k')
        ax[i].set_ylabel(f'HFB {cat[i]} (dB)')
        ax[i].legend()

plt.xlabel('Time (s)')