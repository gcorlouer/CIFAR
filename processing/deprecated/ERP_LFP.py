#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:59:12 2021

@author: guime
"""

import cifar_load_subject as cf
import HFB_process as hf
import numpy as np
import helper_functions as fun
import scipy.stats as stats
import matplotlib.pyplot as plt

from scipy.io import loadmat,savemat

#%%
sub_id = 'DiAs'
sfreq = 250
proc = 'preproc'
stage = '_BP_montage_preprocessed_raw.fif'
categories = ['Rest', 'Face', 'Place']
tmin = -0.5
tmax = 1.75

#%% 
subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_chan = subject.pick_visual_chan()

LFP = subject.load_raw_data(proc=proc, stage=stage)

visual_LFP_dict = fun.LFP_to_dict(LFP, visual_chan, tmin=tmin, tmax=tmax, sfreq=sfreq)

#%% Plot event related potential for each channels in each condition

time = visual_LFP_dict['time']
group = visual_LFP_dict['group']
population_to_channel = visual_LFP_dict['population_to_channel']
nchan = len(group)

fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

for i, cat in enumerate(categories):
    X = visual_LFP_dict[cat]
    X = 1e6 * X
    for j in range(nchan):
        ERP = np.average(X[j, :, :], axis=1)
        ax[i].plot(time, ERP, label= group[j])
        ax[i].set_ylabel(f'{cat}'+r' ERP ($\mu$V)')
        ax[i].legend()
ax[2].set_xlabel('Time (s)')

#%% Plot event related potential for each populations in each condition

time = visual_LFP_dict['time']
group = visual_LFP_dict['group']
population_to_channel = visual_LFP_dict['population_to_channel']

fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

for i, cat in enumerate(categories):
    X = visual_LFP_dict[cat]
    X = 1e6 * X
    for key in population_to_channel:
        channels = population_to_channel[key]
        population_ERP = np.average(X[channels, :, :], axis=(0, 2))
        ax[i].plot(time, population_ERP, label=key)
        ax[i].set_ylabel(f'{cat}'+r' ERP ($\mu$V)')
        ax[i].legend()
ax[2].set_xlabel('Time (s)')