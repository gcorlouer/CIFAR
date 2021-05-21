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
# sub_id = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
sub_id = 'DiAs'
visual_chan_table = 'visual_channels_BP_montage.csv'
proc = 'preproc' 
sfreq = 250;
# picks = ['LGRD58-LGRD59', 'LGRD60-LGRD61', 'LTo1-LTo2', 'LTo3-LTo4']
tmin_crop = -0.5
tmax_crop = 1.75
suffix = 'preprocessed_raw'
ext = '.fif'

#%%

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_chan = subject.pick_visual_chan()
# visual_chan = hf.pick_visual_chan(picks, visual_chan)
HFB = hf.visually_responsive_HFB(sub_id = sub_id)

categories = ['Rest', 'Face', 'Place']

_, visual_data = hf.HFB_to_visual_data(HFB, visual_chan, sfreq=sfreq, cat='Face', 
                                    tmin_crop = tmin_crop, tmax_crop=tmax_crop)
visual_time_series = visual_data

for cat in categories:
    X, visual_data = hf.HFB_to_visual_data(HFB, visual_chan, sfreq=sfreq, cat=cat, 
                                    tmin_crop = tmin_crop, tmax_crop=tmax_crop)
    visual_time_series[cat] = X


#%% Plot event related potential

time = visual_time_series['time']
channel_to_population = visual_time_series['channel_to_population']
population_to_channel = visual_time_series['populations']

fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

for i, cat in enumerate(categories):
    X = visual_time_series[cat]
    for key in population_to_channel:
        channels = population_to_channel[key]
        channels = [j-1 for j in channels]
        population_ERP = np.average(X[channels, :, :], axis=(0, 2))
        ax[i].plot(time, population_ERP, label=key)
        ax[i].set_ylabel(f'{cat}'+' HFB (dB)')
        ax[i].legend()
        ax[i].axvline(x=0, color='k')
ax[2].set_xlabel('Time from stimulus onset (s)')

