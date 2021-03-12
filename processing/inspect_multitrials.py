# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:18:43 2021

@author: guime
"""


import HFB_process as hf
import cifar_load_subject as cf
import pandas as pd
import numpy as np 
import helper_functions as fun 
import matplotlib.pyplot as plt

from scipy.io import savemat

# %matplotlib
#%% TODO

# -Check that there are output visual_data X is correct with HFB_visual (i.e. check that 
# permutation works)
# - Create a module for category specific electrodes
# - Rearrange HFB module consequently
# sub_id = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']

#%% 
pd.options.display.max_rows = 999

sub_id = 'DiAs'
visual_chan_table = 'visual_channels_BP_montage.csv'
proc = 'preproc' 
sfreq = 100;
# picks = ['LGRD58-LGRD59', 'LGRD60-LGRD61', 'LTo1-LTo2', 'LTo3-LTo4']
tmin_crop = 0.050
tmax_crop = 1.5
suffix = 'preprocessed_raw'
ext = '.fif'
categories = ['Rest', 'Face', 'Place']

#%%

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_chan = subject.pick_visual_chan()
# visual_chan = hf.pick_visual_chan(picks, visual_chan)
HFB = hf.visually_responsive_HFB(sub_id = sub_id)


ts, time = fun.ts_all_categories(HFB, tmin_crop=tmin_crop, tmax_crop=tmax_crop)

#%%

(n, m, N, c) = ts.shape
newshape = (n, m*N, c)
X = np.reshape(ts, newshape)
T = np.arange(0, X.shape[1])

#%% Compute signal to noise ratio

signal = np.average(ts, axis=1)
std = np.std(ts, axis=1)
SNR = np.divide(signal, std)
SNR = np.average(SNR,0)
std_mean = np.average(std, axis=0)
signal_mean = np.average(signal, axis=0)
#%%

trial = np.arange(0, N)
ncat = c
for icat in range(ncat):
    plt.subplot(3,1,icat+1)
    plt.plot(trial, signal_mean[:,icat])

#%% Plot trial to trial variability

ntrial = 56
icat = 1;
x = np.average(ts, axis=0)
x = x[:,:,icat]
for i in range(ntrial):
    plt.subplot(8,7, i + 1)
    plt.plot(time, x[:,i])
