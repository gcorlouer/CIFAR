#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:02:57 2021

@author: guime
"""


import cifar_load_subject as cf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import helper_functions as fun
import HFB_process as hf

from scipy.io import loadmat

#%%

sub_id = 'DiAs'
visual_chan_table = 'visual_channels_BP_montage.csv'
proc = 'preproc' 
sfreq = 250;
suffix = 'preprocessed_raw'
ext = '.fif'
categories = ['Rest', 'Face', 'Place']
subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
# %% Plot spectral GC

fres = 1024;
fmax = sfreq/2;

fname = sub_id + 'spectral_gc.mat'
fpath = datadir.joinpath(fname)
sGC = loadmat(fpath)
f = sGC['f']

freqs = np.arange(start=0, stop= fmax, step = fmax/fres)
(nchan, nchan, nfreq, ncat) = f.shape

f_tot = np.zeros((nfreq, ncat))
f_tot = np.mean(f, axis=(0,1))

for icat in range(ncat):
    plt.plot(freqs, f_tot[:,icat], label=categories[icat])

plt.xlabel('Frequency (HZ)')
plt.ylabel('Global GC')
plt.legend()