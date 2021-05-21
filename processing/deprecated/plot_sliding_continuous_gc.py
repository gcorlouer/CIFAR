#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 11:47:01 2021

@author: guime
"""

import cifar_load_subject as cf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import helper_functions as fun
import HFB_process as hf
import pandas as pd

from scipy.io import loadmat

#%%

sub_id = 'DiAs'
proc = 'preproc' 
categories = ['Rest', 'Face', 'Place']
pop = ['R', 'F']

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
fname = sub_id + '_pgc_sliding_continuous.mat' # or _pgc_sliding.mat
fpath = datadir.joinpath(fname)
GC = loadmat(fpath)
#%%

F = GC['F']
sig = GC['sig_GC']
# time = GC['time']
sfreq = GC['sfreq']
sfreq = sfreq[0][0]
(nchan, nchan, nwin, ncat) = F.shape
time = np.arange(0,nwin)
TE = np.zeros_like(F)
for icat in range(ncat):
    TE[:,:,:, icat] = fun.GC_to_TE(F[:,:, :, icat],  sfreq=sfreq)
#%%
sns.set(font_scale=2)
f, ax = plt.subplots(ncat,1)
for i in range(ncat):
    for ichan in range(nchan):
        for jchan in range(nchan):
            if ichan==jchan:
                continue
            else:
                ax[i].plot(time, TE[ichan,jchan,:,i], label=f'{pop[jchan]} to {pop[ichan]}')
                for j in range(nwin):
                    if sig[ichan,jchan,j,i]==1:
                        ax[i].plot(time[j], TE[ichan,jchan,j,i], '*', color='k')
                    else:
                        continue
                ax[i].set_ylim([0, 5])
                ax[i].legend()
                ax[i].set_ylabel(f'TE {categories[i]}')
plt.xlabel('Window')