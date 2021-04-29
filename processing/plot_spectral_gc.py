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

#%% Load data

sub_id = 'DiAs'
visual_chan_table = 'visual_channels_BP_montage.csv'
proc = 'preproc' 
categories = ['Rest', 'Face', 'Place']

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
fname = sub_id + 'spectral_GC.mat'
fpath = datadir.joinpath(fname)
visual_chan = subject.pick_visual_chan()
sgc = loadmat(fpath)
fbin = 1024
sfreq = 100
f = sgc['f']
(nchan, nchan, nfreq, ncat) = f.shape
#%% Average over specific groups of channels

group_indices = hf.parcellation_to_indices(visual_chan, parcellation='group')
npop = len(group_indices)
f_group = np.zeros((npop, npop, nfreq, ncat))
visual_populations = list(group_indices.keys())
for i in range(npop):
    for j in range(npop):
        pop_i = visual_populations[i]
        ipop = group_indices[pop_i]
        pop_j = visual_populations[j]
        jpop = group_indices[pop_j]
        f_sub = np.take(f, indices=ipop, axis=0)
        f_sub =  np.take(f_sub, indices = jpop, axis=1)
        f_group[i, j,:,:] = np.average(f_sub, axis=(0,1))
        
#%% Plot
sns.set(font_scale=1.5)
freq_step = sfreq/(2*(fbin+1))
freqs = np.arange(0, sfreq/2, freq_step)
figure, ax =plt.subplots(npop,npop, sharex=True, sharey=True)
for c in range(ncat):
    for i in range(npop):
        for j in range(npop):
            ax[i,j].plot(freqs, f_group[i,j,:,c], label = f'{categories[c]}')
            ax[i,j].set_ylim(top=0.01)
            ax[i,j].text(x=40, y=0.005, s=f'{visual_populations[j]} -> {visual_populations[i]}')

ax[1,0].set_ylabel('Spectral GC')
ax[2,1].set_xlabel('Frequency (Hz)')


handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
