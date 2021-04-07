#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:31:12 2021
This script plot functional connectivity i.e. Mutual information and pairwise
condifional Granger causality.
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
categories = ['Rest', 'Face', 'Place']

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
fname = sub_id + 'FC.mat'
fpath = datadir.joinpath(fname)
visual_chan = subject.pick_visual_chan()
fc = loadmat(fpath)

#%%

f = fc['F']
sig_gc = fc['sig_GC']
mi = fc['MI']
sig_mi = fc['sig_MI']

#%%
sns.set(font_scale=1.6)
ncat = f.shape[2]
te = np.zeros_like(f)

for icat in range(ncat):
    te[:,:,icat] = fun.GC_to_TE(f[:,:,icat])

te_max = np.max(te)
mi_max = np.max(mi)

fig, ax = plt.subplots(3,2)

for icat in range(ncat):
    populations = visual_chan['group']
    g = sns.heatmap(mi[:,:,icat], vmin=0, vmax=mi_max, xticklabels=populations,
                    yticklabels=populations, cmap='YlOrBr', ax=ax[icat,0])
    g.set_yticklabels(g.get_yticklabels(), rotation = 90)
    ax[icat, 0].xaxis.tick_top()
    ax[0,0].set_title('Mutual information (bit)')
    ax[icat, 0].set_ylabel(categories[icat])
    g = sns.heatmap(te[:,:,icat], vmin=0, vmax=te_max, xticklabels=populations,
                    yticklabels=populations, cmap='YlOrBr', ax=ax[icat,1])
    g.set_yticklabels(g.get_yticklabels(), rotation = 90)
    ax[icat, 1].xaxis.tick_top()
    ax[icat, 1].set_ylabel('Target')
    ax[0,1].set_title('Transfer entropy (bit/s)')
    for y in range(f.shape[0]):
        for x in range(f.shape[1]):
            if sig_mi[y,x,icat] == 1:
                ax[icat,0].text(x + 0.5, y + 0.8, '*',
                         horizontalalignment='center', verticalalignment='center',
                         color='k')
            else:
                continue
            if sig_gc[y,x,icat] == 1:
                ax[icat,1].text(x + 0.5, y + 0.8, '*',
                         horizontalalignment='center', verticalalignment='center',
                         color='k')
            else:
                continue