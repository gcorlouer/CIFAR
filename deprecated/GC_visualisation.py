#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 14:57:36 2020

@author: guime
"""


import HFB_test
import cf_load
import scipy as sp
import re 
import numpy as np 
import seaborn as sn

import mne
import matplotlib.pyplot as plt
import pandas as pd

from scipy import io
# TODO : Change order of chan names for xlabels, try other scale, compare.
plt.rcParams.update({'font.size': 30})

%matplotlib

# %% Import data
# param
proc = 'preproc'
sub = 'DiAs'
task = 'stimuli'
run = '1'
multitrial = True
cat = 'Face'

ext = '.mat'

picks = ['LTo6  ', 'LGRD60', 'LTo1  ','LGRD52', 'LGRD50', 'LTo4  ', 'LTp3  ']

# %% 
path_visual = cf_load.visual_path()
df_visual = pd.read_csv(path_visual)

df = df_visual.loc[df_visual['subject_id']==sub]
df = df.replace(to_replace='Bicat', value='Visual')

def GC_cat_mat(sub='DiAs', cat='Face', multitrial=True, proc = 'preproc', ext = '.mat'):
    
    if multitrial == True:
        suffix = 'GC_multi_HFB_visual_epoch_' + cat
    else:
        suffix = 'GC_sliding_HFB_visual_epoch_' + cat
    subject = cf_load.Subject(sub)
    GC_fpath = subject.fpath(proc = proc, suffix=suffix, ext=ext)
    GC_mat = io.loadmat(GC_fpath)
    return GC_mat

# Find indices of electrodes in GC dictionary

def extract_GC(picks, GC_mat):
    ch_names = list(GC_mat['ch_names'])
    ipicks = []
    for pick in picks:
        ipick = ch_names.index(pick)
        ipicks.append(ipick)
        
    GC_cat = GC_mat['F']
    GC_cat = np.nan_to_num(GC_cat)
    GC = np.zeros((len(ipicks),len(ipicks)))
    for i, ip in enumerate(ipicks):
        for j, jp in enumerate(ipicks): 
            GC[i,j] = GC_cat[i,j]
    return GC


def chan_to_GC(picks, sub='DiAs', cat='Face', multitrial=True, 
                  proc = 'preproc', ext = '.mat'):
    GC_mat = GC_cat_mat(sub=sub, cat=cat, multitrial=multitrial, 
                        proc = proc, ext = ext)
    
    GC = extract_GC(picks, GC_mat)
    return GC 

# GC = extract_GC(picks, GC_mat)

GC_f = chan_to_GC(picks, sub=sub, cat= 'Face', multitrial=multitrial)
GC_p = chan_to_GC(picks, sub=sub, cat= 'Place', multitrial=multitrial)

# %% Change category
picks = ['LTo6', 'LGRD60', 'LTo1','LGRD52', 'LGRD50', 'LTo4', 'LTp3']

picks_category = []
for pick in picks:
    print(pick)
    picks_category.extend(list(df['category'].loc[df['chan_name']==pick]))

print(picks_category)
# %% Plot 

fig, ax = plt.subplots(ncols=2, sharey=False, sharex=False)
F1 = sn.heatmap(GC_f, cmap="YlGnBu", vmin=0, vmax=0.02, annot=False, 
                     square=True, robust=False, cbar= True, xticklabels=picks_category, 
                     yticklabels=picks_category, ax=ax[0], label='Face')
F1 = sn.heatmap(GC_p, cmap="YlGnBu", vmin=0, vmax=0.02, annot=False, 
                     square=True, robust=False, xticklabels=picks_category, 
                     yticklabels=picks_category, ax=ax[1], label='place')
ax[0].set_title('Face presentation')
ax[1].set_title('Place presentation')
plt.show()
plt.legend()

# %% Plot colormap


fig, ax = plt.subplots(1, 2, figsize=(8, 6))
im = ax[0].imshow(GC_f, cmap= 'cividis')
im = ax[1].imshow(GC_p, cmap= 'cividis')

for axi in ax:
    axi.set_xticks(np.arange(len(picks_category)))
    axi.set_yticks(np.arange(len(picks_category)))
    axi.set_xticklabels(picks_category)
    axi.set_yticklabels(picks_category)
    
ax[0].set_title('Conditional G causality, Face stimuli')
ax[1].set_title('Conditional G causality, Place stimuli')
cb_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cb_ax)
