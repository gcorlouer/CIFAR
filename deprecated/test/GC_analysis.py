#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 16:18:20 2020

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
proc = 'preproc'
sub = 'DiAs'
task = 'stimuli'
run = '1'
t_pr = -0.1
t_po = 1.5
multitrial = False
cat = 'Face'

ext = '.mat'

path_visual = cf_load.visual_path()
df_visual = pd.read_csv(path_visual)

df = df_visual.loc[df_visual['subject_id']==sub]

def GC_cat_mat(sub='DiAs', cat='Face', multitrial=True, proc = 'preproc', ext = '.mat'):
    
    if multitrial == True:
        suffix = 'GC_multi_HFB_visual_epoch_' + cat
    else:
        suffix = 'GC_sliding_HFB_visual_epoch_' + cat
    subject = cf_load.Subject(sub)
    GC_fpath = subject.fpath(proc = proc, suffix=suffix, ext=ext)
    GC_mat = io.loadmat(GC_fpath)
    return GC_mat

GC_face_mat = GC_cat(sub='DiAs', cat='Face')
GC_place_mat = GC_cat(sub='DiAs', cat='Place')

# %% 

# Picks electrodes

picks = ['LTo6  ', 'LGRD60', 'LTo1  ','LGRD52', 'LGRD50', 'LTo4  ', 'LTp3  ']

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
                        proc = preproc, ext = ext)
    
    GC = extract_GC(picks, GC_mat)
    return GC 

GC_mat = GC_face_mat
GC = extract_GC(picks, GC_mat)

GC_f = chan_to_GC(picks, sub=sub, cat= 'Face', multitrial=multitrial)
GC_p = chan_to_GC(picks, sub=sub, cat= 'Place', multitrial=multitrial)


GC_face = GC_face_mat['F']
GC_face = np.nan_to_num(GC_face)

GC_f = np.zeros((len(ipicks),len(ipicks)))
for i, ip in enumerate(ipicks):
    for j, jp in enumerate(ipicks): 
        GC_f[i,j] = GC_face[i,j]


GC_place = GC_place_mat['F']
GC_place =  np.nan_to_num(GC_place)

GC_p = np.zeros((len(ipicks),len(ipicks)))
for i, ip in enumerate(ipicks):
    for j, jp in enumerate(ipicks): 
        GC_p[i,j] = GC_place[i,j]
        
        
ROIs = GC_face_mat['ROIs']
category = GC_face_mat['category']
ch_names = GC_face_mat['ch_names']

fig, ax = plt.subplots(ncols=2, sharey=False, sharex=False)
F1 = sn.heatmap(GC_f, cmap="YlGnBu", vmin=0, vmax=0.02, annot=False, 
                     square=True, robust=False, xticklabels=picks, 
                     yticklabels=picks, ax=ax[0], label='Face')
F1 = sn.heatmap(GC_p, cmap="YlGnBu", vmin=0, vmax=0.02, annot=False, 
                     square=True, robust=False, xticklabels=picks, 
                     yticklabels=picks, ax=ax[1], label='place')
ax[0].set_xlabel('Face')
ax[1].set_xlabel('Place')
plt.show()
plt.legend()