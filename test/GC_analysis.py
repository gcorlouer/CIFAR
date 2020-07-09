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

# TODO : Change order of chan names for xlabels, try other scale, compare.
plt.rcParams.update({'font.size': 20})


# %% Import data
preproc = 'preproc'
sub = 'DiAs'
task = 'stimuli'
run = '1'
t_pr = -0.1
t_po = 1.5
cat = 'Face'
suffix = 'GC_HFB_visual_epoch_' + cat
ext = '.mat'

path_visual = cf_load.visual_path()
df_visual = pd.read_csv(path_visual)

df = df_visual.loc[df_visual['subject_id']==sub]

def GC_cat(sub='DiAs', cat='Face', proc = 'preproc'):
    suffix = 'GC_HFB_visual_epoch_' + cat
    ext = '.mat'
    subject = cf_load.Subject(sub)
    GC_fpath = subject.fpath(proc = proc, suffix=suffix, ext=ext)
    GC_mat = sp.io.loadmat(GC_fpath)
    return GC_mat

GC_face_mat = GC_cat(sub='DiAs', cat='Face')
GC_place_mat = GC_cat(sub='DiAs', cat='Place')

# %% 

GC_face = GC_face_mat['F']
GC_face = np.nan_to_num(GC_face)

GC_place = GC_place_mat['F']
GC_place =  np.nan_to_num(GC_place)

ROIs = GC_face_mat['ROIs']
category = GC_face_mat['category']
ch_names = GC_face_mat['ch_names']

fig, ax = plt.subplots(ncols=2, sharey=False, sharex=False)
F1 = sn.heatmap(GC_face, cmap="YlGnBu", vmin=0, vmax=0.002, annot=False, 
                     square=True, robust=False, xticklabels=ch_names, yticklabels=ch_names, ax=ax[0])
F1 = sn.heatmap(GC_place, cmap="YlGnBu", vmin=0, vmax=0.002, annot=False, 
                     square=True, robust=False, xticklabels=ch_names, yticklabels=ch_names, ax=ax[1])
plt.show()