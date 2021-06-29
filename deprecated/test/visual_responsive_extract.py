#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:45:35 2020

@author: guime
"""


import helper
import mne
import scipy.io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as spstats
import statsmodels.stats as stats
import csv
import os 
import argparse

pd.options.display.html.table_schema = True
pd.options.display.max_rows = None

from statsmodels.stats.multitest import fdrcorrection, multipletests
from pathlib import Path, PurePath
from numpy import inf

%matplotlib
plt.rcParams.update({'font.size': 17})

#%% 

# Parameters 
# Subject and task
subject = 'DiAs'
subject_id = '04'
proc = 'raw'
montage = 'preproc'
suffix = '_lnrmv.set'
task = 'stimuli'
run = '1'

# High frequency bands 
l_freq = 60
nband = 6 
band_size = 20 

# Epoch parameter

tmin = -0.5 # Prestimulus
tmax = 1.75 # Poststimulus

# Statitics parameters

alpha = 0.01 # significance threshold of visual response

# Saving paramerters 

save2 = Path('~','projects','CIFAR','visual_info').expanduser()
task_save = 'stimuli'
suffix_place = '_epoch_place.mat'
suffix_face = '_epoch_face.mat'

#%% Extract and normalise amplitude

raw, dfelec = helper.import_data(task=task, proc=proc, montage=montage, 
                                 run=run, subject=subject, subject_id=subject_id, suffix=suffix)
HFB, raw_HFB = helper.HFB_raw(raw, l_freq=60, nband=6, band_size=20);
events, event_id = mne.events_from_annotations(raw)
place_id, face_id = helper.stim_id(event_id)
epochs = mne.Epochs(raw_HFB, events, event_id= event_id, tmin=tmin, 
                    tmax=tmax, baseline=None,preload=True)
HFB_db = helper.HFB_norm(epochs, events, tmin)

# %% Detect visual responsive electrodes

A_pr = HFB_db.copy().crop(tmin=-0.4, tmax=-0.1).get_data()
A_po = HFB_db.copy().crop(tmin=0.1, tmax=0.5).get_data()
reject, pval_correct, visual_chan, visual_cohen = helper.detect_visual(A_pr, A_po, HFB_db, alpha)

# %% Detect place lectrodes 

A_pr = HFB_db[place_id].copy().crop(tmin=-0.4, tmax=-0.1).get_data()
A_po = HFB_db[place_id].copy().crop(tmin=0.1, tmax=0.5).get_data()
reject, pval_correct, face_chan, face_cohen = helper.detect_visual(A_pr, A_po, HFB_db, alpha)

# %% Detect face lectrodes 

A_pr = HFB_db[face_id].copy().crop(tmin=-0.4, tmax=-0.1).get_data()
A_po = HFB_db[face_id].copy().crop(tmin=0.1, tmax=0.5).get_data()
reject, pval_correct, place_chan, place_cohen = helper.detect_visual(A_pr, A_po, HFB_db, alpha)

# %% 

pure_place = list(set(face_chan)-set(place_chan))
pure_face = list(set(place_chan)-set(face_chan))
cat = list(set(visual_chan) - set(pure_place))
cat = list(set(category_selective) -set(pure_face))
noncat = list(set(visual_chan)-set(face_chan))ime/projects/CIFAR/new_code/test/argparse_test.py', args='--x 4', wdir='/home/guime/projects/CIFAR/new_code/test')
16

In [27]: runfile('/home/guime/projects/CIFAR/new_code/test/argparse_test.py', args='--x 4', wdir='/home/guime/projects/CIFAR/new_code/test')
16

In [28]: runfile('/home/guime/projects/CIFAR/new_code/test/argparse_test.py', args='--x 4', wdir='/home/guime/projects/CIFAR/new_code/test')
16

In [29]: runfile('argparse_test.py', args='--x 5')
25

In [30]: runfile('argparse_
noncat = list(set(noncat)-set(place_chan))

# %% Create table

visual_elec = {'Subject': [], 'Electrode': [], 'Category':[], 'Cohen': [],
                         'Brodman': []}
for ichan, chan in enumerate(visual_chan):
    visual_elec['Subject'].append(subject)
    visual_elec['Electrode'].append(chan)
    if chan in pure_place:
        visual_elec['Category'].append('place')
        visual_elec['Cohen'].append(visual_cohen[ichan])
    elif chan in pure_face:
        visual_elec['Category'].append('face')
        visual_elec['Cohen'].append(visual_cohen[ichan])
    elif chan in cat :
        visual_elec['Category'].append('cat')
        visual_elec['Cohen'].append(visual_cohen[ichan])
    else: 
        visual_elec['Category'].append('ncat')
        visual_elec['Cohen'].append(visual_cohen[ichan])
    if chan == 'TRIG':
        visual_elec['Brodman'].append('na')
    else: 
        visual_elec['Brodman'].append(dfelec['Brodman'].loc[dfelec['electrode_name']==chan].iloc[0])


df = pd.DataFrame(data = visual_elec)
df.to_csv(f'{subject}_visual_info.csv', index=False)
fpath = save2
fname = f'{subject}_visual_info.csv'
fpath = fpath.joinpath(fname)
