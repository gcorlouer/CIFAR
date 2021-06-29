#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 15:28:23 2020

@author: guime
"""


import cifar_load_subject as cf
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

from mne.viz import plot_alignment, snapshot_brain_montage

#%% 

pd.options.display.max_rows = 999

sub_id = 'DiAs'
proc = 'preproc' 
sfreq = 500; 
suffix = '_BP_montage_preprocessed_raw'
ext = '.fif'

subject = cf.Subject(name=sub_id)

df_elec = subject.df_electrodes_info()

#%% 

ch_name = df_elec['electrode_name'].tolist()
ch_coord = df_elec[['X', 'Y','Z']].to_numpy(dtype=float)

ch_pos = dict(zip(ch_name, ch_coord))

montage = mne.channels.make_dig_montage(ch_pos, coord_frame='mri')
print('Created %s channel positions' % len(ch_name))

trans = mne.channels.compute_native_head_t(montage)
print(trans)


#%% Read preprocessed data

datadir = subject.processing_stage_path(proc=proc)
fname = sub_id + '_BP_montage_preprocessed_raw.fif'
fpath = datadir.joinpath(fname)
raw = subject.read_eeglab()

# Drop bads channels
bads = ['Event', 'OSAT', 'PR', 'ECG', 'TRIG']
raw.drop_channels(bads)
# attach montage
raw.set_montage(montage)

# set channel types to ECoG (instead of EEG)
raw.set_channel_types({ch_name: 'ecog' for ch_name in raw.ch_names})

#%%

fig = plot_alignment(info = raw.info, surfaces=['pial'], coord_frame='mri')
mne.viz.set_3d_view(fig, 200, 70, focalpoint=[0, -0.005, 0.03])

#xy, im = snapshot_brain_montage(fig, montage)