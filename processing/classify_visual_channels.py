#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:38:01 2020

@author: guime
"""


import HFB_process
import cifar_load_subject
import scipy as sp
import re 
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

from mne.viz import plot_filter, plot_ideal_filter
from scipy import signal, fftpack
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection, multipletests


pd.options.display.max_rows = 999

#%% 

# Subject parameters
proc = 'preproc' # Line noise removed
sub = 'DiAs' 
task = 'stimuli' # stimuli or rest_baseline_1
run = '1'
sfreq = 500; 
suffix = 'preprocessed_raw'
ext = '.fif'

ext2save = '.mat'
suffix2save = 'visual_chans.mat'
#%% 

subject = cifar_load_subject.Subject(name=sub, task= task, run = run)
fpath = subject.dataset_path(proc = proc, suffix=suffix, ext = ext)
raw = mne.io.read_raw_fif(fpath, preload=True)
X = raw.get_data()

#%% Classify and extract visual channels 
bands = HFB_process.freq_bands() # Select Bands of interests 
HFB_db = HFB_process.raw_to_HFB_db(raw, bands)

events, event_id = mne.events_from_annotations(raw)
face_id = HFB_process.extract_stim_id(event_id)
place_id = HFB_process.extract_stim_id(event_id, cat='Place')
image_id = face_id+place_id

visual_channels, tstat = HFB_process.detect_visual_chan(HFB_db, tmin_pr=-0.4, 
                                                           tmax_pr=-0.1, tmin_po=0.1, tmax_po=0.5)
visual_HFB = HFB_db.copy().pick_channels(visual_channels)
# %% Compute latency response

latency_response = HFB_process.compute_latency(visual_HFB, image_id, visual_channels)

#%% Classify V1 and V2 channels

dfelec = subject.df_electrodes_info()
latency_threshold = 180

group = HFB_process.classify_retinotopic(latency_response, visual_channels, 
                                         dfelec, latency_threshold=latency_threshold)

# %% Classify Face and Place selective electrodes

group, tstat_high = HFB_process.classify_Face_Place(visual_HFB, face_id, place_id, 
                            visual_channels, group, alpha=0.05);
# %% Create table of visual channels for all subjects 

functional_group = {'subject_id': [], 'chan_name':[], 'category': [], 
                    'brodman': [], 'DK': []}


