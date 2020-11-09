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

from pathlib import Path, PurePath
from mne.viz import plot_filter, plot_ideal_filter
from scipy import signal, fftpack
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection, multipletests


pd.options.display.max_rows = 999

#%% Parameters 
sub_id = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
proc = 'preproc' # Line noise removed
for sub in ['JuRo', 'NeLa', 'SoGi']:
    # Subject parameters
    sub = sub
    task = 'stimuli' # stimuli or rest_baseline_1
    run = '1'
    sfreq = 500; 
    suffix = 'preprocessed_raw'
    ext = '.fif'
    
    #%% Read preprocessed data
    
    subject = cifar_load_subject.Subject(name=sub, task= task, run = run)
    fpath = subject.dataset_path(proc = proc, suffix=suffix, ext = ext)
    raw = mne.io.read_raw_fif(fpath, preload=True)    
    
    #%% Extract log normalised HFB envelope
    
    bands = HFB_process.freq_bands() # Select Bands of interests 
    HFB_db = HFB_process.raw_to_HFB_db(raw, bands)
    # Extract specific trials
    events, event_id = mne.events_from_annotations(raw)
    face_id = HFB_process.extract_stim_id(event_id)
    place_id = HFB_process.extract_stim_id(event_id, cat='Place')
    image_id = face_id+place_id
    
#%% Detect visual channels and extract their log normalised amplitude 
    
    # Detect visual channels
    visual_channels, tstat = HFB_process.detect_visual_chan(HFB_db, tmin_pr=-0.4, 
                                                            tmax_pr=-0.1, tmin_po=0.1, tmax_po=0.5)
    # Extract visual channels log normalised amplitude                                                                
    visual_HFB = HFB_db.copy().pick_channels(visual_channels)
    # Compute latency response of visual channels
    latency_response = HFB_process.compute_latency(visual_HFB, image_id, visual_channels) 

    #%% Classify channels into  V1 and V2 populations
    
    dfelec = subject.df_electrodes_info()
    latency_threshold = 180
    # Classify V1 and V2 populations
    group = HFB_process.classify_retinotopic(latency_response, visual_channels, 
                                             dfelec, latency_threshold=latency_threshold)
    
    # %% Classify Face and Place populations
    
    group, tstat_visual = HFB_process.classify_Face_Place(visual_HFB, face_id, place_id, 
                                visual_channels, group, alpha=0.05);
    
    # %% Create table of visual channels for a given subject
    
    nvisual = len(visual_channels)
    
    subject_id = [0]*nvisual
    brodman = [0]*nvisual
    DK = [0]*nvisual
    chan_name = [0]*nvisual
    category = [0]*nvisual
    effect_size = [0]*nvisual
    
    for idx, channel in enumerate(visual_channels):
        subject_id[idx] = sub
        brodman[idx] = dfelec['Brodman'].loc[dfelec['electrode_name']==channel].values[0]
        DK[idx] =  dfelec['ROI_DK'].loc[dfelec['electrode_name']==channel].values[0]
    
    
    functional_group = {'subject_id': subject_id, 'Brodman': brodman, 'DK': DK, 
                        'chan_name':visual_channels, 'group': group, 'latency_response': latency_response,
                        'effect_size': tstat_visual}
    
    df_visual = pd.DataFrame(functional_group)
    
    brain_path = subject.brain_path()
    fname = 'visual_channels.csv'
    fpath = brain_path.joinpath(fname)
    df_visual.to_csv(fpath, index=False)
    
    df_visual = pd.read_csv(fpath)
#%% Create table of visual channels for all subject

# functional_group = {'subject_id': [], 'chan_name':[], 'category': [], 
#                     'Brodman': [], 'DK': [], 'effect_size': []}

# functional_group['subject_id'].extend([subject.name]*len(cat))
#         functional_group['chan_name'].extend(cat)
#         functional_group['category'].extend([key]*len(cat))
#         functional_group['brodman'].extend(subject.ROIs(cat))
#         functional_group['DK'].extend(subject.ROI_DK(cat))
