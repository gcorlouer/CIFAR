#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 16:01:44 2020

@author: guime
"""



import HFB_process as hf
import cifar_load_subject as  cf
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

from scipy.io import loadmat,savemat

# %matplotlib

pd.options.display.max_rows = 999

sub = 'DiAs'

proc = 'preproc' 
sfreq = 500; 
cats = ['Rest', 'Place','Face']
tmin_crop = 0.5
tmax_crop = 1.75
suffix = 'preprocessed_raw'
ext = '.fif'

# %%

for cat in cats:
    subject = cf.Subject(name=sub)
    datadir = subject.processing_stage_path(proc=proc)
    fname = sub + '_BP_montage_HFB_raw.fif'
    fpath = datadir.joinpath(fname)
    raw = mne.io.read_raw_fif(fpath, preload=True)
    
    # Pick visual HFB
    
    brain_path = subject.brain_path()
    fname = 'visual_channels_BP_montage.csv'
    fpath = brain_path.joinpath(fname)
    visual_chan = pd.read_csv(fpath)
    
    # Drop unwanted channels
    
    visual_chan = visual_chan[visual_chan.group != 'other']
    visual_chan = visual_chan.reset_index(drop=True)
    visual_chan_name = visual_chan['chan_name'].values.tolist()
    group = visual_chan['group'].unique().tolist()
    
    # Extract visual HFB
    HFB_visual = raw.copy().pick_channels(visual_chan_name)
    
    # Extract category specific time series
    
    visual_data = hf.HFB_to_visual(HFB_visual, group, visual_chan, cat=cat, tmin_crop = tmin_crop, tmax_crop=tmax_crop)
    
    # Save data
    
    fname = sub + '_' + cat + '_visual_HFB.mat'
    fpath = datadir.joinpath(fname)
    
    # Save data in Rest x Face x Place array of time series
    
    savemat(fpath, visual_data)