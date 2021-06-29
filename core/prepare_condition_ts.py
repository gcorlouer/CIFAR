#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:43:20 2021
This script prepare condition specific time series for further analysis in 
mvgc toolbox
@author: guime
"""

#%%

import mne
import pandas as pd
import HFB_process as hf
import numpy as np
import matplotlib.pyplot as plt
import argparse

from pathlib import Path, PurePath
from config import args
from scipy.io import savemat


#%% Loading data parameters

cohort = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']

# Path to subjects repository. Enter your own path in local machine
cohort_path = Path('~','projects', 'CIFAR', 'CIFAR_data', 'iEEG_10', 
                   'subjects').expanduser()

parser = argparse.ArgumentParser()
parser.add_argument("--cohort_path", type=list, default=cohort_path)
parser.add_argument("--cohort", type=list, default=cohort)
parser.add_argument("--subject", type=str, default='DiAs')
parser.add_argument("--proc", type=str, default='preproc')
parser.add_argument("--stage", type=str, default='_hfb_extracted_raw.fif')
parser.add_argument("--epoch", type=bool, default=False)
parser.add_argument("--channels", type=str, default='visual_channels.csv')


# Preprocessing stages:

# _bad_chans_removed_raw.fif: Bad channels removed and concatenated lfp
# '_hfb_extracted_raw.fif' : extracted hfb
# '_hfb_db_epo.fif' epoched and db transformed hfb
# _preprocessed_raw.fif

# Channels:
# 'visual_channels.csv'
# 'all_visual_channels.csv'
# 'electrodes_info.csv'

#%% Create category specific time series

parser.add_argument("--sfreq", type=float, default=150)
parser.add_argument("--tmin_crop", type=float, default=0.3)
parser.add_argument("--tmax_crop", type=float, default=1.5)

#%% Functional connectivity parameters

parser.add_argument("--nfreq", type=float, default=1024)
parser.add_argument("--roi", type=str, default="functional")

# Types of roi:
# functional
# anatomical

#%%

args = parser.parse_args()

#%% Prepare condition time series

ecog = hf.Ecog(args.cohort_path, subject=args.subject, proc=args.proc, 
                       stage = args.stage, epoch=args.epoch)
hfb = ecog.read_dataset()

# Read ROI info for mvgc
df_visual = ecog.read_channels_info(fname=args.channels)
df_electrodes = ecog.read_channels_info(fname='electrodes_info.csv')
functional_indices = hf.parcellation_to_indices(df_visual, 'group', matlab=True)
visual_chan = df_visual['chan_name'].to_list()

# Read condition specific time series
# Read visual hfb
if args.stage == '_hfb_extracted_raw.fif':
    ts_type = 'hfb'
    ts, time = hf.category_ts(hfb, visual_chan, sfreq=args.sfreq, tmin_crop=args.tmin_crop, 
                              tmax_crop=args.tmax_crop)
# Read visual lfp
else:
    ts_type = 'lfp'
    ts, time = hf.category_lfp(hfb, visual_chan, sfreq=args.sfreq, tmin_crop=args.tmin_crop, 
                              tmax_crop=args.tmax_crop)

#%% Save time series for GC analysis

ts_dict = {'data': ts, 'sfreq': args.sfreq, 'time': time, 'sub_id': args.subject, 
           'functional_indices': functional_indices, 'ts_type':ts_type}
fname = 'condition_ts_visual.mat'
result_path = Path('~','projects','CIFAR','CIFAR_data', 'results').expanduser()
fpath = result_path.joinpath(fname)
savemat(fpath, ts_dict)
#%% To look at anatomical regions

# ROI_indices = hf.parcellation_to_indices(df_visual, 'DK', matlab=True)
# ROI_indices = {'LO': ROI_indices['ctx-lh-lateraloccipital'], 
#                'Fus': ROI_indices['ctx-lh-fusiform'] }
