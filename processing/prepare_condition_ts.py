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

from pathlib import Path, PurePath
from config import args
from scipy.io import savemat

#%%

ecog = hf.Ecog(args.cohort_path, subject=args.subject, proc=args.proc, 
                       stage = args.stage, epoch=args.epoch)
hfb = ecog.read_dataset()

# read ROI info for mvgc
df_visual = ecog.read_channels_info(fname=args.channels)
df_electrodes = ecog.read_channels_info(fname='electrodes_info.csv')
functional_indices = hf.parcellation_to_indices(df_visual, 'group', matlab=True)
ROI_indices = hf.parcellation_to_indices(df_visual, 'DK', matlab=True)
ROI_indices = {'LO': ROI_indices['ctx-lh-lateraloccipital'], 
               'Fus': ROI_indices['ctx-lh-fusiform'] }
visual_chan = df_visual['chan_name'].to_list()

# Read condition specific time series

ts, time = hf.category_ts(hfb, visual_chan, sfreq=args.sfreq, tmin_crop=args.tmin_crop, 
                          tmax_crop=args.tmax_crop)

#%% Save time series for GC analysis

ts_dict = {'data': ts, 'sfreq': args.sfreq, 'time': time, 'sub_id': args.subject, 
           'ROI_indices': ROI_indices, 'functional_indices': functional_indices}
fname = args.subject + '_ts_visual.mat'
subject_path = args.cohort_path.joinpath(args.subject)
proc_path = subject_path.joinpath('EEGLAB_datasets', args.proc)
fpath = proc_path.joinpath(fname)
savemat(fpath, ts_dict)

#%%

