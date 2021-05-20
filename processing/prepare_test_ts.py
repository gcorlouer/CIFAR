#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 18:13:34 2021

@author: guime
"""
import mne
import pandas as pd
import HFB_process as hf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path, PurePath
from config import args

#%% load data and visual chans info

subject='DiAs'
ecog = hf.Ecog(args.cohort_path, subject=subject, proc='preproc', 
                   stage = '_hfb_db_epo.fif', epoch=True)
hfb = ecog.read_dataset()
fname = 'visual_channels.csv'
subject_path = args.cohort_path.joinpath(subject)
brain_path = subject_path.joinpath('brain')
fpath = brain_path.joinpath(fname)
df_visual = pd.read_csv(fpath)

chans = ['LTo1-LTo2', 'LTo5-LTo6']

#%%

