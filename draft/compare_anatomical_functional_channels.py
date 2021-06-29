#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:51:02 2021

@author: guime
"""

import mne
import pandas as pd
import HFB_process as hf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path, PurePath
from config import args

#%%
path = args.cohort_path
fname = 'all_visual_channels.csv'
fpath = path.joinpath(fname)

df_all_visual_chans = pd.read_csv(fpath)

#%% Create table with all channels

ecog = hf.Ecog(args.cohort_path, subject=args.subject, proc=args.proc, 
                   stage = args.stage, epoch=args.epoch)
df_elec = ecog.read_channels_info()
columns = list(df_elec.columns)
columns.append('subject_id')
df_all_elec = pd.DataFrame(columns=columns)

for subject in args.cohort:
    ecog = hf.Ecog(args.cohort_path, subject=args.subject, proc=args.proc, 
                       stage = args.stage, epoch=args.epoch)
    df_elec = ecog.read_channels_info()
    subject_id = [subject]*len(df_elec)
    df_elec['subject_id'] = subject_id
    df_elec = df_elec.sort_values(by='Y', ignore_index=True)
    df_all_elec = df_all_elec.append(df_elec)

#%%

anatomical_visual_regions = list(df_all_visual_chans['DK'].unique())
