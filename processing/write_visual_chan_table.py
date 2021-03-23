#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 10:32:55 2021

@author: guime
"""

import HFB_process as hf
import cifar_load_subject as cf
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from pathlib import Path, PurePath

#%% Load visual table
path = cf.cifar_ieeg_path()
fname = 'cross_subjects_visual_BP_channels.csv'
fpath = path.joinpath(fname)
df = pd.read_csv(fpath)

#%%

columns = ['Id', 'visual', 'R', 'F', 'P']
chan_table = pd.DataFrame(columns=columns)
subjects = df['subject_id'].unique().tolist()
chan_table['Id'] = subjects
for sub_id in subjects:
    chan_table['visual'].loc[chan_table['Id']==sub_id] = len(df.loc[df['subject_id']==sub_id])
    sub_df = df.loc[df['subject_id']==sub_id]
    for cat in ['R', 'F', 'P']: 
        chan_table[cat].loc[chan_table['Id']==sub_id] = len(sub_df.loc[sub_df['group'] == cat])
df_total = {'Id':['total'], 'visual' : chan_table['visual'].sum(), 'R': chan_table['R'].sum(), 'F' : chan_table['F'].sum(), 'P': chan_table['P'].sum()}
df_total = pd.DataFrame.from_dict(df_total)
chan_table = chan_table.append(df_total)

#%% Save chan table

ieeg_dir = cf.cifar_ieeg_path()
fname = 'visual_demographics.csv'
fpath = ieeg_dir.joinpath(fname)
chan_table.to_csv(fpath, index=False)

