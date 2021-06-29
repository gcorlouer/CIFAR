#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:58:44 2021

@author: guime
"""

#%% Import libraries

import HFB_process as hf
import cifar_load_subject as cf
import pandas as pd

from pathlib import Path, PurePath
from scipy.io import loadmat,savemat

# %% Parameters

pd.options.display.max_rows = 999

path = cf.cifar_ieeg_path()
fname = 'visual_channels_BP_montage.csv'
fpath = path.joinpath(fname)

sub_id = 'DiAs'
visual_chan_table = 'visual_channels_BP_montage.csv'
proc = 'preproc' 


suffix = 'preprocessed_raw'
ext = '.fif'


#%% Load visually responsive populations

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_chan = subject.pick_visual_chan()

path = cf.cifar_ieeg_path()
fname = 'visual_channels_BP_montage.csv'
fpath = path.joinpath(fname)

df =  pd.read_csv(fpath)

populations = df['group'].unique().tolist()
subjects = df['subject_id'].unique().tolist()

# %% Create table of visually responsive neural populations distribution
 
populations_table = {'subject_id':[], 'ON':[],'RN':[],'RP':[],'RF':[],'HP':[],'HF':[]}

df_populations = pd.DataFrame(columns=populations_table)
df_populations['subject_id'] = subjects

for subject_id in subjects:
    for population in populations:
         npop = df.loc[df['subject_id']==subject_id].loc[df['group']==population].shape[0]
         df_populations[population].loc[df_populations['subject_id']==subject_id] = npop
#%% Save table
         
fname = 'visual_populations_distributions.csv'
fpath = path.joinpath(fname)
df_populations.to_csv(fpath, index=False)