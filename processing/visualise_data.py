#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:50:54 2020

@author: guime
"""


import HFB_process
import cf_load
import scipy as sp
import re 
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

%matplotlib

# %% Parameters
proc = 'preproc'
ext2save = '.mat'
sub = 'DiAs'
task = 'stimuli'
run = '1'
duration = 10 # Event duration for resting state
t_pr = -0.1
t_po = 5
cat = 'Rest'
suffix2save = 'HFB_visual_epoch_' + cat

# cat_id = extract_stim_id(event_id, cat = cat)
# %% Import data
# Load visual channels
path_visual = cf_load.visual_path()
df_visual = pd.read_csv(path_visual)

# Load data
subject = cf_load.Subject(name=sub, task= task, run = run)
fpath = subject.fpath(proc = proc, suffix='lnrmv')
raw = subject.import_data(fpath)

# %% Visualise data

raw.plot(duration = 60, scalings = None)