#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:33:20 2021

@author: guime
"""


import mne
import pandas as pd
import HFB_process as hf
import numpy as np

from pathlib import Path, PurePath
from config import args 
#%%
q = 99
#%% Test read data

ecog = hf.Ecog(args.cohort_path, subject=args.subject, proc='bipolar_montage')
raw_concat = ecog.concatenate_raw()

#%% Drop bad chans

raw_concat = hf.drop_bad_chans(raw_concat)

#%% Plot psd

raw_concat.plot_psd(xscale='log')
