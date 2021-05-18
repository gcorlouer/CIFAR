#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 17:02:35 2021

@author: guime
"""

import mne
import os
import pandas as pd
import HFB_process as hf

from pathlib import Path, PurePath
from config import args 

#%% Test read data

ecog = hf.Ecog(args.cohort_path, proc='bipolar_montage')
raw = ecog.read_dataset(run=2, task='rest_baseline')

#%% Test read channels info

ecog = hf.Ecog(args.cohort_path, proc='bipolar_montage')
channels_info = ecog.read_channels_info(fname='electrodes_info.csv')

#%% Test concatenation

raw_concat = ecog.concatenate_raw()

#%% Plot concatenated data

raw_concat.plot(duration=10, scalings=1e-4)

#%% 
