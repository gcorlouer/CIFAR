#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:35:03 2020

@author: guime
"""
import HFB_process
import cf_load
import os
import scipy as sp
import re 
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path, PurePath

#%matplotlib


sub = 'KiAl' 
subject = cf_load.Subject(name=sub)
subject_path = subject.subject_path()
data_path = 'EEGLAB_datasets/raw_signal/KiAl_freerecall_1_preprocessed.set'
fpath = subject_path.joinpath(data_path)
fpath = os.fspath(fpath)
raw = mne.io.read_raw_eeglab(fpath, preload=True)

#raw.copy().pick('TRIG').plot(duration=50, scalings=1e-5)

#%% 

raw_c = raw.copy()
raw_c.pick('TRIG').plot(duration=250, scalings=1e-4)