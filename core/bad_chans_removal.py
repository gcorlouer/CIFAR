#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:33:20 2021
This script remove bad channels for all subjects and concatenate ieeg dataset
@author: guime
"""


import mne
import pandas as pd
import HFB_process as hf
import numpy as np

from pathlib import Path, PurePath
from config import args 
#%%

for subject in args.cohort:
    ecog = hf.Ecog(args.cohort_path, subject=subject, proc='bipolar_montage')
    raw_concat = ecog.concatenate_raw()
    raw_concat = hf.drop_bad_chans(raw_concat, q=99, voltage_threshold=500e-6, n_std=5)
    # Check if channels looks ok by plotting psd
    raw_concat.plot_psd(xscale='log')
    # Save file
    subject_path = args.cohort_path.joinpath(subject)
    proc_path = subject_path.joinpath('EEGLAB_datasets', args.proc)
    fname = subject + '_bad_chans_removed_raw.fif'
    fpath = proc_path.joinpath(fname)
    raw_concat.save(fpath, overwrite=True)

#%%


