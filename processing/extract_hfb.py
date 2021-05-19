#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:36:49 2021
This script extract HFB, epoch it, and rescale it into db for all subjects
@author: guime
"""

import mne
import pandas as pd
import HFB_process as hf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path, PurePath
from config import args

#%% Extract hfb

for subject in args.cohort:
    ecog = hf.Ecog(args.cohort_path, subject=subject, proc='preproc', 
                   stage = '_bad_chans_removed_raw.fif')
    raw = ecog.read_dataset()
    envelope = hf.Hfb(l_freq=args.l_freq, nband=args.nband, band_size=args.band_size, 
                      l_trans_bandwidth= args.l_trans_bandwidth,
                      h_trans_bandwidth= args.h_trans_bandwidth,
                      filter_length=args.filter_length, phase=args.phase,
                      fir_window=args.fir_window)
    hfb = envelope.extract_hfb(raw)
    subject_path = args.cohort_path.joinpath(subject)
    proc_path = subject_path.joinpath('EEGLAB_datasets', args.proc)
    fname = subject + '_hfb_extracted_raw.fif'
    fpath = proc_path.joinpath(fname)
    hfb.save(fpath, overwrite=True)
#%% Normalise HFB

for subject in args.cohort:
    ecog = hf.Ecog(args.cohort_path, subject=subject, proc='preproc', 
                       stage = '_hfb_extracted_raw.fif')
    raw = ecog.read_dataset()
    normal_hfb = hf.Hfb_db(t_prestim=args.t_prestim, 
                           t_postim = args.t_postim, baseline=args.baseline,
                           preload=args.preload, tmin_baseline=args.tmin_baseline, 
                           tmax_baseline=args.tmax_baseline, mode=args.mode)
    hfb_db = normal_hfb.hfb_to_db(raw)
    subject_path = args.cohort_path.joinpath(subject)
    proc_path = subject_path.joinpath('EEGLAB_datasets', args.proc)
    fname = subject + '_hfb_db_epo.fif'
    fpath = proc_path.joinpath(fname)
    hfb_db.save(fpath, overwrite=True)
#%% Check if plot make sense
