#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:04:52 2021
Classify visual channels for all subjects
@author: guime
"""
import mne
import pandas as pd
import HFB_process as hf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path, PurePath
from config import args

#%% Detect visual channels
for subject in args.cohort:

    ecog = hf.Ecog(args.cohort_path, subject=subject, proc='preproc', 
                       stage = '_hfb_db_epo.fif', epoch=True)
    hfb = ecog.read_dataset()
    visual_detection = hf.VisualDetector(tmin_prestim=args.tmin_prestim, 
                                             tmax_prestim=args.tmax_prestim, 
                                             tmin_postim=args.tmin_postim,
                                             tmax_postim=args.tmax_postim, 
                                             alpha=args.alpha, 
                                             zero_method=args.zero_method, 
                                             alternative=args.alternative)
    visual_chan, effect_size = visual_detection.detect(hfb)
    #%% Classify visual channels
    
    dfelec = ecog.read_channels_info()
    visual_classifier = hf.VisualClassifier(tmin_prestim=args.tmin_prestim, 
                                             tmax_prestim=args.tmax_prestim, 
                                             tmin_postim=args.tmin_postim,
                                             tmax_postim=args.tmax_postim, 
                                             alpha=args.alpha, 
                                             zero_method=args.zero_method, 
                                             alternative=args.alternative)
    visual_populations = visual_classifier.hfb_to_visual_populations(hfb, dfelec)
    
    # Save into csv file
    df_visual = pd.DataFrame.from_dict(visual_populations)
    df_visual = df_visual.sort_values(by='Y', ignore_index=True)
    fname = 'visual_channels.csv'
    subject_path = args.cohort_path.joinpath(subject)
    brain_path = subject_path.joinpath('brain')
    fpath = brain_path.joinpath(fname)
    df_visual.to_csv(fpath, index=False)

#%% Make a table of all visual channels for all subjects

columns = list(df_visual.columns)
columns.append('subject_id')
df_all_visual_chans = pd.DataFrame(columns=columns)

for subject in args.cohort:
    fname = 'visual_channels.csv'
    subject_path = args.cohort_path.joinpath(subject)
    brain_path = subject_path.joinpath('brain')
    fpath = brain_path.joinpath(fname)
    df_visual = pd.read_csv(fpath)
    subject_id = [subject]*len(df_visual)
    df_visual['subject_id'] = subject_id
    df_all_visual_chans = df_all_visual_chans.append(df_visual)
    
path = args.cohort_path
fname = 'all_visual_channels.csv'
fpath = path.joinpath(fname)
df_all_visual_chans.to_csv(fpath, index=False)

#%% Read all visual chans dataframe

df_all_visual_chans = pd.read_csv(fpath)