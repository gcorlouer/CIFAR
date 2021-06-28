#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 12:50:53 2021

Create a table containing results from GC estimation. Table contains all 
subjects, pairs, GC, MI, TE, significance, distance of pair and distance between
pairs. 
@author: guime
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import HFB_process as hf
import pandas as pd

from scipy.io import loadmat
from config import args
from pathlib import Path, PurePath

#%% Load result for each subject:

for subject_id in args.cohort:
    ecog = hf.Ecog(args.cohort_path, subject=subject_id, proc=args.proc, 
                           stage = args.stage, epoch=args.epoch)
    # Read visual channels 
    df_visual = ecog.read_channels_info(fname=args.channels)
    result_path = Path('~','projects', 'CIFAR','CIFAR_data', 'results').expanduser()
    fname = subject_id + '_FC.mat'
    functional_connectivity_path = result_path.joinpath(fname)
    fname = subject_id + '_spectral_GC.mat'
    spectral_gc_path = result_path.joinpath(fname)
    fc = loadmat(functional_connectivity_path)
    sgc = loadmat(spectral_gc_path)
    
    gc = fc['F']
    sig_gc = fc['sig_GC']
    mi = fc['MI']
    sig_mi = fc['sig_MI']
    smvgc = sgc['f']
    te = hf.GC_to_TE(gc)
    #%%
    
    columns = {'pair':[], 'cdt':[],'sub_id': [], 'dist':[], 'length':[], 'pcmi':[],
               'sig_mi':[],'pcgc':[], 'sig_pcgc':[],'pcte':[], 'smvgc':[]}
    
    df_results = pd.DataFrame(columns=columns)
    visual_chan = df_visual['group'].tolist()
    nchan = len(visual_chan)
    ncdt = 3
    cdt = ['Rest', 'Face', 'Place']
    for icdt in range(ncdt):
        for itarget in range(nchan):
            for jsource in range(nchan):
                columns['pair'].append(visual_chan[jsource] + visual_chan[itarget])
                columns['cdt'].append(cdt[icdt])
                columns['sub_id'].append(subject_id)
                dist = df_visual['Y'][jsource]
                length = np.abs(df_visual['Y'][jsource] - df_visual['Y'][itarget])
                columns['dist'].append(dist)
                columns['length'].append(length)
                columns['pcmi'].append(mi[itarget,jsource,icdt])
                columns['sig_mi'].append(sig_mi[itarget,jsource,icdt])
                columns['pcgc'].append(gc[itarget,jsource,icdt])
                columns['sig_pcgc'].append(sig_gc[itarget,jsource,icdt])
                columns['pcte'].append(te[itarget,jsource,icdt])
                columns['smvgc'].append(smvgc[itarget,jsource,:,icdt])
    
    df_results = pd.DataFrame.from_dict(columns)
    df_results = df_results.dropna()
    fname = subject_id + '_fc_results.csv'
    fpath = result_path.joinpath(fname)
    df_results.to_csv(fpath, index=False)
#%% Concatenate FC for all subjects
columns = {'pair', 'cdt','sub_id', 'dist', 'length', 'pcmi',
               'sig_mi','pcgc', 'sig_pcgc','pcte', 'smvgc'}
df_all_subjects_results = pd.DataFrame(columns=columns)
for subject_id in args.cohort:
    fname = subject_id + '_fc_results.csv'
    fpath = result_path.joinpath(fname)
    df_results = pd.read_csv(fpath)
    df_all_subjects_results = df_all_subjects_results.append(df_results)

fname = 'all_subjects_fc_results.csv'
fpath = result_path.joinpath(fname)
df_all_subjects_results.to_csv(fpath, index=False)
print(f"Save all results to {fpath}")