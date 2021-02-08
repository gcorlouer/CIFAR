#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:42:17 2021

@author: guime
"""

import HFB_process as hf
import cifar_load_subject as cf
import scipy as sp
import re 
import os
import shutil
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path, PurePath
from mne.viz import plot_filter, plot_ideal_filter
from scipy import signal, fftpack
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection, multipletests

from scipy.io import loadmat,savemat
# %%

subjects = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
proc = 'preproc'
#%%

for sub_id in ['JuRo', 'NeLa', 'SoGi']:
    subject = cf.Subject(name=sub_id)
    datadir = subject.processing_stage_path(proc=proc)
    
    home = Path('~').expanduser()
    lionel_path = home.joinpath('HFB_data', sub_id)
    os.mkdir(lionel_path)
    
    fname = sub_id + '_visual_HFB_all_categories.mat'
    fpath = datadir.joinpath(fname)
    HFB_visual = loadmat(fpath)
    HFB_visual_path = lionel_path.joinpath(fname)
    savemat(HFB_visual_path, HFB_visual)
    # shutil(HFB_visual_path, lionel_path)
    
    fname = sub_id + '_BP_montage_HFB_raw.fif'
    fpath = datadir.joinpath(fname)
    HFB  = mne.io.read_raw_fif(fpath, preload=True)  
    dfelec = subject.df_electrodes_info()
    X = HFB.copy().get_data()
    raw_HFB = sub_id + '_BP_montage_HFB_raw.mat'
    raw_HFB_path = lionel_path.joinpath(raw_HFB)
    savemat(raw_HFB_path, dict(data=X) )
    elec_file = 'electrodes_info.csv'
    elec_path = lionel_path.joinpath(elec_file)
    dfelec.to_csv(elec_path, index=False)
    
path = cf.cifar_ieeg_path()
fname = 'visual_channels_BP_montage.csv'
fpath = path.joinpath(fname)
HFB_path = home.joinpath('HFB_data')
shutil.copy(fpath, HFB_path)

