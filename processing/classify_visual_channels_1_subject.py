#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:44:21 2021

@author: guime
"""



import HFB_process as hf
import cifar_load_subject as cf
import scipy as sp
import re 
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path, PurePath
from mne.viz import plot_filter, plot_ideal_filter
from scipy import signal, fftpack
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection, multipletests
from netneurotools import stats as nnstats
from scipy.stats.stats import _chk2_asarray
from sklearn.utils.validation import check_random_state

pd.options.display.max_rows = 999
#%%

sub = 'DiAs'
proc = 'preproc' 
sfreq = 500; 
suffix = '_BP_montage_preprocessed_raw'
ext = '.fif'

#%% Read preprocessed data

subject = cf.Subject(name=sub)
datadir = subject.processing_stage_path(proc=proc)
fname = sub + '_BP_montage_HFB_raw.fif'
fpath = datadir.joinpath(fname)
HFB = mne.io.read_raw_fif(fpath, preload=True)        
dfelec = subject.df_electrodes_info()
HFB_db = hf.HFB_to_db(HFB)

#%% 

def multiple_test(A_po, A_pr, nchans, alpha=0.05):
    A_po = hf.sample_mean(A_po)
    A_pr = hf.sample_mean(A_pr)
    # Initialise inflated p values
    pval = [0]*nchans
    tstat = [0]*nchans
    # Compute inflated stats
    for i in range(0,nchans):
        tstat[i], pval[i] = nnstats.permtest_rel(A_po[:,i], A_pr[:,i])  
    # Correct for multiple testing    
    reject, pval_correct = fdrcorrection(pval, alpha=alpha)
    return reject

def cohen_d(x, y):
    
    n1 = np.size(x)
    n2 = np.size(y)
    m1 = np.mean(x)
    m2 = np.mean(y)
    s1 = np.var(x)
    s2 = np.var(y)
    
    s = (n1 - 1)*(s1**2) + (n2 - 1)*(s2**2)
    s = s/(n1+n2-2)
    s= np.sqrt(s)
    num = m1 - m2
    
    cohen = num/s
    
    return cohen

def compute_visual_responsivity(A_po, A_pr):
    
    nchan = A_po.shape[1]
    visual_responsivity = [0]*nchan
    
    for i in range(nchan):
        x = np.ndarray.flatten(A_po[:,i,:])
        y = np.ndarray.flatten(A_pr[:,i,:])
        visual_responsivity[i] = cohen_d(x,y)
        
    return visual_responsivity


def visual_chans_stats(reject, visual_responsivity, HFB_db):
    """Used in detect_visual_chan function"""
    idx = np.where(reject==True)
    idx = idx[0]
    visual_chan = []
    effect_size = []
    
    for i in list(idx):
        if visual_responsivity[i]>0:
            visual_chan.append(HFB_db.info['ch_names'][i])
            effect_size.append(visual_responsivity[i])
        else:
            continue
    return visual_chan, effect_size

def detect_visual_chan(HFB_db, tmin_pr=-0.4, tmax_pr=-0.1, tmin_po=0.1, tmax_po=0.5, alpha=0.05):
    """Return statistically significant visual channels with effect size"""
    A_pr = hf.crop_HFB(HFB_db, tmin=tmin_pr, tmax=tmax_pr)
    A_po = hf.crop_HFB(HFB_db, tmin=tmin_po, tmax=tmax_po)
    nchans = len(HFB_db.info['ch_names'])
    reject = multiple_test(A_pr, A_po, nchans, alpha=alpha)
    visual_responsivity = compute_visual_responsivity(A_po, A_pr)
    visual_chan, effect_size = visual_chans_stats(reject, visual_responsivity, HFB_db)
    return visual_chan, effect_size

    
#%%
A_pr = hf.crop_HFB(HFB_db, tmin=-0.4, tmax=-0.1)
A_po = hf.crop_HFB(HFB_db, tmin=0.1, tmax=0.4)
nchans = len(HFB_db.info['ch_names'])
t_stats = multiple_test(A_pr, A_po, nchans, alpha=0.01) 


#%% Detect visual chans
    
visual_chan, effect_size = hf.detect_visual_chan(HFB_db, tmin_pr=-0.1, tmax_pr=0,
                                         tmin_po=0.1, tmax_po=0.5, alpha=0.05)
#%% 

