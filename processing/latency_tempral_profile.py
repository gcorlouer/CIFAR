#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 10:32:55 2021

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

# TODO: Plot linear regression
#%%

sub_id = 'DiAs'
proc = 'preproc' 
sfreq = 500; 
suffix = '_BP_montage_preprocessed_raw'
ext = '.fif'

#%% Read preprocessed data

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_chan = subject.pick_visual_chan()
HFB = hf.visually_responsive_HFB(sub_id = sub_id)

path = cf.cifar_ieeg_path()
fname = 'visual_channels_BP_montage.csv'
fpath = path.joinpath(fname)
visual_table = pd.read_csv(fpath)
#%% 

latency = visual_table['latency']
category_selectivity = visual_table['category_selectivity']
visual_response = visual_table['visual_responsivity']
Y_coord = visual_table['Y']
X_coord = visual_table['X']
Z_coord = visual_table['Z']

#%% Compute linear regression

linreg = stats.linregress(visual_response, latency)
print(linreg)

a = linreg.slope
b = linreg.intercept
x = range(50,400)
y = a*x+b 

#%% Correlation

corr = stats.pearsonr(latency, Y_coord)

# %% Plot linear regression

#%matplotlib 
plt.rcParams.update({'font.size': 22})

plt.subplot(2,2,1)
plt.plot(latency, Y_coord, '.')
plt.xlabel('latency response (ms)')
plt.ylabel('Y coordinate (mm)')

plt.subplot(2,2,2)
plt.plot(latency, Z_coord, '.')
plt.xlabel('latency response (ms)')
plt.ylabel('Z coordinate (mm)')


plt.subplot(2,2,3)
plt.plot(latency, visual_response, '.')
plt.xlabel('latency response (ms)')
plt.ylabel(' visual responsivity (db)')

plt.subplot(2,2,4)
plt.plot(latency, category_selectivity, '.')
plt.xlabel('latency response (ms)')
plt.ylabel('category selectivity (dB)')






