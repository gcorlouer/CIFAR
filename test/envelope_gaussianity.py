#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:18:32 2020

@author: guime
"""

# Prepare data for correlation analysis.

import HFB_test
import cf_load
import scipy as sp
import re 
import numpy as np 
import seaborn as sn
from scipy.stats import kurtosis, skew

import mne
import matplotlib.pyplot as plt
import pandas as pd

from scipy import io
# TODO : Change order of chan names for xlabels, try other scale, compare.
plt.rcParams.update({'font.size': 30})

%matplotlib

# %% Parameters
proc = 'preproc'
ext2save = '.mat'
sub = 'AnRa'
task = 'stimuli'
run = '1'
t_pr = -0.1
t_po = 1.75
cat = 'Place'
suffix2save = 'HFB_visual_epoch_' + cat

# cat_id = extract_stim_id(event_id, cat = cat)
# %% Import data
# Load visual channels
path_visual = cf_load.visual_path()
df_visual = pd.read_csv(path_visual)

# Load data
subject = cf_load.Subject(name=sub, task= task, run = run)
df_elec = subject.dfelec()

fpath = subject.fpath(proc = proc, suffix='lnrmv')
raw = subject.import_data(fpath)

bands = HFB_test.freq_bands() # Select Bands of interests 

HFB_db = HFB_test.extract_HFB_db(raw, bands)
HFB = HFB_test.extract_HFB(raw, bands)



# %% Check distribution

visual_chans = list(df_visual['chan_name'].loc[df_visual['subject_id']==sub])

HFB_visual = HFB.copy().pick(picks=visual_chans)

HFB_visual = HFB_visual.get_data()

# Plot histogram 


HFB_visual_flat = HFB_visual.flatten()


plt.hist(HFB_visual_flat, bins=1000, density=True)
plt.xscale('log')
plt.title('Histogram of visually responsive channel envelope, subject DiAs')
plt.ylabel('Number of occurence')
plt.xlabel(' HFB log Amplitude')

#%% 

def plot_hist(data, bins=1000, density=True, scale='linear'):
    plt.hist(data, bins=bins, density=density)
    plt.xscale('linear')
    plt.title('Histogram of visually responsive channel envelope, subject DiAs')
    plt.ylabel('Number of occurence')
    plt.xlabel(' HFB log Amplitude')
    
plot_hist(visual_dict['data'].flatten())

#%% 
HFB_log = np.log(HFB_visual_flat)
skewness = skew(HFB_log)
Kurtosis = kurtosis(HFB_log)
print(f'Skewness is {skewness}' )
print(f'Kurtosis is {Kurtosis}')

#%% downsampling


