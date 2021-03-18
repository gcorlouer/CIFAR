#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:04:56 2021

TODO:   - Test skewness kurtosis on individual subjects
        - Save hfb and then extract hfb_db and test different baseline normalisation
        - Plot histogram  and boxplot for each individual subject
        - Plot histogram cross subjects
@author: guime
"""

import HFB_process as hf
import cifar_load_subject as cf
import mne
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy import stats
from pathlib import Path, PurePath

#%% Parameters
proc = 'preproc'
tmin_prestim=-0.4
tmax_prestim=-0.1
tmin_postim=0.2
tmax_postim=0.5
alpha=0.01
zero_method='pratt'
alternative='greater'

#%% Test visual channel detection for all subjects

subjects = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
postim_amplitude_list = [0]*len(subjects)
prestim_amplitude_list = [0]*len(subjects)
for i, sub_id in enumerate(subjects):
    subject = cf.Subject(name=sub_id)
    datadir = subject.processing_stage_path(proc=proc)
    fname = sub_id + '_hfb_db_epo.fif'
    fpath = datadir.joinpath(fname)
    hfb_db = mne.read_epochs(fpath, preload=True)
    visual_channels, effect_size = hf.detect_visual_chan(hfb_db, tmin_prestim=tmin_prestim, 
                                                  tmax_prestim=-tmax_prestim
                                                  ,tmin_postim=tmin_postim,
                           tmax_postim=tmax_postim, alpha=alpha, zero_method=zero_method, 
                           alternative=alternative)
    hfb_visual = hfb_db.copy().pick_channels(visual_channels).crop()
    A_prestim = hfb_visual.copy().crop(tmin=tmin_prestim, tmax=tmax_prestim).get_data()
    A_prestim = np.ndarray.flatten(A_prestim)
    A_postim = hfb_visual.copy().crop(tmin=tmin_postim, tmax=tmax_postim).get_data()
    A_postim = np.ndarray.flatten(A_postim)
    skewness_prestim = stats.skew(A_prestim)
    skewness_postim = stats.skew(A_postim)
    kurtosis_prestim = stats.kurtosis(A_prestim)
    kurtosis_postim = stats.kurtosis(A_postim)
    print(f'For prestimulus amplitude skewness is {skewness_prestim}, kurtosis is {kurtosis_prestim}\n')
    print(f'For postimulus amplitude skewness is {skewness_postim}, kurtosis is {kurtosis_postim}')
    prestim_amplitude_list[i] = A_prestim
    postim_amplitude_list[i] = A_postim

# columns=['prestim_amplitude', 'postim_amplitude']
# df  = pd.DataFrame(columns=columns)
# df['prestim_amplitude'] = prestim_amplitude
# df['postim_amplitude'] = postim_amplitude

prestim_amplitude = np.concatenate(prestim_amplitude_list)
postim_amplitude = np.concatenate(postim_amplitude_list)

#%% Create dataframe

#%% Compute some stats

skewness_postim = stats.skew(postim_amplitude)
skewness_prestim = stats.skew(prestim_amplitude)
kurtosis_prestim = stats.kurtosis(prestim_amplitude)
kurtosis_postim = stats.kurtosis(postim_amplitude)
print(f'For prestimulus amplitude skewness is {skewness_prestim}, kurtosis is {kurtosis_prestim}\n')
print(f'For postimulus amplitude skewness is {skewness_postim}, kurtosis is {kurtosis_postim}')

#%% Plot pre stim and postim amplitude distributions
sns.set()
nbins = 150
fig, ax = plt.subplots(1,2)
ax[0].hist(prestim_amplitude, bins=nbins, density=True)
ax[1].hist(postim_amplitude, bins=nbins, density=True)

#%%

# fig, ax = plt.subplots(1,2)
# sns.violinplot(prestim_amplitude, ax=ax[0])
# sns.violinplot(postim_amplitude, ax=ax[1])