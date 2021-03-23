#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 10:32:55 2021

@author: guime
"""

import HFB_process as hf
import cifar_load_subject as cf
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from pathlib import Path, PurePath


#%% Read preprocessed data

path = cf.cifar_ieeg_path()
fname = 'cross_subjects_visual_BP_channels.csv'
fpath = path.joinpath(fname)
visual_table = pd.read_csv(fpath)
index_names = visual_table[visual_table['latency']==0].index
visual_table = visual_table.drop(index_names)
#%% 

latency = visual_table['latency']
category_selectivity = visual_table['category_selectivity']
visual_response = visual_table['visual_responsivity']
Y_coord = visual_table['Y']
X_coord = visual_table['X']
Z_coord = visual_table['Z']
peak_time = visual_table['peak_time']
#%% Compute linear regression

def regression_line(x, y):
    linreg = stats.linregress(x, y)
    a = linreg.slope
    b = linreg.intercept
    Y = a*x+b 
    print(linreg)
    return Y

peak_time_hat = regression_line(latency, peak_time)
Y_coord_hat = regression_line(latency, Y_coord)
Z_coord_hat = regression_line(latency, Z_coord)
visual_response_hat = regression_line(latency, visual_response)
category_selectivity_hat = regression_line(latency, category_selectivity)
#%% Correlation

corr = stats.pearsonr(latency, Y_coord)

# %% Plot linear regression
sns.set(font_scale=2)

#%matplotlib 
# plt.rcParams.update({'font.size': 22})

plt.subplot(2,2,1)
plt.plot(latency, Y_coord, '.')
plt.plot(latency,Y_coord_hat)
plt.ylabel('Y coordinate (mm)')

plt.subplot(2,2,2)
plt.plot(latency, Z_coord, '.')
plt.plot(latency, Z_coord_hat)
plt.ylabel('Z coordinate (mm)')


plt.subplot(2,2,3)
plt.plot(latency, visual_response, '.')
plt.plot(latency, visual_response_hat)
plt.xlabel('latency response (ms)')
plt.ylabel(' visual responsivity (db)')

plt.subplot(2,2,4)
plt.plot(latency, peak_time, '.')
plt.plot(latency, peak_time_hat)
plt.xlabel('latency response (ms)')
plt.ylabel('peak time (ms)')

#%%

#visual_table_sorted = visual_table.sort_values(by='Y')

#%%





