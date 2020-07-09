#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:36:56 2020

@author: guime
"""
import helper
import mne
import scipy.io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as spstats
import statsmodels.stats as stats

pd.options.display.html.table_schema = True
pd.options.display.max_rows = None

from statsmodels.stats.multitest import fdrcorrection, multipletests
from pathlib import Path, PurePath
from numpy import inf

%matplotlib
plt.rcParams.update({'font.size': 17})

#%% 

# Parameters 

# Subject and task
subject = 'DiAs'
subject_id = '04'
proc = 'raw'
montage = 'preproc'
suffix = '_lnrmv.set'
task = 'stimuli'
run = '1'

# High frequency bands 
l_freq = 60
nband = 6 
band_size = 20 

# Epoch parameter

tmin = -0.5 # Prestimulus
tmax = 1.75 # Poststimulus

# Saving paramerters 

save2 = Path('~','projects','CIFAR','data_fun').expanduser()
task_save = 'stimuli'
suffix_place = '_epoch_place.mat'
suffix_face = '_epoch_face.mat'

#%% 
raw, dfelec = helper.import_data(task=task, proc=proc, montage=montage, 
                                 run=run, subject=subject, subject_id=subject_id, suffix=suffix)
HFB, raw_HFB = helper.HFB_raw(raw, l_freq=60, nband=6, band_size=20);
events, event_id = mne.events_from_annotations(raw)
place_id, face_id = helper.stim_id(event_id)
epochs = mne.Epochs(raw_HFB, events, event_id= event_id, tmin=tmin, 
                    tmax=tmax, baseline=None,preload=True)
HFB_db = helper.HFB_norm(epochs, events, tmin)

# %% 
picks = 'LTo4'
conditions = [face_id, place_id]
labels = ['Face ERP', 'Place ERP']
        
helper.plot_stim_response(picks=picks, HFB_db=HFB_db, cdt=face_id, label='Face ERP')
helper.plot_stim_response(picks=picks, HFB_db=HFB_db, cdt=place_id, label='Place ERP')
plt.title(f'{picks} stimulus response')
plt.xlabel('time (s)')
plt.ylabel('Amplitude (dB)')
plt.axhline(y=0)
plt.axvline(x=0)
plt.legend()
# %% 

A_pr = HFB_db.copy().crop(tmin=-0.5, tmax=-0.1).get_data()
A_po = HFB_db.copy().crop(tmin=0.1, tmax=0.5).get_data()
reject, pval_correct, visual_chan, visual_cohen = helper.detect_visual(A_pr, A_po, HFB_db)

#%% Multitrial statistical inference: visual selective

A_pr = HFB_db.copy().crop(tmin=-0.5, tmax=-0.05).get_data()
A_po = HFB_db.copy().crop(tmin=0.05, tmax=0.5).get_data()
M1 = np.mean(A_pr,axis=2)
M2 = np.mean(A_po,axis=2)

from numpy import inf
M1[M1==-inf] = 0
M2[M2 == -inf] = 0

pval = [0]*len(epochs.info['ch_names'])
degf = [0]*len(epochs.info['ch_names'])
tstat = [0]*len(epochs.info['ch_names'])
for i in range(0,len(epochs.info['ch_names'])):
    tstat[i], pval[i] = spstats.wilcoxon(M1[:,i], M2[:,i], zero_method='zsplit')
    
from statsmodels.stats.multitest import fdrcorrection, multipletests
reject, pval_correct = fdrcorrection(pval, alpha=0.05)

# Check channels
idx = np.where(reject==True)
idx = idx[0]
for i in list(idx):
    print(raw.info['ch_names'][i])

# Cohen d 
MC1 = np.mean(M1, axis=0)
MC2 = np.mean(M2, axis=0)
std1 = np.std(M1, axis=0)
std2 = np.std(M2, axis=0)
n1 = M1.shape[1]
n2 = M2.shape[1]
std = np.sqrt(np.divide((n1-1)*std1**2+(n2-1)*std2**2,(n1+n2-2)))
cohen = np.divide(MC1-MC2, std)

idx = np.where(reject==True)
idx = idx[0]
for i in list(idx):
    print(raw.info['ch_names'][i])
    print(np.abs(cohen[i]))

# %% 

def detect_visual(A_pr, A_po, HFB_db):
    
    M1 = np.mean(A_pr,axis=2)
    M2 = np.mean(A_po,axis=2)
    # Get rid of infinity 
    M1[M1==-inf] = 0
    M2[M2 == -inf] = 0
    # Compute inflated p values
    pval = [0]*len(HFB_db.info['ch_names'])
    degf = [0]*len(HFB_db.info['ch_names'])
    tstat = [0]*len(HFB_db.info['ch_names'])
    for i in range(0,len(HFB_db.info['ch_names'])):
        tstat[i], pval[i] = spstats.wilcoxon(M1[:,i], M2[:,i], zero_method='zsplit') # Non normal distrib 
    # Correct multiplt testing    
    reject, pval_correct = fdrcorrection(pval, alpha=0.05)
    
    # Compute effect size: Cohen d 
    MC1 = np.mean(M1, axis=0)
    MC2 = np.mean(M2, axis=0)
    std1 = np.std(M1, axis=0)
    std2 = np.std(M2, axis=0)
    n1 = M1.shape[1]
    n2 = M2.shape[1]
    std = np.sqrt(np.divide((n1-1)*std1**2+(n2-1)*std2**2,(n1+n2-2)))
    cohen = np.divide(MC1-MC2, std)
    # Return visual channels
    idx = np.where(reject==True)
    idx = idx[0]
    visual_chan = []
    visual_cohen = []
    for i in list(idx):
        visual_chan.append(HFB_db.info['ch_names'][i])
        visual_cohen.append(np.abs(cohen[i]))
    return reject, pval_correct, visual_chan
    

#%% Probably deprecated (except cohen d)
evok_po = HFB_db.copy().crop(tmin=0.1, tmax=0.5).average()
evok_pr = HFB_db.copy().crop(tmin=-0.5, tmax=-0.1).average()
A_po = evok_po.data
A_pr = evok_pr.data

# compute cohen d
M1 = np.mean(A_po, axis=1)
M2 = np.mean(A_pr, axis=1)
std1 = np.std(A_po, axis=1)
std2 = np.std(A_pr, axis=1)
n1 = A_po.shape[1]
n2 = A_pr.shape[1]
std = np.sqrt(np.divide((n1-1)*std1**2+(n2-1)*std2**2,(n1+n2-2)))
cohen = np.divide(M1-M2, std)

# Compute number of bins

iqr = spstats.iqr(M1)
n = M1.size 
maximum = np.max(M1)
minimum = np.min(M1)
h = 2*iqr/(n**(1/3))
nbin = (maximum - minimum)/h

from numpy import inf
M1[M1==-inf]=0
M2[M2 == -inf] =0
plt.hist(M2, bins = 80)
plt.xlabel('Mean normalised Amplitude (dB)')
plt.ylabel('Number of electrodes')
plt.title('Prestimulus mean stimulus related HFB amplitude (-400 to -100 ms) ')
plt.hist(M1, bins = 80)
plt.xlabel('Mean normalised Amplitude (dB)')
plt.ylabel('Number of electrodes')
plt.title('Postimulus mean stimulus related HFB amplitude (100 to 400 ms) ')

#%% Histograms 

A_pr = HFB_db.copy().crop(tmin=-0.5, tmax=-0.1).get_data()
A_po = HFB_db.copy().crop(tmin=0.1, tmax=0.5).get_data()
M1 = np.mean(A_pr,axis=2)
M2 = np.mean(A_po,axis=2)

from numpy import inf
M1[M1==-inf]=0
M2[M2 == -inf] =0

plt.hist(M1[:,:], bins = 180)
plt.xlabel('Mean normalised Amplitude (dB)')
plt.ylabel('frequency')
plt.title('Prestimulus mean stimulus related HFB amplitude (-400 to -100 ms) ')

plt.hist(M2, bins = 30)
plt.xlabel('Mean normalised Amplitude (dB)')
plt.ylabel('frequency')
plt.title('Postimulus mean stimulus related HFB amplitude (100 to 400 ms) ')

M1_flat = np.ndarray.flatten(M1)
M2_flat = np.ndarray.flatten(M2)
M_flat = np.concatenate((M1_flat[:, np.newaxis], M2_flat[:, np.newaxis]), axis=1)
g = sns.violinplot(data = M_flat)
g.set(xticklabels = ['Prestimulus', 'Poststimulus'])
g.set_title('Violin plot of mean normalised amplitude in subject DiAs')
g.set_ylabel('frequency')

g = sns.boxplot(data = M_flat)
g.set(xticklabels = ['Prestimulus', 'Poststimulus'])
g.set_title('Box plot of mean normalised amplitude in subject DiAs')
g.set_ylabel('Normalised amplitude (dB)')

g = sns.stripplot(data = M_flat)
g.set(xticklabels = ['Prestimulus', 'Poststimulus'])
g.set_title('Strip plot of mean normalised amplitude in subject DiAs')
g.set_ylabel('Normalised amplitude (dB)')
# %% Statistical inference

A_pr = HFB_db.copy().crop(tmin=-0.5, tmax=-0.1).get_data()
A_po = HFB_db.copy().crop(tmin=0.1, tmax=0.5).get_data()
M1 = np.mean(A_pr,axis=2)
M2 = np.mean(A_po,axis=2)

from numpy import inf
M1[M1==-inf]=0
M2[M2 == -inf] =0
pval = [0]*len(epochs.info['ch_names'])
degf = [0]*len(epochs.info['ch_names'])
tstat = [0]*len(epochs.info['ch_names'])
for i in range(0,len(epochs.info['ch_names'])):
    tstat[i], pval[i] = spstats.wilcoxon(A_po[i,:], A_pr[i,:])

from statsmodels.stats.multitest import fdrcorrection, multipletests
reject, pval_correct = fdrcorrection(pval, alpha=0.01)
# %% Rank post stimulus activity

SE = spstats.sem(A_po, axis=1)

M = np.sort(M1)
x=np.arange(0,len(list(M1))).tolist()
plt.bar(x,list(M))
plt.xlabel('Rank (from least to most resposive)')
plt.ylabel('Mean amplitude response (dB)')
plt.title('Mean amplitude response sorted from lowest to highest response')