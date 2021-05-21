
"""
Created on Wed MIeb 24 16:24:05 2021

@author: guime
"""


import cifar_load_subject as cf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import helper_functions as fun
import HFB_process as hf

from scipy.io import loadmat

#%%

sub_id = 'DiAs'
visual_chan_table = 'visual_channels_BP_montage.csv'
proc = 'preproc' 
sfreq = 250;
suffix = 'preprocessed_raw'
ext = '.fif'
categories = ['Rest', 'Face', 'Place']

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
fname = sub_id + 'MI.mat'
fpath = datadir.joinpath(fname)
visual_chan = subject.pick_visual_chan()
MI_data = loadmat(fpath)

#%%

MI = MI_data['I']
sig = MI_data['sig']

#%% Sort indices

MI_sorted = np.zeros_like(MI)
sig_sorted = np.zeros_like(MI)
ch_idx_sorted = visual_chan.index.tolist()
for isort, i in enumerate(ch_idx_sorted):
    for jsort, j in enumerate(ch_idx_sorted):
        MI_sorted[isort,jsort,:] = MI[i, j, :]
        sig_sorted[isort,jsort,:] = sig[i, j, :]
#%% Plot pairwise MI
plt.rcParams['font.size'] = '19'

ncat = MI.shape[2]

MI = 1/np.log(2) * MI
MI = np.nan_to_num(MI, copy=True, nan=0.0)
MI_max = np.max(MI)

for icat in range(ncat):
    populations = visual_chan['group']
    plt.subplot(2,2, icat+1)
    sns.heatmap(MI[:,:,icat], vmin=0, vmax=MI_max, xticklabels=populations,
                    yticklabels=populations, cmap='YlOrBr')
    
    for y in range(MI.shape[0]):
        for x in range(MI.shape[1]):
            if sig_sorted[y,x,icat] == 1:
                plt.text(x + 0.5, y + 0.5, '*',
                         horizontalalignment='center', verticalalignment='center',
                         color='k')
            else:
                continue
    plt.title('MI ' + categories[icat] + ' (bits)')
