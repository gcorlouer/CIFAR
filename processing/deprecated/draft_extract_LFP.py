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

from scipy.io import loadmat,savemat

# %matplotlib
#%% TODO

# -Check that there are output visual_data X is correct with HFB_visual (i.e. check that 
# permutation works)
# - Create a module for category specific electrodes
# - Rearrange HFB module consequently

#%% 
pd.options.display.max_rows = 999

sub_id = 'DiAs'

proc = 'bipolar_montage' 
sfreq = 250;
picks = ['LGRD60-LGRD61', 'LTo1-LTo2']
tmin_crop = 0.5
tmax_crop = 1.5
suffix = 'BP_montage'
ext = '.set'

#%%

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_chan = subject.low_high_chan()
visual_chan_name = visual_chan['chan_name'].tolist()

raw = subject.read_eeglab(proc= proc, suffix=suffix)

raw_visual = raw.copy().pick(visual_chan_name)

#%% Plot


raw_visual.plot(duration = 0.1, scalings=1e-4)

#%% 

epochs = hf.epoch_category(raw_visual, cat='Face', tmin=-0.5, tmax=1.75)

#%% Make dictionary with all state

categories = ['Rest', 'Face', 'Place']
columns = ['Rest', 'Face', 'Place','populations']
visual_time_series = {'Rest': [], 'Face': [], 'Place': [], 'populations': []}

for cat in categories:
    visual_data = hf.HFB_to_visual_data(HFB, visual_chan, cat=cat, 
                                    tmin_crop = tmin_crop, tmax_crop=tmax_crop)
    X = visual_data['data']
    visual_time_series[cat] = X
    
visual_time_series['populations'] = visual_data['populations']

#%% Test functions
subject = cf.Subject(name=sub_id)
visual_chan = subject.low_high_chan()
group = visual_chan['group'].unique().tolist()
    
HFB = hf.low_high_HFB(visual_chan)


HFB_cat = hf.category_specific_HFB(HFB,cat= cat,
                                   tmin_crop = tmin_crop, tmax_crop=tmax_crop)


# Epoch 


#%% 

picks = ['LTo1-LTo2','LGRD60-LGRD61']

evok = HFB_cat.copy().pick_channels(picks).average()

evok.plot()

# %% Pick epoch for 2 channels

picks = ['LTo1-LTo2','LGRD60-LGRD61']

epoch_data = HFB_cat.copy().pick_channels(picks)

X = epoch_data.get_data()
X = np.transpose(X, axes = (1,2,0))
test_data = dict(data=X)

fpath = subject.processing_stage_path(proc = proc)
fname = sub_id + '_' + cat + '_test.mat'
fpath = fpath.joinpath(fname)
savemat(fpath , test_data)

