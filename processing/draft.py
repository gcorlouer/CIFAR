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

proc = 'preproc' 
sfreq = 500; 
cat = 'Face'
tmin_crop = 0.30
tmax_crop = 1.5
suffix = 'preprocessed_raw'
ext = '.fif'


#%% Test functions
subject = cf.Subject(name=sub_id)
visual_chan = subject.low_high_chan()
group = visual_chan['group'].unique().tolist()

HFB = hf.low_high_HFB()

HFB_cat = hf.category_specific_HFB(HFB, group, visual_chan, cat= cat,
                                   tmin_crop = tmin_crop, tmax_crop=tmax_crop)
# Epoch 

visual_data = hf.HFB_to_visual_data(HFB, visual_chan)
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

