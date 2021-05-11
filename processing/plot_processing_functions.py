#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:04:25 2021

@author: guime
"""

import HFB_process as hf
import cifar_load_subject as cf
import mne
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path, PurePath
from config import args

ichan=6

#%% Load data

subject = hf.Subject()
fpath = subject.processing_stage_path(proc = args.proc)
fname = args.sub_id + args.stage
fpath = fpath.joinpath(fname)
raw = mne.io.read_raw_fif(fpath, preload=True)
#%%
raw = raw.crop(tmin=100, tmax=102)
times = raw.times
raw_filt = raw.copy().filter(l_freq=args.l_freq, h_freq=80,
                                 phase=args.phase, filter_length=args.filter_length,
                                 l_trans_bandwidth= args.l_trans_bandwidth, 
                                 h_trans_bandwidth= args.h_trans_bandwidth,
                                     fir_window=args.fir_window)
lfp_filt = raw_filt.copy().get_data()*1e6
hfb = hf.Hfb()
hfb = hfb.extract_envelope(raw)
hfb = hfb*1e6


matplotlib.rcParams.update({'font.size': 18})
sns.set()

plt.plot(times, hfb[ichan, :], label='HFB amplitude')
plt.plot(times, lfp_filt[ichan, :], label='LFP')



