#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:04:25 2021

@author: guime
"""

import HFB_process as hf
import mne
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path, PurePath
from config import args

ichan=6

#%% Load data

subject = hf.Subject()
raw = subject.load_data(proc= 'raw_signal', stage= args.stage,
                  preload=True, epoch=args.epoch)

#%% Test narrow band envelope extraction

fpath = subject.processing_stage_path(proc = args.proc)
fname = args.sub_id + args.stage
fpath = fpath.joinpath(fname)
raw = mne.io.read_raw_fif(fpath, preload=True)

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

#%% Test hfb extraction

A = hf.Hfb()
hfb = A.extract_hfb(raw)
plt.plot(times, hfb[ichan, :][0][0], label='HFB amplitude')

#%% Test epoching

subject = hf.Subject()
hfb = subject.load_data(proc=args.proc, stage=args.stage, epoch=False)
A = hf.Hfb_db(t_prestim=args.t_prestim, t_postim=args.t_postim, baseline=args.baseline,
                 preload=args.preload, tmin=args.tmin_baseline, tmax=args.tmax_baseline,
                 mode=args.mode)
hfb_db = A.raw_to_hfb_db(hfb)
epochs_pick = hfb_db.copy().pick_channels(['LTo1-LTo2'])
epochs_pick.plot_image()

#%% Test visual channel detection

dv = hf.Detect_visual_site(tmin_prestim=args.tmin_prestim, tmax_prestim=args.tmax_prestim,
                           tmin_postim=args.tmin_postim, tmax_postim=args.tmax_postim, 
                           alpha=args.alpha, zero_method=args.zero_method, alternative=args.alternative)

visual_chan, effect_size = dv.detect(hfb_db)

#%% Test visual channel classification

dfelec = subject.df_electrodes_info()
cv = hf.Classify_visual_site(tmin_prestim=args.tmin_prestim, tmax_prestim=args.tmax_prestim,
                           tmin_postim=args.tmin_postim, tmax_postim=args.tmax_postim, 
                           alpha=args.alpha, zero_method=args.zero_method, alternative=args.alternative)

visual_populations = cv.hfb_to_visual_populations(hfb_db, dfelec)

#%% Test category time series

visual_populations = subject.pick_visual_chan()
hfb, visual_chan = subject.load_visual_hfb(proc= args.proc, 
                            stage= args.stage)

epochs = hf.category_hfb(hfb, cat='Face', tmin_crop = 0.5, tmax_crop=1.5)
