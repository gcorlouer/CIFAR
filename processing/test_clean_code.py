#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 09:23:48 2021

@author: guime
"""
# TODO Make numbers of trials in rest and stimuli the same
import HFB_process as hf
import cifar_load_subject as cf
import numpy as np
import mne
#%%

sub_id = 'DiAs'
visual_chan_table = 'visual_channels_BP_montage.csv'
proc = 'preproc' 
sfreq = 250
categories = ['Rest', 'Face', 'Place']
tmin_crop = -0.5
tmax_crop = 1.75
suffix = 'preprocessed_raw'
ext = '.fif'

#%%

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_chan = subject.pick_visual_chan()
HFB_visual = hf.visually_responsive_HFB(sub_id = sub_id)

#%%

# HFB_db = hf.HFB_to_db(HFB, t_pr=-0.5, t_po=1.75, baseline=None,
#                       preload=True, tmin=-0.2, tmax=0)
# events, event_id = mne.events_from_annotations(HFB)
# face_id = hf.extract_stim_id(event_id, cat='Face')
# place_id = hf.extract_stim_id(event_id, cat='Place')
# image_id = face_id+place_id

#%% Partition time.Need to partition sample instead


def event_related_time_to_sample(time, t):
    time = time.tolist()
    time = [round(time[i], 3) for i in range(len(time))]
    sample = time.index(t)
    return sample


def event_related_sample_to_time(time, sample):
    t = time[sample]
    return t


def time_to_sample(t, sfreq=250):
    sample = t*sfreq
    return sample


def sample_to_time(sample, sfreq=250):
    t = sample/sfreq
    return t


def partition_time(time_stamp_min=0, time_stamp_max=1, tau=0.01,
                   window_size=0.150):
    """Create a partition of a time segment into time windows of size
    window_size translated by parameter tau. Units are in seconds. Even stamps
    starts the time window and odd stamps ends it."""
    nwin = 2*round((time_stamp_max - time_stamp_min - window_size)/tau)
    time_stamp = [0]*nwin
    time_stamp[0] = time_stamp_min
    for i in range(nwin):
        if i % 2 == 0:
            time_stamp[i] = time_stamp[0] + i/2*tau
        else:
            time_stamp[i] = time_stamp[0] + (i-1)/2*tau + window_size
    return time_stamp


def partition_sample(sample_min=125, sample_max=375, kappa=3, segment_size=20):
    nseg = 2*round((sample_max - sample_min - segment_size)/kappa)
    sample_stamp = [0]*nseg
    sample_stamp[0] = sample_min
    for i in range(nseg):
        if i % 2 == 0:
            sample_stamp[i] = sample_stamp[0] + i/2*kappa
        else:
            sample_stamp[i] = sample_stamp[0] + (i-1)/2*kappa + segment_size
    return sample_stamp


def epoch_slide(epochs, sample_min=125, sample_max=375, kappa=3, segment_size=20):
    """Return a slided version of epochs to run later sliding window analysis"""
    sample_stamp = partition_sample(sample_min, sample_max, kappa, segment_size)
    nstamp = len(sample_stamp)
    nseg = round(nstamp/2)
    epoch_slide = [0]*nseg
    time_slide = [0]*nseg
    X = epochs.copy().get_data()
    time = epochs.times
    for i in range(nseg):
        seg_start = int(sample_stamp[2*i])
        seg_end = int(sample_stamp[2*i+1])
        epoch_slide[i] = X[:, ..., seg_start:seg_end]
        time_slide[i] = time[seg_start:seg_end]
    X = np.stack(epoch_slide)
    time_slide = np.stack(time_slide)
    return X, time_slide

#%%Category specific HFB slided


ncat = len(categories)
sample_min = 125
sample_max = 375
kappa = 3
segment_size = 20
ts = [0]*ncat
ts_time = [0]*ncat
for idx, cat in enumerate(categories):
    HFB = hf.category_specific_HFB(HFB_visual, cat=cat, tmin_crop = tmin_crop, tmax_crop=tmax_crop)
    HFB = HFB.resample(sfreq=sfreq)
    X, time_slide = epoch_slide(HFB, sample_min, sample_max, kappa,
                                segment_size)
    ts[idx] = X
    ts_time[idx] = time_slide

ts = np.stack(ts)
ts_time = np.stack(time_slide)

# %% Test
