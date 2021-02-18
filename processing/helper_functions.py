#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:35:51 2021

@author: guime
"""
import HFB_process as hf
import numpy as np
import mne
import matplotlib.pyplot as plt

from mne.time_frequency import psd_array_multitaper


#%% PSD utilities


def HFB_to_psd(HFB, start=500, stop=None, duration=20, tmin=-0.1, tmax=20,
               preload=True, baseline=None, fmin=0.1, fmax=20, adaptive=True,
               bandwidth=0.5, sfreq=500):
    events = mne.make_fixed_length_events(HFB, start=start, stop=stop, duration=duration)
    epochs = mne.Epochs(HFB, events, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True)
    X = epochs.copy().get_data()
    (n_trials, n_chans, n_times) = X.shape
    psd, freqs = psd_array_multitaper(X, sfreq, fmin=fmin, fmax=fmax,
                                                 bandwidth=bandwidth, adaptive=adaptive)
    psd = np.mean(psd, axis=0)  # average over channels
    return psd, freqs

def plot_psd(psd, freqs, average=True, label='PSD Rest', font = {'size':20}):
    (nchan, nfreq) = psd.shape
    psd = np.log(psd)
    bands = [4, 8, 16]
    bands_name = [r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$']
    xbands = [2, 6, 12, 18]
    ybands = [-19]*4
    if average is True:
        psd = np.mean(psd, axis=0)
        plt.plot(freqs, psd, label=label)
        plt.xscale('log')
        plt.xlabel('Frequency (Hz)', fontdict=font)
        plt.ylabel('Power (dB)', fontdict=font)
        for i in range(len(bands)):
            plt.axvline(x=bands[i], color='k')
        for i in range(len(xbands)):
            plt.text(xbands[i]+1, ybands[i], bands_name[i], fontdict=font)
    else:
        for i in range(nchan):
            plt.plot(freqs, psd[i, :])
            plt.xscale('log')
            plt.xlabel('Frequency (Hz)', fontdict=font)
            plt.ylabel('Power (dB)', fontdict=font)
            for i in range(len(bands)):
                plt.axvline(x=bands[i], color='k')
            for i in range(len(xbands)):
                plt.text(xbands[i]+1, ybands[i], bands_name[i], fontdict=font)
    plt.title('Power spectral density of visually responsive HFB envelope', fontdict=font)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.text(5, -19, 'lol')
    plt.legend()

# %% LFP utilities


def LFP_to_dict(LFP, visual_chan, tmin=-0.5, tmax=1.75, sfreq=250):
    """Return dictionary with all category specific LFP and visual channels
    information"""
    visual_LFP = LFP.copy().pick(visual_chan['chan_name'].tolist())
    LFP_dict = visual_chan.to_dict(orient='list')
    categories = ['Rest', 'Face', 'Place']
    for cat in categories:
        epochs, events = hf.epoch_category(visual_LFP, cat=cat, tmin=-0.5, tmax=1.75)
        epochs = epochs.resample(sfreq)
        X = epochs.copy().get_data()
        X = np.transpose(X, axes=(1, 2, 0))
        LFP_dict[cat] = X
    LFP_dict['time'] = epochs.times
    population_to_channel = hf.parcellation_to_indices(visual_chan, parcellation='group')
    DK_to_channel = hf.parcellation_to_indices(visual_chan, parcellation='DK')
    LFP_dict['population_to_channel'] = population_to_channel
    LFP_dict['DK_to_channel'] = DK_to_channel
    return LFP_dict

# %% Sliding window analysis

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


def slide_window(X, time, start=125, stop=375, step=5, window_size=20):
    nobs = stop - start + 1
    nwin = round((nobs - 1 - window_size)/step)
    win_start = [0]*nwin
    win_end = [0]*nwin
    X_win = [0]*nwin
    time_win = [0]*nwin
    for k in range(nwin):
        win_start[k] = k*step + start
        win_end[k] = win_start[k] + window_size
        X_win[k] = X[:, ..., win_start[k]:win_end[k]]
        time_win[k] = time[win_start[k]:win_end[k]]
    X_win = np.stack(X_win)
    time_win = np.stack(time_win)
    return X_win, time_win

def epoch_win(epochs, sample_start=125, sample_stop = 375, step=5, window_size=20):
    X = epochs.copy().get_data()
    time = epochs.times
    X, time_win = slide_window(X, time, sample_start, sample_stop, step, window_size)
    return X, time_win

def ts_win_cat(HFB_visual, visual_chan, categories=['Rest', 'Face', 'Place'], tmin_crop=-0.5,
               tmax_crop=1.75, sfreq=250, sample_start=125, sample_stop = 375, step=5, 
               window_size=20):
    ncat = len(categories)
    ts = [0]*ncat
    ts_time = [0]*ncat
    for idx, cat in enumerate(categories):
        HFB = hf.category_specific_HFB(HFB_visual, cat=cat, tmin_crop = tmin_crop,
                                       tmax_crop=tmax_crop)
        HFB = HFB.resample(sfreq=sfreq)
        X, time_win = epoch_win(HFB, sample_start, sample_stop, step, window_size)
        # X_ordered = np.zeros_like(X)
        # sorted_ch_indices = visual_chan.index.tolist()
        # for ichan, i in enumerate(sorted_ch_indices):
        #     X_ordered[:, ichan, :] = X[:, i, :]
        #     X = X_ordered
        ts[idx] = X

    ts = np.stack(ts)
    (ncat, nwin, ntrial, nchan, nobs) = ts.shape
    ts = np.transpose(ts, (3, 4, 2, 1, 0))
    time_win = np.transpose(time_win, (1, 0))
    return ts, time_win

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

def category_slided_ts(HFB_visual, visual_chan, categories=['Rest', 'Face', 'Place'], tmin_crop=-0.5,
                       tmax_crop=1.75, sfreq=250, sample_min=125, sample_max=375,
                       kappa=3, segment_size=20):
    ncat = len(categories)
    ts = [0]*ncat
    ts_time = [0]*ncat
    for idx, cat in enumerate(categories):
        HFB = hf.category_specific_HFB(HFB_visual, cat=cat, tmin_crop = tmin_crop,
                                       tmax_crop=tmax_crop)
        HFB = HFB.resample(sfreq=sfreq)
        X, time_slide = epoch_slide(HFB, sample_min, sample_max, kappa,
                                    segment_size)
        X_ordered = np.zeros_like(X)
        sorted_ch_indices = visual_chan.index.tolist()
        for ichan, i in enumerate(sorted_ch_indices):
            X_ordered[:, ichan, :] = X[:, i, :]
            X = X_ordered
        ts[idx] = X
        ts_time[idx] = time_slide

    ts = np.stack(ts)
    ts_time = np.stack(time_slide)
    (ncat, nseg, ntrial, nchan, nobs) = ts.shape
    ts = np.transpose(ts, (3, 4, 2, 1, 0))
    ts_time = np.transpose(ts_time, (1, 0))
    return ts, ts_time
