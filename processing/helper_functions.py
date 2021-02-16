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

