#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 16:19:34 2021

@author: guime
"""



import HFB_process as hf
import cifar_load_subject as cf
import numpy as np
import mne
import matplotlib.pyplot as plt

from pathlib import Path, PurePath

from mne.time_frequency import psd_array_multitaper
#%% TODO 
# add alpha beta gamma bands 
# Suggestion : concatenate all visually responsive channels for HFB analysis
#%% Parameters

# sub_id = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
sub_id = 'DiAs'
proc = 'preproc'
sfreq = 250
adaptive = True
duration = 60
tmax = duration
sfreq = 250
fmin = 0.1
fmax = 125
bandwidth = 0.5
font = {'size':20}
#%% Load continuous HFB 

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
visual_chan = subject.pick_visual_chan()
# visual_chan = hf.pick_visual_chan(picks, visual_chan)
HFB = hf.visually_responsive_HFB(sub_id=sub_id)

#%% Plot HFB

# %matplotlib qt

# events, event_id = mne.events_from_annotations(HFB)
# HFB.plot(duration=200, scalings=1e-4)

#%% Helper functions


def HFB_to_psd(HFB, start=500, stop=None, duration=20, tmin=-0.1, tmax=20,
               preload=True, baseline=None, fmin=0.1, fmax=20,
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

def plot_psd(psd, freqs, average=True, label='PSD Rest'):
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
            plt.xscale('linear')
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

#%%


psd, freqs = HFB_to_psd(HFB, start=50, stop=300, tmax=tmax, duration=duration,
                        bandwidth=bandwidth, fmax=fmax, sfreq=sfreq)
plot_psd(psd, freqs, average=True, label='Rest')
psd, freqs = HFB_to_psd(HFB, start=450, stop=None, tmax=tmax, duration=duration,
                        bandwidth=bandwidth, fmax=fmax, sfreq=sfreq)
plot_psd(psd, freqs, average=True, label='Stimuli')
