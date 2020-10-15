#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:09:57 2020

@author: guime
"""


import mne 
import numpy as np
import re
import scipy.stats as spstats
import matplotlib.pyplot as plt

from numpy import inf
from statsmodels.stats.multitest import fdrcorrection, multipletests

# TODO: Find all place and face selective elctrodes and plot grand average response.
# And split in 2 scripts
def freq_bands(l_freq=60, nband=6, band_size=20):
    bands = [l_freq+i*band_size for i in range(0, nband)]
    return bands

def extract_envelope(raw, l_freq=60, band_size=20):
    raw_band = raw.copy().filter(l_freq=l_freq, h_freq=l_freq+band_size, 
                                 phase='minimum', filter_length='auto',
                                 l_trans_bandwidth= 10, h_trans_bandwidth= 10, 
                                     fir_window='blackman')
    envelope = raw_band.copy().apply_hilbert(envelope=True).get_data()
    return envelope 

def mean_normalise(envelope):
    envelope_mean = np.mean(envelope, axis=1)
    envelope_norm = np.divide(envelope, envelope_mean[:,np.newaxis])
    return envelope_norm 

def extract_HFB(raw, bands):
    
    nobs = len(raw.times)
    nchan = len(raw.info['ch_names'])
    nband = len(bands)
    HFB = np.zeros(shape=(nchan, nobs))
    mean_amplitude = np.zeros(shape=(nchan,))
    band_size= bands[1]-bands[0]
    
    for band in bands:
        envelope = extract_envelope(raw, l_freq=band, band_size=band_size)
        env_norm = mean_normalise(envelope)
        HFB = HFB + env_norm
        mean_amplitude = mean_amplitude + np.mean(envelope, axis=1)

    HFB = HFB/nband # average over all bands
    HFB = HFB * mean_amplitude[:,np.newaxis] # bring back to volts
    HFB = np.nan_to_num(HFB) # replace nan with zeros
    HFB = mne.io.RawArray(HFB, raw.info)
    return HFB

def extract_stim_id(event_id, cat = 'Face'):
    p = re.compile(cat)
    stim_id = []
    for key in event_id.keys():
        if p.match(key):
            stim_id.append(key)
    return stim_id 

def epoch_HFB(HFB, raw, t_pr = -0.5, t_po = 1.75, baseline=None, preload=True):
    events, event_id = mne.events_from_annotations(raw) 
    epochs = mne.Epochs(HFB, events, event_id= event_id, tmin=t_pr, 
                    tmax=t_po, baseline=baseline,preload=preload)
    return epochs

def extract_baseline(epochs, tmin=-0.4, tmax=-0.1):
    baseline = epochs.copy().crop(tmin=tmin, tmax=tmax) # Extract prestimulus baseline
    baseline = baseline.get_data()
    baseline = spstats.mstats.gmean(baseline,axis=2)
    return baseline 

def baseline_normalisation(epochs, tmin=-0.4, tmax=-0.1):
    baseline = extract_baseline(epochs, tmin=tmin, tmax=tmax)
    A = epochs.get_data()
    A_norm = np.zeros_like(A)
    for i in range(len(epochs)):
        for j in range(len(epochs.info['ch_names'])):
            A_norm[i,j,:] = np.divide(A[i,j,:], baseline[i,j])# divide by baseline
        A_norm = np.nan_to_num(A_norm)
    return A_norm 

def dB_transform(A_norm, raw, t_pr=-0.5):
    events, event_id = mne.events_from_annotations(raw)
    A_db =  np.zeros_like(A_norm) 
    for i in range(np.size(A_db,0)):
        for j in range(np.size(A_db,1)):
            A_db[i,j,:] = 10*np.log10(A_norm[i,j,:]) # transform into normal distribution
        HFB_db = np.nan_to_num(A_db)
    del event_id['boundary']
    HFB_db = mne.EpochsArray(HFB_db, raw.info, events=events[1:], 
                             event_id=event_id, tmin=t_pr) # Drop boundary event (otherwise event size don't match)
    return HFB_db 

def log_transform(epochs, picks):
    # transform into log normal distribution, should also work with raw structure
    data = epochs.copy().pick(picks=picks).get_data()
    log_HFB = np.log(data)
    return log_HFB

def extract_db(epochs, raw, tmin=-0.4, tmax=-0.1, t_pr=-0.5):
    A_norm = baseline_normalisation(epochs, tmin=-0.4, tmax=-0.1)
    HFB_db = dB_transform(A_norm, raw,  t_pr)
    return HFB_db

def extract_HFB_db(raw, bands, t_pr = -0.5, t_po = 1.75, baseline=None,
                       preload=True, tmin=-0.4, tmax=-0.1):
    HFB = extract_HFB(raw, bands)
    events, event_id = mne.events_from_annotations(raw)
    epochs = epoch_HFB(HFB, raw, t_pr = t_pr, t_po = t_po, baseline=baseline,
                       preload=preload)
    HFB_db = extract_db(epochs, raw, tmin=tmin, tmax=tmax, t_pr=t_pr)
    return HFB_db

def plot_stim_response(HFB_db, stim_id, picks='LTo4'):
        evok = HFB_db[stim_id].copy().pick(picks).average()
        evok_std = HFB_db[stim_id].copy().pick(picks).standard_error()
        ERP = evok.data
        ERP_std = evok_std.data
        time = HFB_db.times
        plt.plot(time, ERP[0,:])
        plt.fill_between(time, ERP[0,:]-1.96*ERP_std[0,:], ERP[0,:]+1.96*ERP_std[0,:],
                         alpha=0.3)

#%% Detect visual electrodes

def cf_mean(A):
    M = np.mean(A,axis=2)
    # Get rid of infinity 
    M[M==-inf] = 0
    return M

def crop_HFB(HFB_db, tmin=-0.5, tmax=-0.05):
    A = HFB_db.copy().crop(tmin=tmin, tmax=tmax).get_data()
    return A

def crop_stim_HFB(HFB_db, stim_id, tmin=-0.5, tmax=-0.05):
    A = HFB_db[stim_id].copy().crop(tmin=tmin, tmax=tmax).get_data()
    return A

def stats_visual(A_pr, A_po, HFB_db, alpha=0.01):
    # maybe HFB_db variablenot necessary
    """Wilcoxon test for visual responsivity"""
    M1 = cf_mean(A_pr)
    M2 = cf_mean(A_po)
    # Iniitialise inflated p values
    pval = [0]*len(HFB_db.info['ch_names'])
    tstat = [0]*len(HFB_db.info['ch_names'])
    # Compute inflated stats
    for i in range(0,len(HFB_db.info['ch_names'])):
        tstat[i], pval[i] = spstats.wilcoxon(M1[:,i], M2[:,i], zero_method='zsplit') # Non normal distrib 
    # Correct for multiple testing    
    reject, pval_correct = fdrcorrection(pval, alpha=alpha)
    return reject, pval_correct


def cf_cohen(M1, M2):
    """Compute effect size: Cohen d """
    MC1 = np.mean(M1, axis=0)
    MC2 = np.mean(M2, axis=0)
    std1 = np.std(M1, axis=0)
    std2 = np.std(M2, axis=0)
    n1 = M1.shape[1]
    n2 = M2.shape[1]
    std = np.sqrt(np.divide((n1-1)*std1**2+(n2-1)*std2**2,(n1+n2-2)))
    cohen = np.divide(MC1-MC2, std)
    return cohen 

def return_visual_chan(reject, HFB_db, cohen):
    """Return statistically significant visual channels with effect size"""
    idx = np.where(reject==True)
    idx = idx[0]
    visual_chan = []
    visual_cohen = []
    for i in list(idx):
        visual_chan.append(HFB_db.info['ch_names'][i])
        visual_cohen.append(np.abs(cohen[i]))
    return visual_chan, visual_cohen

def detect_visual_chan(HFB_db, tmin_pr=-0.5, tmax_pr=-0.1, tmin_po=0.1, tmax_po=0.5):
    """Return statistically significant visual channels with effect size"""
    A_pr = crop_HFB(HFB_db, tmin=tmin_pr, tmax=tmax_pr)
    A_po = crop_HFB(HFB_db, tmin=tmin_po, tmax=tmax_po)
    M1 = cf_mean(A_pr)
    M2 = cf_mean(A_po)
    reject, pval_correct = stats_visual(A_pr, A_po, HFB_db, alpha=0.01)
    cohen = cf_cohen(M1, M2)
    visual_chan, visual_cohen = return_visual_chan(reject, HFB_db, cohen)
    return visual_chan, visual_cohen

def detect_stim_chan(HFB_db, stim_id, tmin_pr=-0.5, tmax_pr=-0.1, tmin_po=0.1, tmax_po=0.5):
    """Return statistically significant stim channels with effect size"""
    A_pr = crop_stim_HFB(HFB_db, stim_id, tmin=tmin_pr, tmax=tmax_pr)
    A_po = crop_stim_HFB(HFB_db, stim_id, tmin=tmin_po, tmax=tmax_po)
    M1 = cf_mean(A_pr)
    M2 = cf_mean(A_po)
    reject, pval_correct = stats_visual(A_pr, A_po, HFB_db, alpha=0.01)
    cohen = cf_cohen(M1, M2)
    visual_chan, visual_cohen = return_visual_chan(reject, HFB_db, cohen)
    return visual_chan, visual_cohen

def detect_pure_face(place_chan, face_chan):
    pure_face = list(set(face_chan)-set(place_chan))
    return pure_face

def detect_pure_place(place_chan, face_chan):
    pure_place = list(set(place_chan)-set(face_chan))
    return pure_place

def detect_bicat(visual_chan, face_chan, place_chan):
    pure_face = detect_pure_face(place_chan, face_chan)
    pure_place = detect_pure_place(place_chan, face_chan)
    place = list(set(visual_chan)-set(pure_face))
    bicat = list(set(place)-set(pure_place))
    return bicat

def HFB2face(HFB_db, face_id, place_id):
    face_chan, face_cohen = detect_stim_chan(HFB_db, face_id)
    place_chan, place_cohen = detect_stim_chan(HFB_db, place_id)
    pure_face = detect_pure_face(place_chan, face_chan)
    return pure_face

def HFB2place(HFB_db, face_id, place_id):
    face_chan, face_cohen = detect_stim_chan(HFB_db, face_id)
    place_chan, place_cohen = detect_stim_chan(HFB_db, place_id)
    pure_place = detect_pure_place(place_chan, face_chan)
    return pure_place

def HFB2bicat(HFB_db, face_id, place_id):
    face_chan, face_cohen = detect_stim_chan(HFB_db, face_id)
    place_chan, place_cohen = detect_stim_chan(HFB_db, place_id)
    visual_chan, visual_cohen = detect_visual_chan(HFB_db)
    bicat = detect_bicat(visual_chan, face_chan, place_chan)
    return bicat

def make_visual_cat(HFB_db, face_id, place_id):
    face = HFB2face(HFB_db, face_id, place_id)
    place = HFB2place(HFB_db, face_id, place_id)
    bicat = HFB2bicat(HFB_db, face_id, place_id)
    visual_cat = {'Face' : face, 'Place': place, 'Bicat': bicat}
    return visual_cat

def functional_grouping(subject, visual_cat):
    functional_group = {'subject_id': [], 'chan_name': [], 'category': [], 'brodman': []}
    for key in visual_cat.keys():
        cat = visual_cat[key]
        functional_group['subject_id'].extend([subject.name]*len(cat))
        functional_group['chan_name'] = cat
        functional_group['category'].extend([key]*len(cat))
        functional_group['brodman'].extend(subject.ROIs(cat))
        functional_group['DK'].extend(subject.ROI_DK(cat))
    return functional_group

def epoch(HFB, raw, task='stimuli',
                            cat='Face', duration=5, t_pr = -0.1, t_po = 1.75):
    # TODO : must depend on wether raw or bipolar
    if task=='stimuli':
        events, event_id = mne.events_from_annotations(raw)
        cat_id = extract_stim_id(event_id, cat = cat)
        epochs = epoch_HFB(HFB, raw, t_pr = t_pr, t_po = t_po)
        epochs = epochs[cat_id].copy()
    else:
        events = mne.make_fixed_length_events(raw, start=10, stop=200, duration=duration)
        epochs = mne.Epochs(HFB, events, tmin = t_pr, tmax = t_po, 
                            baseline=None, preload=True)
    return epochs   

def make_visual_chan_dictionary(df_visual, raw, HFB, epochs, sub='DiAs'): 
   # Return visual channels in dictionary to save in matfile 
    events, event_id = mne.events_from_annotations(raw)
    visual_chan = list(df_visual['chan_name'].loc[df_visual['subject_id']== sub])
    category = list(df_visual['category'].loc[df_visual['subject_id']== sub])
    brodman = list(df_visual['brodman'].loc[df_visual['subject_id']== sub])
    DK = list(df_visual['DK'].loc[df_visual['subject_id']== sub] )
    ts = log_transform(HFB, picks=visual_chan)
    multitrial_ts = log_transform(epochs, picks=visual_chan) # make data normal
    # ch_idx = mne.pick_channels(epochs.info['ch_names'], include=visual_chan)
    visual_dict = dict(ts=ts, multitrial_ts=multitrial_ts, chan=visual_chan, 
                   category=category, brodman=brodman, DK = DK, events=events,
                   event_id = event_id)
    return visual_dict 




