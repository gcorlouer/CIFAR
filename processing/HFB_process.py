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
import cifar_load_subject as cf

from numpy import inf
from statsmodels.stats.multitest import fdrcorrection, multipletests

#%% Extract and epoch high frequency band envelope

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
    HFB = HFB * mean_amplitude[:,np.newaxis] # multiply by mean amplitude to bring back to volts
    HFB = np.nan_to_num(HFB) # replace nan with zeros
    HFB = mne.io.RawArray(HFB, raw.info)
    HFB.set_annotations(raw.annotations)
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
    baseline = np.mean(baseline,axis=(0,2)) # average over prestimulus and trials
    return baseline 


def db_transform(epochs, raw, tmin=-0.4, tmax=-0.1, t_pr=-0.5):
    events, event_id = mne.events_from_annotations(raw)
    baseline = extract_baseline(epochs, tmin=tmin, tmax=tmax)
    A = epochs.get_data()
    A = np.divide(A, baseline[np.newaxis,:,np.newaxis]) # divide by baseline
    A = np.nan_to_num(A)
    A = 10*np.log10(A) # convert to db
    #del event_id['boundary']
    HFB = mne.EpochsArray(A, raw.info, events=events[1:], 
                             event_id=event_id, tmin=t_pr) # Drop boundary event
    return HFB


def log_transform(epochs, picks):
    # transform into log normal distribution, should also work with raw structure
    data = epochs.copy().pick(picks=picks).get_data()
    log_HFB = np.log(data)
    return log_HFB

def raw_to_HFB_db(raw, bands, t_pr = -0.5, t_po = 1.75, baseline=None,
                       preload=True, tmin=-0.4, tmax=-0.1):
    HFB = extract_HFB(raw, bands)
    events, event_id = mne.events_from_annotations(raw)
    epochs = epoch_HFB(HFB, raw, t_pr = t_pr, t_po = t_po, baseline=baseline,
                       preload=preload)
    HFB_db = db_transform(epochs, raw, tmin=tmin, tmax=tmax, t_pr=t_pr)
    return HFB_db

def plot_HFB_response(HFB_db, stim_id, picks='LTo4'):
        evok = HFB_db[stim_id].copy().pick(picks).average()
        evok_std = HFB_db[stim_id].copy().pick(picks).standard_error()
        ERP = evok.data
        ERP_std = evok_std.data
        time = HFB_db.times
        plt.plot(time, ERP[0,:])
        plt.fill_between(time, ERP[0,:]-1.96*ERP_std[0,:], ERP[0,:]+1.96*ERP_std[0,:],
                         alpha=0.3)


def epoch(HFB, raw, task='stimuli',
                            cat='Face', duration=5, t_pr = -0.1, t_po = 1.75):
    """Epoch HFB depending task"""
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

#%% Classify visually responsive populations

def sample_mean(A):
    M = np.mean(A,axis=-1) # average over sample
    # Get rid of infinity 
    M[M==-inf] = 0
    return M

def crop_HFB(HFB_db, tmin=-0.5, tmax=-0.05):
    A = HFB_db.copy().crop(tmin=tmin, tmax=tmax).get_data()
    return A

def crop_stim_HFB(HFB_db, stim_id, tmin=-0.5, tmax=-0.05):
    A = HFB_db[stim_id].copy().crop(tmin=tmin, tmax=tmax).get_data()
    return A

def multiple_wilcoxon_test(A_po, A_pr, nchans, alpha=0.05):
    # maybe HFB_db variablenot necessary
    """Wilcoxon test for visual responsivity"""
    A_po = sample_mean(A_po)
    A_pr = sample_mean(A_pr)
    # Iniitialise inflated p values
    pval = [0]*nchans
    tstat = [0]*nchans
    # Compute inflated stats
    for i in range(0,nchans):
        tstat[i], pval[i] = spstats.wilcoxon(A_po[:,i], A_pr[:,i], zero_method='zsplit') # Non normal distrib 
    # Correct for multiple testing    
    reject, pval_correct = fdrcorrection(pval, alpha=alpha)
    w_test = reject, pval_correct, tstat
    return w_test

def multiple_t_test(A_po, A_pr, nchans, alpha=0.05):
    # maybe HFB_db variablenot necessary
    """t test for visual responsivity"""
    A_po = sample_mean(A_po)
    A_pr = sample_mean(A_pr)
    # Initialise inflated p values
    pval = [0]*nchans
    tstat = [0]*nchans
    # Compute inflated stats
    for i in range(0,nchans):
        tstat[i], pval[i] = spstats.ttest_ind(A_po[:,i], A_pr[:,i], equal_var=False)  
    # Correct for multiple testing    
    reject, pval_correct = fdrcorrection(pval, alpha=alpha)
    w_test = reject, pval_correct, tstat
    return w_test

def significant_chan(reject, HFB_db):
    """Used in detect_visual_chan function"""
    idx = np.where(reject==True)
    idx = idx[0]
    visual_chan = []
    for i in list(idx):
        visual_chan.append(HFB_db.info['ch_names'][i])
    return visual_chan

def detect_visual_chan(HFB_db, tmin_pr=-0.4, tmax_pr=-0.1, tmin_po=0.1, tmax_po=0.5, alpha=0.05):
    """Return statistically significant visual channels with effect size"""
    A_pr = crop_HFB(HFB_db, tmin=tmin_pr, tmax=tmax_pr)
    A_po = crop_HFB(HFB_db, tmin=tmin_po, tmax=tmax_po)
    nchans = len(HFB_db.info['ch_names'])
    w_test = multiple_t_test(A_pr, A_po, nchans, alpha=alpha)
    reject = w_test[0]
    w_size = w_test[2]
    visual_chan = significant_chan(reject, HFB_db)
    return visual_chan, w_size

def compute_latency(visual_HFB, image_id, visual_channels, alpha = 0.05):
    """Compute latency response of visual channels"""
    A_po = crop_stim_HFB(visual_HFB, image_id, tmin=0, tmax=1.5)
    A_pr = crop_stim_HFB(visual_HFB, image_id, tmin=-0.4, tmax=-0.1)
    A_baseline = sample_mean(A_pr) #No 
    
    pval = [0]*A_po.shape[2]
    tstat = [0]*A_po.shape[2]
    latency_response = [0]*len(visual_channels)
    
    for i in range(0, len(visual_channels)):
        for t in range(0,np.size(A_po,2)):
            tstat[t] = spstats.ttest_ind(A_po[:,i,t], A_baseline[:,i], equal_var=False)
            pval[t] = tstat[t][1]
            
        reject, pval_correct = fdrcorrection(pval, alpha=alpha) # correct for multiple hypotheses
        
        for t in range(0,np.size(A_po,2)):
            if np.all(reject[t:t+25])==True :
                latency_response[i]=t/500*1e3
                break 
            else:
                continue
    return latency_response

def classify_retinotopic(latency_response, visual_channels, dfelec, latency_threshold=180):
    """Return retinotopic areas V1 and V2"""
    group = ['other']*len(visual_channels)
    visual_channels_split = [visual_channels[i].split('-')[0] for i in range(len(visual_channels))]
    for idx, channel in enumerate(visual_channels_split):
        brodman = dfelec['Brodman'].loc[dfelec['electrode_name']==channel]
        brodman = brodman.to_string(index=False)
        if brodman ==' V1' and latency_response[idx] <= latency_threshold :
            group[idx]='V1'
        elif brodman==' V2' and latency_response[idx] <= latency_threshold:
            group[idx]='V2'
        else:
            continue 
    return group

def classify_Face_Place(visual_HFB, face_id, place_id, visual_channels, group, alpha=0.05):
    A_face = crop_stim_HFB(visual_HFB, face_id, tmin=0.1, tmax=0.5)
    A_place = crop_stim_HFB(visual_HFB, place_id, tmin=0.1, tmax=0.5)
    
    n_visuals = len(visual_HFB.info['ch_names'])
    w_test = multiple_t_test(A_face, A_place, n_visuals, alpha=alpha)
    reject = w_test[0]
    w_size = w_test[2]
    
    # Significant electrodes located outside of V1 and V2 are Face or Place responsive
    for idx, channel in enumerate(visual_channels):
        if reject[idx]==False:
            continue
        else:
            if group[idx]=='V1':
                continue
            elif group[idx]=='V2':
                continue
            else:
                if w_size[idx]>0:
                   group[idx] = 'Face'
                else:
                   group[idx] = 'Place'
    return group, w_size

# %% Make a dictionary informative about visually responsive time series 

def raw_to_visual_populations(raw, bands, dfelec,latency_threshold=170):
    
    # Extract and normalise HFB
    HFB_db = raw_to_HFB_db(raw, bands, t_pr = -0.5, t_po = 1.75, baseline=None,
                       preload=True, tmin=-0.4, tmax=-0.1)
    events, event_id = mne.events_from_annotations(raw)
    face_id = extract_stim_id(event_id)
    place_id = extract_stim_id(event_id, cat='Place')
    image_id = face_id+place_id
    
    # Detect visual channels
    visual_chan = detect_visual_chan(HFB_db, tmin_pr=-0.4, tmax_pr=-0.1, tmin_po=0.1, tmax_po=0.5)
    visual_chan = visual_chan[0]
    visual_HFB = HFB_db.copy().pick_channels(visual_chan)
    
    # Compute latency response
    latency_response = compute_latency(visual_HFB, image_id, visual_chan)
    
    # Classify V1 and V2 populations
    group = classify_retinotopic(latency_response, visual_chan, 
                                             dfelec, latency_threshold=latency_threshold)
    # Classify Face and Place populations
    group, w_size = classify_Face_Place(visual_HFB, face_id, place_id, visual_chan, group, alpha=0.05)
    
    # Create visual_populations dictionary 
    visual_populations = {'chan_name': [], 'group': [], 'latency': [], 
                          'effect_size':[], 'brodman': [], 'DK': []}
    
    visual_populations['chan_name'] = visual_chan
    visual_populations['group'] = group
    visual_populations['latency'] = latency_response
    visual_populations['effect_size'] = w_size
    for chan in visual_chan:
        visual_populations['brodman'].extend(dfelec['Brodman'].loc[dfelec['electrode_name']==chan])
        visual_populations['DK'].extend(dfelec['ROI_DK'].loc[dfelec['electrode_name']==chan])
    return visual_populations
    
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


def make_visual_chan_dictionary(df_visual, raw, HFB, epochs, sub='DiAs'): 
   # Return visual channels in dictionary to save in matfile 
    events, event_id = mne.events_from_annotations(raw)
    visual_chan = list(df_visual['chan_name'].loc[df_visual['subject_id']== sub])
    category = list(df_visual['category'].loc[df_visual['subject_id']== sub])
    brodman = list(df_visual['brodman'].loc[df_visual['subject_id']== sub])
    DK = list(df_visual['DK'].loc[df_visual['subject_id']== sub] )
    ts = log_transform(HFB, picks=visual_chan)
    multitrial_ts = log_transform(epochs, picks=visual_chan) # make data normal
    #multitrial_ts = np.exp(multitrial_ts)
    # ch_idx = mne.pick_channels(epochs.info['ch_names'], include=visual_chan)
    visual_dict = dict(ts=ts, multitrial_ts=multitrial_ts, chan=visual_chan, 
                   category=category, brodman=brodman, DK = DK, events=events,
                   event_id = event_id)
    return visual_dict 


# %% Create category specific time series 

def epoch_category(HFB_visual, cat='Rest', tmin=-0.5, tmax=1.75):
    if cat == 'Rest':
        events_1 = mne.make_fixed_length_events(HFB_visual, id=32, start=70, 
                                                stop=200, duration=2, first_samp=False, overlap=0.0)
        events_2 = mne.make_fixed_length_events(HFB_visual, id=32, 
                                                start=280, stop=400, duration=2, first_samp=False, overlap=0.0)
        
        events = np.concatenate((events_1,events_2))
        rest_id = {'Rest': 32}
        # epoch
        epochs= mne.Epochs(HFB_visual, events, event_id= rest_id, 
                            tmin=tmin, tmax=tmax, baseline= None, preload=True)
    else:
        stim_events, stim_events_id = mne.events_from_annotations(HFB_visual)
        cat_id = extract_stim_id(stim_events_id, cat = cat)
        epochs= mne.Epochs(HFB_visual, stim_events, event_id= stim_events_id, 
                            tmin=tmin, tmax=tmax, baseline= None, preload=True)
        epochs = epochs[cat_id]
        events = epochs.events
    return epochs, events

def db_transform_category(epochs, events, event_id=None, tmin=-0.4, tmax=-0.1, t_pr=-0.5):
    baseline = extract_baseline(epochs, tmin=tmin, tmax=tmax)
    A = epochs.get_data()
    A = np.divide(A, baseline[np.newaxis,:,np.newaxis]) # divide by baseline
    A = np.nan_to_num(A)
    A = 10*np.log10(A) # convert to db
    HFB = mne.EpochsArray(A, epochs.info, events=events, 
                             event_id=event_id, tmin=t_pr) 
    return HFB


def visually_responsive_HFB(sub_id= 'DiAs', proc= 'preproc', 
                            stage= '_BP_montage_HFB_raw.fif'):
    """Load high frequency envelope of visually responsive channels"""
    subject = cf.Subject(name=sub_id)
    raw = subject.load_raw_data(proc= proc, stage= stage)
    visual_chan = subject.pick_visual_chan()
    visual_chan_name = visual_chan['chan_name'].values.tolist()
    HFB_visual = raw.copy().pick_channels(visual_chan_name)
    return HFB_visual

def low_high_HFB(sub_id= 'DiAs', proc= 'preproc', stage= '_BP_montage_HFB_raw.fif'):
    """Extract high frequency envelope of low and high channels"""
    subject = cf.Subject(name=sub_id)
    raw = subject.load_raw_data(proc= proc, stage= stage)
    visual_chan = subject.low_high_chan()
    visual_chan_name = visual_chan['chan_name'].values.tolist()
    HFB_visual = raw.copy().pick_channels(visual_chan_name)
    return HFB_visual
        
def category_specific_HFB(HFB_visual, group, visual_chan, cat='Rest', tmin_crop = 0.5, tmax_crop=1.75) :
        epochs, events = epoch_category(HFB_visual, cat=cat, tmin=-0.5, tmax=1.75)
        HFB = db_transform_category(epochs, events, tmin=-0.4, tmax=-0.1, t_pr=-0.5)
        HFB = HFB.crop(tmin=tmin_crop, tmax=tmax_crop)
        return HFB 
        # Get data and permute into hierarchical order

def subject_specific_visual_indices(visual_chan):
    """Return indices of channel belonging to a visual population 
    present in a subject"""
    group = visual_chan['group'].unique().tolist()
    indices = dict.fromkeys(group)
    for key in group:
       indices[key] = visual_chan[visual_chan['group'] == key].index.to_list()
    return indices

def visual_population_indices(visual_chan):
    """Return population indices of a given subject in each population from a 
    visual hierarchy of interest"""
    group = visual_chan['group'].unique().tolist()
    visual_hierarchy = ['V1', 'V2', 'Place', 'Face']
    population_indices = dict.fromkeys(visual_hierarchy) 
    indices = subject_specific_visual_indices(visual_chan)
   # Find indices of channels in each population
    for key in population_indices:
       if key in group:
           population_indices[key] = indices[key]
       else: 
           population_indices[key] = [] # no channel in population
    return population_indices

def order_channel_indices(population_indices, group):
    """Order channel indices along visual hierarchy"""
    
    ordered_channel_indices = [] 
    for key in population_indices:
       if key in group:
           ordered_channel_indices.extend(population_indices[key])
       else:
           continue
    return ordered_channel_indices

def order_population_indices(population_indices, ordered_channel_indices, group):
    """Return population indices from channel ordered along visual hierarhcy"""
    
    ordered_population_indices = dict.fromkeys(population_indices) 
    for key in population_indices:
     if key in group:
         ordered_population_indices[key] = [0]*len(population_indices[key])
         for idx, i in enumerate(population_indices[key]):
             ordered_population_indices[key][idx] = ordered_channel_indices.index(i) +1 #adapt to matlab indexing
     else: 
         ordered_population_indices[key] = []
    return ordered_population_indices

def order_visual_data_indices(ordered_channel_indices, HFB):
    """Order visual HFB data indices along visual herarchy"""
    X = HFB.get_data()
    X_ordered = np.zeros_like(X)
    for idx, i in enumerate(ordered_channel_indices):
            X_ordered[:,idx,:] = X[:,i,:]
    X = X_ordered
    return X

def visual_data_dict(X, ordered_population_indices):
    """Build a dictionary for mvgc analysis of category specific visual time series"""
    # Save time series into dictionary 
    X = np.transpose(X, axes = (1,2,0)) # permute for compatibility with mvgc
    visual_data = dict(data= X, populations=ordered_population_indices)
    return visual_data

def category_specific_HFB_to_visual_data(HFB, visual_chan):
    
    group = visual_chan['group'].unique().tolist()
    population_indices = visual_population_indices(visual_chan)
    ordered_channel_indices =order_channel_indices(population_indices, group)
    ordered_population_indices = order_population_indices(population_indices, 
                                                          ordered_channel_indices, group)
    X = order_visual_data_indices(ordered_channel_indices, HFB)
    visual_data = visual_data_dict(X, ordered_population_indices)
    return visual_data

def HFB_to_visual_data(HFB, visual_chan, cat='Rest', tmin_crop = 0.5, tmax_crop=1.75):
    
    group = visual_chan['group'].unique().tolist()
    HFB = category_specific_HFB(HFB, group, visual_chan, cat='Rest', tmin_crop = 0.5, tmax_crop=1.75)
    visual_data = category_specific_HFB_to_visual_data(HFB, visual_chan)
    return visual_data