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
from netneurotools import stats as nnstats

# TODO: solve mismatch between events and epoch
# %% Extract hfb envelope

def extract_hfb(raw, l_freq=60.0, nband=6, band_size=20.0, l_trans_bandwidth= 10.0,
                h_trans_bandwidth= 10.0, filter_length='auto', phase='minimum'):
    """
    Extract the high frequency broadband (hfb) from LFP iEEG signal.
    ----------
    Parameters
    ----------
    raw: MNE raw object
        the LFP data to be filtered (in MNE python raw structure)
    l_freq: float, optional
            lowest frequency in Hz
    nband: int, optional
           Number of frequency bands
    band_size: float, optional
                size of frequency band in Hz
    See mne.io.Raw.filter documentation for additional optional parameters
    -------
    Returns
    -------
    hfb: MNE raw object
        The high frequency broadband
    """
    nobs = len(raw.times)
    nchan = len(raw.info['ch_names'])
    bands = freq_bands(l_freq=l_freq, nband=nband, band_size=band_size)
    hfb = np.zeros(shape=(nchan, nobs))
    mean_amplitude = np.zeros(shape=(nchan,))
    
    for band in bands:
        # extract band specific envelope
        envelope = extract_envelope(raw, l_freq=band, band_size=band_size)
    # hfb is weighted average of bands specific envelope over high gamma
        env_norm = mean_normalise(envelope)
        hfb += env_norm
        mean_amplitude += np.mean(envelope, axis=1)
    hfb = hfb/nband
    # Convert hfb in volts
    hfb = hfb * mean_amplitude[:,np.newaxis]
    hfb = np.nan_to_num(hfb) # convert NaN to 0
    # Create Raw object for further MNE processing
    hfb = mne.io.RawArray(hfb, raw.info)
    hfb.set_annotations(raw.annotations)
    return hfb

def freq_bands(l_freq=60.0, nband=6, band_size=20.0):
    """
    Create a list of 20Hz spaced frequencies from [60, 160]Hz (high gamma)
    These frequencies will be used to banpass the iEEG signal for 
    high frequency envelope extraction
    
    Parameters
    ----------
    l_freq: float, optional
            lowest frequency in Hz
    nband: int, optional
           Number of frequency bands
    band_size: int, optional
                size of frequency band in Hz
    
    Returns
    -------
    bands: list
            List of frequency bands
            
    """
    bands = [l_freq + i * band_size for i in range(0, nband)]
    return bands

def extract_envelope(raw, l_freq=60.0, band_size=20.0, l_trans_bandwidth= 10.0, 
                     h_trans_bandwidth= 10.0, filter_length='auto', phase='minimum'):
    """
    Extract the envelope of a bandpass signal. The filter is constructed 
    using MNE python filter function. Hilbert transform is computed from MNE
    apply_hilbert() function. Filter and Hilber function themselves rely mostly
    on scipy signal filtering and hilbert funtions.
    ----------
    Parameters
    ----------
    raw: MNE raw object
        the LFP data to be filtered (in MNE python raw structure)
    %(l_freq)s
    %(band_size)s
    See mne.io.Raw.filter documentation for additional optional parameters
    
    -------
    Returns
    -------
    envelope: MNE raw object
             The envelope of the bandpass signal
    """
    raw_band = raw.copy().filter(l_freq=l_freq, h_freq=l_freq+band_size,
                                 phase=phase, filter_length=filter_length,
                                 l_trans_bandwidth= l_trans_bandwidth, 
                                 h_trans_bandwidth= h_trans_bandwidth,
                                     fir_window='blackman')
    envelope = raw_band.copy().apply_hilbert(envelope=True).get_data()
    return envelope

def mean_normalise(envelope):
    """
    Normalise the envelope by its mean. Useful for extracting hfb which is a
    weighted average of each envelope accross 20Hz frequency bands.
    ----------
    Parameters
    ----------
    envelope: MNE raw object
            The envelope of the band pass signal
    -------
    Returns
    -------
    envelope_norm: MNE raw object
                    The mean normalised envelope
    """
    envelope_mean = np.mean(envelope, axis=1)
    envelope_norm = np.divide(envelope, envelope_mean[:,np.newaxis])
    return envelope_norm
#%% Normalise with baseline, log transform and epoch hfb

def raw_to_hfb_db(raw, l_freq=60.0, nband=6, band_size=20.0, t_prestim = -0.5,
                  l_trans_bandwidth= 10.0, h_trans_bandwidth= 10.0, 
                  filter_length='auto', phase='minimum', t_postim = 1.75, 
                  baseline=None, preload=True, tmin=-0.4, tmax=-0.1, mode='logratio'):
    """
    Compute hfb in decibel from raw LFP
    ----------
    Parameters
    ----------
    raw: MNE raw object
        The raw LFP
    t_postim: float, optional
        post stimulus epoch stop
    t_prestim: float
        pre stimulus epoch starts
    tmin: float
        baseline starts
    tmax: float
        baseline stops
    See MNE python documentation for other optional parameters
    """
    hfb = extract_hfb(raw, l_freq=l_freq, nband=nband, band_size=band_size,
                l_trans_bandwidth= l_trans_bandwidth, h_trans_bandwidth= h_trans_bandwidth,
                filter_length=filter_length, phase=phase)
    epochs = epoch_hfb(hfb, t_prestim = t_prestim, t_postim = t_postim, baseline=baseline,
                       preload=preload)
    hfb_db = db_transform(epochs, tmin=tmin, tmax=tmax, t_prestim=t_prestim, 
                          mode='logratio')
    return hfb_db


def epoch_hfb(hfb, t_prestim = -0.5, t_postim = 1.75, baseline=None, preload=True):
    """
    Epoch stimulus condition hfb using MNE Epochs function
    """
    events, event_id = mne.events_from_annotations(hfb) 
    epochs = mne.Epochs(hfb, events, event_id= event_id, tmin=t_prestim, 
                    tmax=t_postim, baseline=baseline,preload=preload)
    return epochs


def db_transform(epochs, tmin=-0.4, tmax=-0.1, t_prestim = -0.5, mode='logratio'):
    """
    Normalise hfb with pre stimulus baseline and log transform for result in dB
    Allows for cross channel comparison via a single scale.
    """
    events = epochs.events
    event_id = epochs.event_id
    # Drop boundary event for compatibility. This does not affect results.
    del event_id['boundary'] 
    A = epochs.get_data()
    times = epochs.times
    # db transform
    A = 10*mne.baseline.rescale(A,times,baseline=(tmin,tmax),mode=mode)
    # Create epoch object from array
    hfb = mne.EpochsArray(A, epochs.info, events=events, 
                             event_id=event_id, tmin=t_prestim)
    return hfb


def extract_baseline(epochs, tmin=-0.4, tmax=-0.1):
    """
    Extract baseline by averaging prestimulus accross time and trials. From 
    testing, it does not differs much to MNE baseline.rescale, so might as well
    use MNE
    """
    baseline = epochs.copy().crop(tmin=tmin, tmax=tmax) # Extract prestimulus baseline
    baseline = baseline.get_data()
    baseline = np.mean(baseline, axis=(0,2)) # average over time and trials
    return baseline 


#%% Detect visually responsive populations

def detect_visual_chan(hfb_db, tmin_prestim=-0.4, tmax_prestim=-0.1,tmin_postim=0.1,
                       tmax_postim=0.5, alpha=0.05, zero_method='pratt', 
                       alternative='greater'):
    """
    Detect visually responsive channels by testing hypothesis of no difference 
    between prestimulus and postimulus HFB amplitude.
    ----------
    Parameters
    ----------
    hfb_db: MNE raw object
            HFB of iEEG in decibels
    tmin_prestim: float
                starting time prestimulus amplitude
    tmax_preststim: float
                    stoping time prestimlus amplituds
    tmin_postim: float
                 starting time postimuls amplitude
    tmax_postim: float
                stopping time postimulus amplitude
    alpha: float
        significance threshold to reject the null
    From scipy.stats.wilcoxon:
    alternative: {“two-sided”, “greater”, “less”}, optional
    zero_method: {“pratt”, “wilcox”, “zsplit”}, optional
    -------
    Returns
    -------
    visual_chan: list.
                List of visually responsive channels
    effect_size: list
                 visual responsivity effect size
    """
    A_prestim = crop_hfb(hfb_db, tmin=tmin_prestim, tmax=tmax_prestim)
    A_postim = crop_hfb(hfb_db, tmin=tmin_postim, tmax=tmax_postim)
    reject, pval_correct, tstat = multiple_wilcoxon_test(A_postim, A_prestim,
                                                         alpha=alpha, 
                                                         zero_method=zero_method,
                                                         alternative=alternative)
    visual_responsivity = compute_visual_responsivity(A_postim, A_prestim)
    visual_chan, effect_size = visual_chans_stats(reject, visual_responsivity, hfb_db)
    return visual_chan, effect_size


def crop_hfb(hfb_db, tmin=-0.5, tmax=-0.05):
    """
    crop hfb between over [tmin tmax].
    Input : MNE raw object
    Return: array
    """
    A = hfb_db.copy().crop(tmin=tmin, tmax=tmax).get_data()
    return A


def crop_stim_hfb(hfb_db, stim_id, tmin=-0.5, tmax=-0.05):
    """
    crop condition specific hfb between over [tmin tmax].
    Input : MNE raw object
    Return: array
    """
    A = hfb_db[stim_id].copy().crop(tmin=tmin, tmax=tmax).get_data()
    return A


def multiple_perm_test(A_postim, A_prestim, nchans, alpha=0.05):
    A_postim = np.mean(A_postim, axis=-1)
    A_prestim = np.mean(A_prestim, axis=-1)
    # Initialise inflated p values
    pval = [0]*nchans
    tstat = [0]*nchans
    # Compute inflated stats
    for i in range(0,nchans):
        tstat[i], pval[i] = nnstats.permtest_rel(A_postim[:,i], A_prestim[:,i])  
    # Correct for multiple testing    
    reject, pval_correct = fdrcorrection(pval, alpha=alpha)
    return reject


def multiple_wilcoxon_test(A_postim, A_prestim, zero_method='pratt', alternative = 'greater', alpha=0.05):
    """
    Wilcoxon test hypothesis of no difference between prestimulus and postimulus amplitude
    Correct for multilple hypothesis test.
    ----------
    Parameters
    ----------
    A_postim: (...,times) array
            Postimulus amplitude
    A_prestim: (...,times) array
                Presimulus amplitude
    alpha: float
        significance threshold to reject the null
    From scipy.stats.wilcoxon:
    alternative: {“two-sided”, “greater”, “less”}, optional
    zero_method: {“pratt”, “wilcox”, “zsplit”}, optional
    """
    A_postim = np.mean(A_postim, axis=-1)
    A_prestim = np.mean(A_prestim, axis=-1)
    # Iniitialise inflated p values
    nchans = A_postim.shape[1]
    pval = [0]*nchans
    tstat = [0]*nchans
    # Compute inflated stats given non normal distribution
    for i in range(0,nchans):
        tstat[i], pval[i] = spstats.wilcoxon(A_postim[:,i], A_prestim[:,i],
                                             zero_method=zero_method, alternative=alternative) 
    # Correct for multiple testing    
    reject, pval_correct = fdrcorrection(pval, alpha=alpha)
    w_test = reject, pval_correct, tstat
    return w_test


def multiple_t_test(A_postim, A_prestim, nchans, alpha=0.05):
    # maybe hfb_db variablenot necessary
    """t test for visual responsivity"""
    A_postim = np.mean(A_postim, axis=-1)
    A_prestim = np.mean(A_prestim, axis=-1)
    # Initialise inflated p values
    pval = [0]*nchans
    tstat = [0]*nchans
    # Compute inflated stats
    for i in range(0,nchans):
        tstat[i], pval[i] = spstats.ttest_ind(A_postim[:,i], A_prestim[:,i], equal_var=False)
    # Correct for multiple testing    
    reject, pval_correct = fdrcorrection(pval, alpha=alpha)
    w_test = reject, pval_correct, tstat
    return w_test


def cohen_d(x, y):
    """
    Compute cohen d effect size between 1D array x and y
    """
    n1 = np.size(x)
    n2 = np.size(y)
    m1 = np.mean(x)
    m2 = np.mean(y)
    s1 = np.std(x)
    s2 = np.std(y)
    
    s = (n1 - 1)*(s1**2) + (n2 - 1)*(s2**2)
    s = s/(n1+n2-2)
    s= np.sqrt(s)
    num = m1 - m2
    
    cohen = num/s
    
    return cohen


def compute_visual_responsivity(A_postim, A_prestim):
    """
    Compute visual responsivity of a channel from cohen d.
    """
    nchan = A_postim.shape[1]
    visual_responsivity = [0]*nchan
    
    for i in range(nchan):
        x = np.ndarray.flatten(A_postim[:,i,:])
        y = np.ndarray.flatten(A_prestim[:,i,:])
        visual_responsivity[i] = cohen_d(x,y)
        
    return visual_responsivity


def visual_chans_stats(reject, visual_responsivity, hfb_db):
    """
    Return visual channels with their corresponding responsivity
    """
    idx = np.where(reject==True)
    idx = idx[0]
    visual_chan = []
    effect_size = []
    
    for i in list(idx):
        if visual_responsivity[i]>0:
            visual_chan.append(hfb_db.info['ch_names'][i])
            effect_size.append(visual_responsivity[i])
        else:
            continue
    return visual_chan, effect_size

#%% Compute visual channels latency response

def pval_series(visual_hfb, image_id, visual_channels, alpha = 0.05):
    """
    Return pvalue of postimulus visual responsivity along observations
    """
    nchan = len(visual_channels)
    A_postim = crop_stim_hfb(visual_hfb, image_id, tmin=0, tmax=1.5)
    A_prestim = crop_stim_hfb(visual_hfb, image_id, tmin=-0.4, tmax=0)
    A_baseline = np.mean(A_prestim, axis=-1) #No 
    nobs = A_postim.shape[2]
    
    pval = [0]*nobs
    tstat = [0]*nobs
    
    reject = np.zeros((nchan, nobs))
    pval_correct = np.zeros((nchan, nobs))
    
    for i in range(0, nchan):
        for t in range(0,nobs):
            tstat[t] = spstats.wilcoxon(A_postim[:,i,t], A_baseline[:,i], zero_method='pratt')
            pval[t] = tstat[t][1]
            
        reject[i,:], pval_correct[i, :] = fdrcorrection(pval, alpha=alpha) # correct for multiple hypotheses
        
    return reject, pval_correct


def compute_latency(visual_hfb, image_id, visual_channels, alpha = 0.05):
    """
    Compute latency response of visual channels"
    """
    A_postim = crop_stim_hfb(visual_hfb, image_id, tmin=0, tmax=1.5)
    A_prestim = crop_stim_hfb(visual_hfb, image_id, tmin=-0.4, tmax=0)
    A_baseline = np.mean(A_prestim, axis=-1) #No
    
    pval = [0]*A_postim.shape[2]
    tstat = [0]*A_postim.shape[2]
    latency_response = [0]*len(visual_channels)
    
    for i in range(0, len(visual_channels)):
        for t in range(0,np.size(A_postim,2)):
            tstat[t] = spstats.wilcoxon(A_postim[:,i,t], A_baseline[:,i], zero_method='pratt')
            pval[t] = tstat[t][1]
            
        reject, pval_correct = fdrcorrection(pval, alpha=alpha) # correct for multiple hypotheses
        
        for t in range(0,np.size(A_postim,2)):
            if np.all(reject[t:t+50])==True :
                latency_response[i]=t/500*1e3
                break 
            else:
                continue
    return latency_response

# %% Classify channels into Face, Place and retinotopic channels

def classify_Face_Place(visual_hfb, face_id, place_id, visual_channels, 
                        tmin_postim=0.2, tmax_postim=0.5, alpha=0.05):
    nchan = len(visual_channels)
    group = ['O']*nchan
    category_selectivity = [0]*len(group)
    A_face = crop_stim_hfb(visual_hfb, face_id, tmin=tmin_postim, tmax=tmax_postim)
    A_place = crop_stim_hfb(visual_hfb, place_id, tmin=tmax_postim, tmax=tmax_postim)
    
    n_visuals = len(visual_hfb.info['ch_names'])
    w_test_plus = multiple_wilcoxon_test(A_face, A_place, n_visuals, 
                                         zero_method='pratt', alternative = 'greater',alpha=alpha)
    reject_plus = w_test_plus[0]

    
    w_test_minus = multiple_wilcoxon_test(A_face, A_place, n_visuals,
                                          zero_method='pratt', alternative = 'less',alpha=alpha)
    reject_minus = w_test_minus[0]

    
    # Significant electrodes located outside of V1 and V2 are Face or Place responsive
    for idx, channel in enumerate(visual_channels):
        A_face = crop_stim_hfb(visual_hfb, face_id, tmin=tmin_postim, tmax=tmax_postim)
        A_place = crop_stim_hfb(visual_hfb, place_id, tmin=tmax_postim, tmax=tmax_postim)
        A_face = np.ndarray.flatten(A_face[:,idx,:])
        A_place = np.ndarray.flatten(A_place[:,idx,:])
        if reject_plus[idx]==False and reject_minus[idx]==False :
            continue
        else:
            if reject_plus[idx]==True :
               group[idx] = 'F'
               category_selectivity[idx] = cohen_d(A_face, A_place)
            elif reject_minus[idx]==True :
               group[idx] = 'P'
               category_selectivity[idx] = cohen_d(A_place, A_face)
    return group, category_selectivity


def classify_retinotopic(visual_channels, group, dfelec):
    """Return retinotopic from V1 and V2"""
    nchan = len(group)
    bipolar_visual = [visual_channels[i].split('-') for i in range(nchan)]
    for i in range(nchan):
        brodman = (dfelec['Brodman'].loc[dfelec['electrode_name']==bipolar_visual[i][0]].to_string(index=False), 
                   dfelec['Brodman'].loc[dfelec['electrode_name']==bipolar_visual[i][1]].to_string(index=False))
        if ' V1' in brodman or ' V2' in brodman and group[i]!='F' and  group[i] !='P':
            group[i]='R'
    return group


# %% Make a dictionary to save visual channel table in mat file

def hfb_to_visual_populations(hfb, dfelec, t_prestim = -0.5, t_postim = 1.75, baseline=None,
                       preload=True, tmin_prestim=-0.2, tmax_prestim=0, tmin_postim=0.1,
                       tmax_postim=0.5, alpha= 0.01):
    
    # Extract and normalise hfb
    hfb_db = hfb_to_db(hfb, t_prestim = t_prestim, t_postim = t_postim, baseline=None,
                       preload=True, tmin=tmin_prestim, tmax=tmax_prestim)
    events, event_id = mne.events_from_annotations(hfb)
    face_id = extract_stim_id(event_id, cat = 'Face')
    place_id = extract_stim_id(event_id, cat='Place')
    image_id = face_id+place_id
    
    # Detect visual channels
    visual_chan, visual_responsivity = detect_visual_chan(hfb_db, tmin_prestim=tmax_prestim, tmax_prestim=tmax_prestim, 
                                     tmin_postim=tmin_postim, tmax_postim=tmax_postim, alpha=alpha)
    
    visual_hfb = hfb_db.copy().pick_channels(visual_chan)
    
    # Compute latency response
    latency_response = compute_latency(visual_hfb, image_id, visual_chan)
    
    # Classify Face and Place populations
    group, category_selectivity = classify_Face_Place(visual_hfb, face_id, place_id, visual_chan, 
                                tmin_postim=tmin_postim, tmax_postim=tmax_postim, alpha=alpha)
    # Classify retinotopic populations
    group = classify_retinotopic(visual_chan, group, dfelec)
    
    # Create visual_populations dictionary 
    visual_populations = {'chan_name': [], 'group': [], 'latency': [], 
                          'brodman': [], 'DK': [], 'X':[], 'Y':[], 'Z':[]}
    
    visual_populations['chan_name'] = visual_chan
    visual_populations['group'] = group
    visual_populations['latency'] = latency_response
    visual_populations['visual_responsivity'] = visual_responsivity
    visual_populations['category_selectivity'] = category_selectivity
    for chan in visual_chan: 
        chan_name_split = chan.split('-')[0]
        visual_populations['brodman'].extend(dfelec['Brodman'].loc[dfelec['electrode_name']==chan_name_split])
        visual_populations['DK'].extend(dfelec['ROI_DK'].loc[dfelec['electrode_name']==chan_name_split])
        visual_populations['X'].extend(dfelec['X'].loc[dfelec['electrode_name']==chan_name_split])
        visual_populations['Y'].extend(dfelec['Y'].loc[dfelec['electrode_name']==chan_name_split])
        visual_populations['Z'].extend(dfelec['Z'].loc[dfelec['electrode_name']==chan_name_split])
        
    return visual_populations

def make_visual_chan_dictionary(df_visual, raw, hfb, epochs, sub='DiAs'): 
   # Return visual channels in dictionary to save in matfile 
    events, event_id = mne.events_from_annotations(raw)
    visual_chan = list(df_visual['chan_name'].loc[df_visual['subject_id']== sub])
    category = list(df_visual['category'].loc[df_visual['subject_id']== sub])
    brodman = list(df_visual['brodman'].loc[df_visual['subject_id']== sub])
    DK = list(df_visual['DK'].loc[df_visual['subject_id']== sub] )
    ts = log_transform(hfb, picks=visual_chan)
    multitrial_ts = log_transform(epochs, picks=visual_chan) # make data normal
    #multitrial_ts = np.exp(multitrial_ts)
    # ch_idx = mne.pick_channels(epochs.info['ch_names'], include=visual_chan)
    visual_dict = dict(ts=ts, multitrial_ts=multitrial_ts, chan=visual_chan, 
                   category=category, brodman=brodman, DK = DK, events=events,
                   event_id = event_id)
    return visual_dict 

def hfb_to_db(hfb, t_prestim = -0.5, t_postim = 1.75, baseline=None,
                       preload=True, tmin=-0.4, tmax=-0.1):
    epochs = epoch_hfb(hfb, t_prestim = t_prestim, t_postim = t_postim, baseline=baseline,
                       preload=preload)
    hfb_db = db_transform(epochs, tmin=tmin, tmax=tmax, t_prestim=t_prestim)
    return hfb_db


def log_transform(epochs, picks):
    # transform into log normal distribution, should also work with raw structure
    data = epochs.copy().pick(picks=picks).get_data()
    log_hfb = np.log(data)
    return log_hfb


def extract_stim_id(event_id, cat = 'Face'):
    p = re.compile(cat)
    stim_id = []
    for key in event_id.keys():
        if p.match(key):
            stim_id.append(key)
    return stim_id 

# %% Create category specific time series and input for mvgc

def epoch_category(hfb_visual, cat='Rest', tmin=-0.5, tmax=1.75):
    """Epoch category specific envelope"""
    if cat == 'Rest':
        events_1 = mne.make_fixed_length_events(hfb_visual, id=32, start=100, 
                                                stop=156, duration=2, first_samp=False, overlap=0.0)
        events_2 = mne.make_fixed_length_events(hfb_visual, id=32, 
                                                start=300, stop=356, duration=2, first_samp=False, overlap=0.0)
        
        events = np.concatenate((events_1,events_2))
        rest_id = {'Rest': 32}
        # epoch
        epochs= mne.Epochs(hfb_visual, events, event_id= rest_id, 
                            tmin=tmin, tmax=tmax, baseline= None, preload=True)
    else:
        stim_events, stim_events_id = mne.events_from_annotations(hfb_visual)
        cat_id = extract_stim_id(stim_events_id, cat = cat)
        epochs= mne.Epochs(hfb_visual, stim_events, event_id= stim_events_id, 
                            tmin=tmin, tmax=tmax, baseline= None, preload=True)
        epochs = epochs[cat_id]
        events = epochs.events
    return epochs, events

def db_transform_category(epochs, events, event_id=None, tmin=-0.4, tmax=-0.1, t_prestim=-0.5):
    """DB transform category specific epoched envelope to get closer to gaussianity"""
    baseline = extract_baseline(epochs, tmin=tmin, tmax=tmax)
    A = epochs.get_data()
    A = np.divide(A, baseline[np.newaxis,:,np.newaxis]) # divide by baseline
    A = np.nan_to_num(A)
    A = 10*np.log10(A) # convert to db
    hfb = mne.EpochsArray(A, epochs.info, events=events, 
                             event_id=event_id, tmin=t_prestim) 
    return hfb


def visually_responsive_hfb(sub_id= 'DiAs', proc= 'preproc', 
                            stage= '_BP_montage_hfb_raw.fif'):
    """Load high frequency envelope of visually responsive channels"""
    subject = cf.Subject(name=sub_id)
    raw = subject.load_raw_data(proc= proc, stage= stage)
    visual_chan = subject.pick_visual_chan()
    visual_chan = visual_chan['chan_name'].values.tolist()
    hfb_visual = raw.copy().pick_channels(visual_chan)
    return hfb_visual
        
def category_specific_hfb(hfb_visual, cat='Rest', tmin_crop = -0.5, tmax_crop=1.75) :
    """Return category specific visually respinsive hfb (rest, face, place) during a specific stimulus"""
    epochs, events = epoch_category(hfb_visual, cat=cat, tmin=-0.5, tmax=1.75)
    hfb = db_transform_category(epochs, events, tmin=-0.4, tmax=-0.1, t_prestim=-0.5)
    hfb = hfb.crop(tmin=tmin_crop, tmax=tmax_crop)
    return hfb 
        # Get data and permute into hierarchical order

def pick_visual_chan(picks, visual_chan):
    """Pick specific visual channels from all visually reponsive channels"""
    drop_index = []
    
    for chan in visual_chan['chan_name'].to_list():
        if chan in picks:
            continue
        else:
            drop_index.extend(visual_chan.loc[visual_chan['chan_name']==chan].index.tolist())
    
    visual_chan = visual_chan.drop(drop_index)
    visual_chan = visual_chan.reset_index(drop=True)
    return visual_chan

def parcellation_to_indices(visual_chan, parcellation='group'):
    """Return indices of channels from a given population
    parcellation: group (default, functional), DK (anatomical)"""
    group = visual_chan[parcellation].unique().tolist()
    group_indices = dict.fromkeys(group)
    for key in group:
       group_indices[key] = visual_chan.loc[visual_chan[parcellation]== key].index.to_list()
    if parcellation == 'DK': # adapt indexing for matlab
        for key in group:
            for i in range(len(group_indices[key])):
                group_indices[key][i] = group_indices[key][i]
    return group_indices

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

def order_visual_data_indices(ordered_channel_indices, hfb):
    """Order visual hfb channels indices along visual herarchy"""
    X = hfb.get_data()
    X_ordered = np.zeros_like(X)
    for idx, i in enumerate(ordered_channel_indices):
            X_ordered[:,idx,:] = X[:,i,:]
    X = X_ordered
    return X

def visual_data_dict(sorted_visual_chan, ordered_population_indices):
    """Build a dictionary for mvgc analysis of category specific visual time series"""
    # Save time series into dictionary 
    # X = np.transpose(X, axes = (1,2,0)) # permute for compatibility with mvgc
    visual_data = dict(populations=ordered_population_indices)
    visual_data['channel_to_population'] = sorted_visual_chan['group'].to_list()
    visual_data['brodman'] = sorted_visual_chan['brodman'].to_list()
    visual_data['DK'] = sorted_visual_chan['DK'].to_list()
    visual_data['latency'] = sorted_visual_chan['latency'].to_list()
    visual_data['visual_responsivity'] = sorted_visual_chan['visual_responsivity'].to_list()
    visual_data['category_selectivity'] = sorted_visual_chan['category_selectivity'].to_list()
    visual_data['chan_name'] = sorted_visual_chan['chan_name'].to_list()
    return visual_data

def category_specific_hfb_to_visual_data(hfb, visual_chan):
    
    group = visual_chan['group'].unique().tolist()
    population_indices = parcellation_to_indices(visual_chan, parcellation='group')
    DK_indices = parcellation_to_indices(visual_chan, parcellation='DK')
    sorted_visual_chan = visual_chan.sort_values(by='latency')
    sorted_indices = sorted_visual_chan.index.to_list()
    ordered_population_indices = order_population_indices(population_indices, 
                                                          sorted_indices, group)
    X = order_visual_data_indices(sorted_indices, hfb)
    X = np.transpose(X, axes = (1,2,0)) # permute for compatibility with mvgc
    
    time = hfb.times
    visual_data = visual_data_dict(sorted_visual_chan, ordered_population_indices)
    visual_data['time'] = time
    visual_data['DK_to_indices'] = DK_indices
    return X, visual_data

def hfb_to_visual_data(hfb, visual_chan, sfreq=250, cat='Rest', tmin_crop = 0.5, tmax_crop=1.75):
    hfb = category_specific_hfb(hfb, cat=cat, tmin_crop = tmin_crop, tmax_crop=tmax_crop)
    hfb = hfb.resample(sfreq=sfreq)
    X, visual_data = category_specific_hfb_to_visual_data(hfb, visual_chan)
    return X, visual_data

def plot_hfb_response(hfb_db, stim_id, picks='LTo4'):
    evok = hfb_db[stim_id].copy().pick(picks).average()
    evok_std = hfb_db[stim_id].copy().pick(picks).standard_error()
    ERP = evok.data
    ERP_std = evok_std.data
    time = hfb_db.times
    plt.plot(time, ERP[0,:])
    plt.fill_between(time, ERP[0,:]-1.96*ERP_std[0,:], ERP[0,:]+1.96*ERP_std[0,:],
                     alpha=0.3)
