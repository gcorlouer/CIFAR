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
import seaborn as sns
import pandas as pd
import os 
import mne 

from numpy import inf
from statsmodels.stats.multitest import fdrcorrection, multipletests
from netneurotools import stats as nnstats
from scipy import stats
from pathlib import Path, PurePath
from shutil import copy

#%% Create subject class for loading data

class Subject:
    """
    Class subject
    """
    def __init__(self, name='DiAs', task='stimuli', run='1'):
        """
        Parameters:
            - name: name of subject
            - task: 'stimuli', 'rest_baseline', 'sleep' 
            - run : 1, 2  
        """
        self.name = name
        self.task = task
        self.run = run 
#%% Load data at a given processing stage

    def load_data(self, proc= 'preproc', stage= '_BP_montage_HFB_raw.fif',
                  preload=True, epo=False):
        """
        Load data at specific processing stage into RAW object
        """
        datadir = self.processing_stage_path(proc=proc)
        sub = self.name
        fname = sub + stage
        fpath = datadir.joinpath(fname)
        if epo==False:
            data = mne.io.read_raw_fif(fpath, preload=preload)
        else:
            data = mne.read_epochs(fpath, preload=preload)
        return data

    def processing_stage_path(self, proc='raw_signal'):
        """
        Return data path at some processed stage
        """
        subject_path = self.subject_path()
        proc_path = subject_path.joinpath('EEGLAB_datasets', proc)
        return proc_path

    def subject_path(self):
        """
        Return path of the subject
        """    
        cohort_path = cifar_cohort_path() # home directory
        subject_path = cohort_path.joinpath(self.name)
        return subject_path

#%% Read EEGLAB dataset

    def read_eeglab(self, proc= 'raw_signal', suffix = '', ext='.set',preload=True):
        """
        Read EEGlab dataset as RAW object in MNE
        """
        fpath = self.dataset_path(proc = proc, suffix=suffix)
        raw = mne.io.read_raw_eeglab(fpath, preload=preload)
        
        return raw

    def dataset_path(self, proc='raw_signal', suffix='', ext='.set'):
        """
        Return dataset path
        """
        processing_stage_path = self.processing_stage_path(proc)
        dataset_ext = self.dataset_ext(suffix=suffix, ext=ext)
        dataset_path = processing_stage_path.joinpath(dataset_ext)
        dataset_path = os.fspath(dataset_path)
        return dataset_path
    
    def dataset(self, suffix=''):
        """
        Return  dataset name
        """
        dataset = [self.name, "freerecall", self.task, self.run, 'preprocessed']
        if suffix  == '':
            dataset = dataset
        else:
            dataset = dataset +[suffix]
        dataset = "_".join(dataset)
        return dataset

    def dataset_ext(self, suffix='', ext='.set'):
        """
        Return  dataset name  with extension
        """
        dataset = self.dataset(suffix)
        dataset_ext = dataset+ext
        return dataset_ext

#%% Pick visual channels

    def pick_visual_chan(self, fname = 'visual_BP_channels.csv'):
        """
        Return list of visually responsive channels
        """
        brain_path = self.brain_path()
        fpath = brain_path.joinpath(fname)
        visual_chan = pd.read_csv(fpath)
        # Not necessary but uncomment in case need sorting channels
        # visual_chan = visual_chan.sort_values(by='latency')
        return visual_chan
    
    def low_high_chan(self, fname = 'visual_channels_BP_montage.csv'):
        """
        Drop channels in other category to only keep retinotopic (low)  
        Face and Place selective channels (high)
        """
        visual_chan = self.pick_visual_chan(fname)
        visual_chan = visual_chan[visual_chan.group != 'other']
        visual_chan = visual_chan.reset_index(drop=True)
        return visual_chan

    def brain_path(self):
        """
        Return anatomical information path
        """
        subject_path = self.subject_path()
        brain_path = subject_path.joinpath('brain')
        return brain_path

#%% Return electrodes info

    def df_electrodes_info(self):
        """"
        Return electrode info as dataframe
        """
        electrodes_file = self.electrodes_file()
        df_electrodes_info = pd.read_csv(electrodes_file)
        return df_electrodes_info
    
    def electrodes_file(self):
        """
        Return path of file containging electrode information
        """
        brain_path = self.brain_path()
        electrodes_file = brain_path.joinpath('electrodes_info.csv')
        return electrodes_file

    def ROI_DK(self, picks):
        """
        Return list of DK ROI corresponding to list of picked channels 
        """
        df_electrodes_info = self.df_electrodes_info()
        ROI_DK = picks2DK(df_electrodes_info, picks)
        return ROI_DK

#%% Return list of ROI given list of channels

    def ROI_brodman(self, picks):
        """
        Return list of Brodman area corresponding to list of picked channels
        """
        df_electrodes_info = self.df_electrodes_info()
        ROI_brodman = picks2brodman(df_electrodes_info, picks)
        return ROI_brodman

    def brodman(self, chan_name):
        """
        Return Brodman area of a given channel
        """
        df_electrodes_info = self.df_electrodes_info()
        brodman = chan2brodman(df_electrodes_info, chan_name)
        return brodman

    def chan2brodman(df_electrodes_info, electrode_name):
        """
        Given channel name return corresponding brodman area
        """
        brodman = df_electrodes_info['Brodman'].loc[df_electrodes_info['electrode_name']== electrode_name]
        brodman = brodman.values
        if len(brodman)==1: #known ROI
            brodman = brodman[0]
        else:
            brodman = 'unknown'
        return brodman

    def chan2DK(df_electrodes_info, electrode_name):
        """
        Given channel name, return corresponding DK ROI
        """
        DK = df_electrodes_info['ROI_DK'].loc[df_electrodes_info['electrode_name']== electrode_name]
        DK = DK.values
        if len(DK)==1: #known ROI
            DK = DK[0]
        else:
            DK = 'unknown'
        return DK
    
    def picks2brodman(df_electrodes_info, picks):
        """
        Given list of channels return corresponding list of brodman area
        """
        ROI_brodman = []
        for chan in picks:
            brodman = chan2brodman(df_electrodes_info, chan)
            ROI_brodman.append(brodman)
        return ROI_brodman
    
    def picks2DK(df_electrodes_info, picks):
        """
        Given list of channels return corresponding list of DK area
        """
        ROI_DK = []
        for chan in picks:
            DK = chan2DK(df_electrodes_info, chan)
            ROI_DK.append(DK)
        return ROI_DK

#%% Helper path functions.
# Note that user migt want to modify path appropriate to local machine

def cifar_ieeg_path(home='~'):
    """
    Return CIFAR data path
    """
    home = Path(home).expanduser()
    ieeg_path = home.joinpath('projects', 'CIFAR', 'CIFAR_data', 'iEEG_10')
    return ieeg_path 

def all_visual_channels_path(home='~'):
    """
    Return table containing all visual electrodes path
    """
    all_visual_channels_path = cifar_ieeg_path()
    all_visual_channels_path = all_visual_channels_path.joinpath('visual_electrodes.csv')
    return all_visual_channels_path

def cifar_cohort_path(home='~'):
    """
    Return subject path
    """
    ieeg_path = cifar_ieeg_path(home)
    cohort_path = ieeg_path.joinpath('subjects')
    return cohort_path

# %% Extract hfb envelope

class Hfb:
    """
    Class for HFB extraction
    """
    def __init__(self, l_freq=60.0, nband=6, band_size=20.0, l_trans_bandwidth= 10.0,
                h_trans_bandwidth= 10.0, filter_length='auto', phase='minimum',
                fir_window='blackman'):
        self.l_freq = l_freq
        self.nband = nband
        self.band_size = band_size
        self.l_trans_bandwidth = l_trans_bandwidth
        self.h_trans_bandwidth = h_trans_bandwidth
        self.filter_length = filter_length
        self.phase = phase
        self.fir_window = fir_window

    def extract_hfb(self, raw):
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
        bands = self.freq_bands()
        hfb = np.zeros(shape=(nchan, nobs))
        mean_amplitude = np.zeros(shape=(nchan,))
        
        for band in bands:
            # extract band specific envelope
            envelope = self.extract_envelope(raw)
        # hfb is weighted average of bands specific envelope over high gamma
            env_norm = self.mean_normalise(envelope)
            hfb += env_norm
            mean_amplitude += np.mean(envelope, axis=1)
        hfb = hfb/self.nband
        # Convert hfb in volts
        hfb = hfb * mean_amplitude[:,np.newaxis]
        hfb = np.nan_to_num(hfb) # convert NaN to 0
        # Create Raw object for further MNE processing
        hfb = mne.io.RawArray(hfb, raw.info)
        hfb.set_annotations(raw.annotations)
        return hfb

    def freq_bands(self):
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
        bands = [self.l_freq + i * self.band_size for i in range(0, self.nband)]
        return bands

    def extract_envelope(self, raw):
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
        raw_band = raw.copy().filter(l_freq=self.l_freq, h_freq=self.l_freq+self.band_size,
                                     phase=self.phase, filter_length=self.filter_length,
                                     l_trans_bandwidth= self.l_trans_bandwidth, 
                                     h_trans_bandwidth= self.h_trans_bandwidth,
                                         fir_window=self.fir_window)
        envelope = raw_band.copy().apply_hilbert(envelope=True).get_data()
        return envelope

    def mean_normalise(self, envelope):
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

class Hfb_db(Hfb):
    """
    Class for HFB rescaling and dB transform HFB
    """
    def __init__(self, t_prestim=-0.5, t_postim = 1.75, baseline=None,
                 preload=True, tmin=-0.4, tmax=-0.1, mode='logratio'):
        super().__init__(t_prestim=-0.5, t_postim = 1.75, baseline=None,
                         preload=True, tmin=-0.4, tmax=-0.1, mode='logratio')
        self.t_prestim = t_prestim
        self.t_postim = t_postim
        self.baselin = baseline
        self.preload = preload
        self.tmin = tmin
        self.tmax = tmax
        self.mode = mode

    def raw_to_hfb_db(self, raw):
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
        hfb = self.extract_hfb(raw)
        epochs = self.epoch_hfb(hfb)
        hfb_db = self.db_transform(epochs)
        return hfb_db


    def epoch_hfb(self, hfb):
        """
        Epoch stimulus condition hfb using MNE Epochs function
        """
        events, event_id = mne.events_from_annotations(hfb) 
        epochs = mne.Epochs(hfb, events, event_id= event_id, tmin=self.t_prestim, 
                        tmax=self.t_postim, baseline=self.baseline,preload=self.preload)
        return epochs


    def db_transform(self, epochs):
        """
        Normalise hfb with pre stimulus baseline and log transform for result in dB
        Allows for cross channel comparison via a single scale.
        """
        events = epochs.events
        event_id = epochs.event_id
        # Drop boundary event for compatibility. This does not affect results.
        # del event_id['boundary'] 
        A = epochs.get_data()
        times = epochs.times
        # db transform
        A = 10*mne.baseline.rescale(A,times,baseline=(self.tmin,self.tmax),
                                    mode=self.mode)
        # Create epoch object from array
        hfb = mne.EpochsArray(A, epochs.info, events=events, 
                                 event_id=event_id, tmin=self.t_prestim)
        return hfb


    def extract_baseline(self, epochs):
        """
        Extract baseline by averaging prestimulus accross time and trials. From 
        testing, it does not differs much to MNE baseline.rescale, so might as well
        use MNE
        """
        baseline = epochs.copy().crop(tmin=self.tmin, tmax=self.tmax) # Extract prestimulus baseline
        baseline = baseline.get_data()
        baseline = np.mean(baseline, axis=(0,2)) # average over time and trials
        return baseline 


#%% Detect visually responsive populations

class Visual_response:
    """
    Detect visually responsive channels
    """
    def __init__(self, tmin_prestim=-0.4, tmax_prestim=-0.1, tmin_postim=-0.1,
               tmax_postim=0.5, alpha=0.05, zero_method='pratt',
               alternative='two-sided'):
        self.tmin_prestim = tmin_prestim
        self.tmax_prestim = tmax_prestim
        self.tim_postim = tim_postim
        self.tmax_postim = tmax_postim
        self.alpha = alpha
        self.zero_method = zero_method
        self.alternative = alternative
        
    def detect(self, hfb_db):
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
        A_prestim = self.crop_hfb(hfb_db, tmin=self.tmin_prestim, tmax=self.tmax_prestim)
        A_postim = self.crop_hfb(hfb_db, tmin=self.tmin_postim, tmax=self.tmax_postim)
        reject, pval_correct, tstat = self.multiple_wilcoxon_test(A_postim, A_prestim)
        visual_responsivity = self.compute_visual_responsivity(A_postim, A_prestim)
        visual_chan, effect_size = self.visual_chans_stats(reject, visual_responsivity, hfb_db)
        return visual_chan, effect_size
    
    
    def crop_hfb(self, hfb_db, tmin=-0.5, tmax=-0.05):
        """
        crop hfb between over [tmin tmax].
        Input : MNE raw object
        Return: array
        """
        A = hfb_db.copy().crop(tmin=tmin, tmax=tmax).get_data()
        return A
    
    
    def crop_stim_hfb(self, hfb_db, stim_id, tmin=-0.5, tmax=-0.05):
        """
        crop condition specific hfb between [tmin tmax].
        Input : MNE raw object
        Return: array
        """
        A = hfb_db[stim_id].copy().crop(tmin=tmin, tmax=tmax).get_data()
        return A
    
    
    def multiple_perm_test(self, A_postim, A_prestim, nchans):
        A_postim = np.mean(A_postim, axis=-1)
        A_prestim = np.mean(A_prestim, axis=-1)
        # Initialise inflated p values
        pval = [0]*nchans
        tstat = [0]*nchans
        # Compute inflated stats
        for i in range(0,nchans):
            tstat[i], pval[i] = nnstats.permtest_rel(A_postim[:,i], A_prestim[:,i])  
        # Correct for multiple testing    
        reject, pval_correct = fdrcorrection(pval, alpha=self.alpha)
        return reject
    
    
    def multiple_wilcoxon_test(self, A_postim, A_prestim):
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
                                                 zero_method=self.zero_method, 
                                                 alternative=self.alternative) 
        # Correct for multiple testing    
        reject, pval_correct = fdrcorrection(pval, alpha=self.alpha)
        w_test = reject, pval_correct, tstat
        return w_test
    
    
    def multiple_t_test(self, A_postim, A_prestim, nchans):
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
    
    
    def compute_visual_responsivity(self, A_postim, A_prestim):
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
    
    
    def visual_chans_stats(self, reject, visual_responsivity, hfb_db):
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
    A_prestim = crop_stim_hfb(visual_hfb, image_id, tmin=-0.4, tmax=-0.1)
    A_baseline = np.mean(A_prestim, axis=-1)
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
    
class Visual_function(Visual_response):
    """
    Classify visual channels into Face, Place and retinotopic channels
    """
    def __init__(self, tmin_postim=0.2, tmax_postim=0.5, alpha=0.05, 
                 zero_method='pratt'):
        super().__init__(tmin_postim=0.2, tmax_postim=0.5, alpha=0.05, 
                 zero_method='pratt')

    def face_place(self, visual_hfb, face_id, place_id, visual_channels):
        """
        Classify Face and place selective sites using one sided signed ranke wilcoxon
        test. 
        """
        nchan = len(visual_channels)
        group = ['O']*nchan
        category_selectivity = [0]*len(group)
        A_face = crop_stim_hfb(visual_hfb, face_id, tmin=self.tmin_postim, tmax=self.tmax_postim)
        A_place = crop_stim_hfb(visual_hfb, place_id, tmin=self.tmax_postim, tmax=self.tmax_postim)
        
        n_visuals = len(visual_hfb.info['ch_names'])
        w_test_plus = multiple_wilcoxon_test(A_face, A_place, zero_method=self.zero_method,
                                             alternative = 'greater', alpha=self.alpha)
        reject_plus = w_test_plus[0]
    
        
        w_test_minus = multiple_wilcoxon_test(A_face, A_place,
                                              zero_method=self.zero_method,
                                              alternative = 'less', alpha=self.alpha)
        reject_minus = w_test_minus[0]
    
        
        # Significant electrodes located outside of V1 and V2 are Face or Place responsive
        for idx, channel in enumerate(visual_channels):
            A_face = crop_stim_hfb(visual_hfb, face_id, tmin=self.tmin_postim, tmax=self.tmax_postim)
            A_place = crop_stim_hfb(visual_hfb, place_id, tmin=self.tmax_postim, tmax=self.tmax_postim)
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
    
    
    def retinotopic(self, visual_channels, group, dfelec):
        """
        Return retinotopic site from V1 and V2 given Brodman atlas.
        """
        nchan = len(group)
        bipolar_visual = [visual_channels[i].split('-') for i in range(nchan)]
        for i in range(nchan):
            brodman = (dfelec['Brodman'].loc[dfelec['electrode_name']==bipolar_visual[i][0]].to_string(index=False), 
                       dfelec['Brodman'].loc[dfelec['electrode_name']==bipolar_visual[i][1]].to_string(index=False))
            if ' V1' in brodman or ' V2' in brodman and group[i]!='F' and  group[i] !='P':
                group[i]='R'
        return group
    
    def extract_stim_id(event_id, cat = 'Face'):
        """
        Returns event id of specific stimuli category (Face or Place)
        """
        p = re.compile(cat)
        stim_id = []
        for key in event_id.keys():
            if p.match(key):
                stim_id.append(key)
        return stim_id

#%%

def compute_peak_time(hfb, visual_chan, tmin=0.05, tmax=1.75):
    """
    Return time of peak amplitude for each visual channel
    """
    nchan = len(visual_chan)
    peak_time = [0] * nchan
    hfb = hfb.copy().pick_channels(visual_chan)
    hfb = hfb.copy().crop(tmin=tmin, tmax = tmax)
    time = hfb.times
    A = hfb.copy().get_data()
    evok = np.mean(A,axis=0)
    for i in range(nchan):
        peak = np.amax(evok[i,:])
        peak_sample = np.where(evok[i,:]==peak)
        peak_sample = peak_sample[0][0]
        peak_time[i] = time[peak_sample]
    return peak_time
#%% Return all relevant informaiton for visually repsonsive populations

class Visual_sites(Visual_function):
    
     def __init__(self, tmin_prestim=-0.4, tmax_prestim=-0.1, tmin_postim=0.1,
               tmax_postim=0.5, alpha=0.05, zero_method='pratt',
               alternative='two-sided'):
        super().__init__(tmin_prestim=-0.4, tmax_prestim=-0.1, tmin_postim=0.1,
               tmax_postim=0.5, alpha=0.05, zero_method='pratt',
               alternative='two-sided')

     def hfb_to_visual_populations(self, hfb_db, dfelec):
        """
        Create dictionary containing relevant information on visually responsive channels
        """
        # Extract and normalise hfb
        event_id = hfb_db.event_id
        face_id = extract_stim_id(event_id, cat = 'Face')
        place_id = extract_stim_id(event_id, cat='Place')
        image_id = face_id+place_id
        
        # Detect visual channels
        visual_chan, visual_responsivity = self.detect(hfb_db)
        
        visual_hfb = hfb_db.copy().pick_channels(visual_chan)
        
        # Compute latency response
        latency_response = compute_latency(visual_hfb, image_id, visual_chan)
        
        # Classify Face and Place populations
        group, category_selectivity = self.face_place(visual_hfb, face_id, place_id, visual_chan)
        # Classify retinotopic populations
        group = self.retinotopic(visual_chan, group, dfelec)
        
        # Compute peak time
        
        peak_time = compute_peak_time(hfb_db, visual_chan, tmin=0.05, tmax=1.75)
        
        # Create visual_populations dictionary 
        visual_populations = {'chan_name': [], 'group': [], 'latency': [], 
                              'brodman': [], 'DK': [], 'X':[], 'Y':[], 'Z':[]}
        
        visual_populations['chan_name'] = visual_chan
        visual_populations['group'] = group
        visual_populations['latency'] = latency_response
        visual_populations['visual_responsivity'] = visual_responsivity
        visual_populations['category_selectivity'] = category_selectivity
        visual_populations['peak_time'] = peak_time
        for chan in visual_chan: 
            chan_name_split = chan.split('-')[0]
            visual_populations['brodman'].extend(dfelec['Brodman'].loc[dfelec['electrode_name']==chan_name_split])
            visual_populations['DK'].extend(dfelec['ROI_DK'].loc[dfelec['electrode_name']==chan_name_split])
            visual_populations['X'].extend(dfelec['X'].loc[dfelec['electrode_name']==chan_name_split])
            visual_populations['Y'].extend(dfelec['Y'].loc[dfelec['electrode_name']==chan_name_split])
            visual_populations['Z'].extend(dfelec['Z'].loc[dfelec['electrode_name']==chan_name_split])
            
        return visual_populations


# %% Create category specific time series as input for mvgc toolbox

def category_ts(hfb, visual_chan, sfreq=250, tmin_crop=0.050, tmax_crop=0.250):
    """
    Return time series in all conditions ready for mvgc analysis
    ----------
    Parameters
    ----------
    visual_chan : list
                List of visually responsive channels
    """
    categories = ['Rest', 'Face', 'Place']
    ncat = len(categories)
    ts = [0]*ncat
    for idx, cat in enumerate(categories):
        X, time = hfb_to_category_time_series(hfb, visual_chan, sfreq=sfreq, cat=cat, 
                                        tmin_crop=tmin_crop, tmax_crop=tmax_crop)
        ts[idx] = X

    ts = np.stack(ts)
    (ncat, ntrial, nchan, nobs) = ts.shape
    ts = np.transpose(ts, (2, 3, 1, 0))
    return ts, time

def category_lfp(lfp, visual_chan, tmin_crop=-0.5, tmax_crop =1.75, sfreq=500):
    """
    Return ieeg time series in all conditions ready for mvgc analysis
    ----------
    Parameters
    ----------
    visual_chan : list
                List of visually responsive channels
    """
    categories = ['Rest', 'Face', 'Place']
    ncat = len(categories)
    ts = [0]*ncat
    for idx, cat in enumerate(categories):
        epochs, events = epoch_category(lfp, cat=cat, tmin=tmin_crop, tmax=tmax_crop)
        epochs = epochs.resample(sfreq=sfreq)
        time = epochs.times
        sorted_indices = sort_indices(epochs, visual_chan)
        X = sort_visual_chan(sorted_indices, epochs)
        ts[idx] = X
    
    ts = np.stack(ts)
    (ncat, ntrial, nchan, nobs) = ts.shape
    ts = np.transpose(ts, (2, 3, 1, 0))
    return ts, time


def load_visual_hfb(sub_id= 'DiAs', proc= 'preproc', 
                            stage= '_BP_montage_hfb_raw.fif'):
    """
    Load visual hfb and visual channels
    """
    subject = Subject(name=sub_id)
    raw = subject.load_data(proc= proc, stage= stage)
    visual_chan = subject.pick_visual_chan()
    visual_chan = visual_chan['chan_name'].values.tolist()
    hfb_visual = raw.copy().pick_channels(visual_chan)
    return hfb_visual, visual_chan


def pick_visual_chan(picks, visual_chan):
    """
    Pick specific visual channels from visual channels
    """
    drop_index = []
    
    for chan in visual_chan['chan_name'].to_list():
        if chan in picks:
            continue
        else:
            drop_index.extend(visual_chan.loc[visual_chan['chan_name']==chan].index.tolist())
    
    visual_chan = visual_chan.drop(drop_index)
    visual_chan = visual_chan.reset_index(drop=True)
    return visual_chan


def hfb_to_category_time_series(hfb, visual_chan, sfreq=250, cat='Rest', tmin_crop = 0.5, tmax_crop=1.75):
    """
    Return category visual time series cropped in a time interval [tmin_crop tmax_crop]
    of interest, resampled
    """
    hfb = category_hfb(hfb, cat=cat, tmin_crop = tmin_crop, tmax_crop=tmax_crop)
    hfb = hfb.resample(sfreq=sfreq)
    time = hfb.times
    sorted_indices = sort_indices(hfb, visual_chan)
    X = sort_visual_chan(sorted_indices, hfb)
    return X, time


def category_hfb(hfb_visual, cat='Rest', tmin_crop = -0.5, tmax_crop=1.75) :
    """
    Return category visual time hfb_db cropped in a time interval [tmin_crop tmax_crop]
    of interest
    """
    epochs, events = epoch_category(hfb_visual, cat=cat, tmin=-0.5, tmax=1.75)
    hfb = db_transform(epochs, tmin=-0.4, tmax=-0.1, t_prestim = -0.5, mode='logratio')
    hfb = hfb.crop(tmin=tmin_crop, tmax=tmax_crop)
    return hfb


def sort_visual_chan(sorted_indices, hfb):
    """
    Order visual hfb channels indices along visual herarchy (Y coordinate)
    """
    X = hfb.get_data()
    X_ordered = np.zeros_like(X)
    for idx, i in enumerate(sorted_indices):
            X_ordered[:,idx,:] = X[:,i,:]
    X = X_ordered
    return X


def sort_indices(hfb, visual_chan):
    """
    Order channel indices along visual hierarchy
    """
    unsorted_chan = hfb.info['ch_names']
    
    sorted_indices = [0]*len(visual_chan)
    for idx, chan in enumerate(unsorted_chan):
        sorted_indices[idx] = visual_chan.index(chan)
    return sorted_indices


def epoch_category(hfb_visual, cat='Rest', tmin=-0.5, tmax=1.75):
    """
    Epoch category specific hfb
    """
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


def parcellation_to_indices(visual_population, parcellation='group'):
    """
    Return indices of channels from a given population
    parcellation: group (default, functional), DK (anatomical)
    """
    group = visual_population[parcellation].unique().tolist()
    group_indices = dict.fromkeys(group)
    for key in group:
       group_indices[key] = visual_population.loc[visual_population[parcellation]== key].index.to_list()
    if parcellation == 'DK': # adapt indexing for matlab
        for key in group:
            for i in range(len(group_indices[key])):
                group_indices[key][i] = group_indices[key][i]
    return group_indices

#%% Return population specifc hfb

def ts_to_population_hfb(ts, visual_populations, parcellation='group'):
    """
    Return hfb of each population from category specific time series.
    """
    (nchan, nobs, ntrials, ncat) = ts.shape
    populations_indices = parcellation_to_indices(visual_populations,
                                                     parcellation=parcellation)
    populations = populations_indices.keys()
    npop = len(populations)
    population_hfb = np.zeros((npop, nobs, ntrials, ncat))
    for ipop, pop in enumerate(populations):
        pop_indices = populations_indices[pop]
        # population hfb is average of hfb over each population-specific channel
        population_hfb[ipop,...] = np.average(ts[pop_indices,...], axis=0)
    # Return list of populations to keep track of population ordering
    populations = list(populations)
    return population_hfb, populations

#%% Create category time series with specific channels

def chan_specific_category_ts(picks, proc='preproc', stage='_BP_montage_HFB_raw.fif', 
                     sub_id='DiAs', sfreq=250, tmin_crop=0, tmax_crop=1.75):
    """
    Create category time series with specific channels
    """
    subject = Subject(sub_id)
    visual_populations = subject.pick_visual_chan()
    hfb, visual_chan = load_visual_hfb(sub_id= sub_id, proc= proc, 
                                stage= stage)
    hfb = hfb.pick_channels(picks)
    
    ts, time = category_ts(hfb, picks, sfreq=sfreq, tmin_crop=tmin_crop,
                              tmax_crop=tmax_crop)
    return ts, time

def chan_specific_category_lfp(picks, proc='preproc', stage='_BP_montage_preprocessed_raw.fif', 
                     sub_id='DiAs', sfreq=250, tmin_crop=0, tmax_crop=1.75):
    """
    Create category specific LFP time series with specific channels
    """
    subject = Subject(sub_id)
    visual_populations = subject.pick_visual_chan()
    lfp, visual_chan = load_visual_hfb(sub_id= sub_id, proc= proc, 
                                stage= stage)
    lfp = lfp.pick_channels(picks)
    
    ts, time = category_lfp(lfp, picks, sfreq=sfreq, tmin_crop=tmin_crop,
                              tmax_crop=tmax_crop)
    return ts, time

# %% Substract average event related amplitude

def substract_AERA(ts, axis=2):
    """
    Substract the average event related amplitude for stimuli conditions. 
    This is useful for stationarity
    and gaussianity. Similar to detrending (remove trend due to transient increase)
    upon stimuli presentation.
    Pb: Maybe this remove too much interesting signal?
    """
    ntrials = ts.shape[axis]
    # Remove AERA on Face and Place conditions only 
    cat = [1,2]
    average_event_related_amplitude = np.average(ts, axis=axis)
    for j in cat:
        for i in range(ntrials):
            ts[:,:,i,j] -= average_event_related_amplitude[:,:,j]
    return ts

def plot_trials(ts, time, ichan=1, icat=1, label='raw'):
    """
    Plot individual trials of a single channel.
    """
    sns.set()
    ntrials = ts.shape[2]
    for i in range(ntrials):
        plt.subplot(7,8,i+1)
        plt.plot(time, ts[ichan,:,i,icat], label=label)
        plt.axis('off')

#%% Cross subject functions

def cross_subject_ts(subjects, proc='preproc', stage= '_BP_montage_HFB_raw.fif',
                     sfreq=250, tmin_crop=0.50, tmax_crop=0.6):
    """
    Return cross subject time series in each condition
    ----------
    Parameters
    ----------
    
    """
    ts = [0]*len(subjects)
    for s in range(len(subjects)):
        sub_id = subjects[s]  
        subject = Subject(name=sub_id)
        datadir = subject.processing_stage_path(proc=proc)
        visual_chan = subject.pick_visual_chan()
        hfb, visual_chan = load_visual_hfb(sub_id = sub_id, proc= proc, 
                                stage= stage)
        ts[s], time = category_ts(hfb, visual_chan, sfreq=sfreq, 
                                  tmin_crop=tmin_crop, tmax_crop=tmax_crop)
        # Beware might raise an error if shape don't match along axis !=0
        # cross_ts = np.concatenate(ts, axis=0)
    return ts, time

def chanel_statistics(cross_ts, nbin=30, fontscale=1.6):
    """
    Plot skewness and kurtosis from cross channels time series to estimate
    deviation from gaussianity.
    """
    (n, m, N, c) = cross_ts.shape
    new_shape = (n, m*N, c)
    X = np.reshape(cross_ts, new_shape)
    skewness = np.zeros((n,c))
    kurtosis = np.zeros((n,c))
    for i in range(n):
        for j in range(c):
            a = X[i,:,j]
            skewness[i,j] = stats.skew(a)
            kurtosis[i,j] = stats.kurtosis(a)
    # Plot skewness, kurtosis
    categories = ['Rest', 'Face', 'Place']
    skew_xticks = np.arange(-1,1,0.5)
    kurto_xticks = np.arange(0,5,1)
    sns.set(font_scale=fontscale)
    f, ax = plt.subplots(2,3)
    for i in range(c):
        ax[0,i].hist(skewness[:,i], bins=nbin, density=False)
        ax[0,i].set_xlim(left=-1, right=1)
        ax[0,i].set_ylim(bottom=0, top=35)
        ax[0,i].xaxis.set_ticks(skew_xticks)
        ax[0,i].axvline(x=-0.5, color='k')
        ax[0,i].axvline(x=0.5, color='k')
        ax[0,i].set_xlabel(f'Skewness ({categories[i]})')
        ax[0,i].set_ylabel('Number of channels')
        ax[1,i].hist(kurtosis[:,i], bins=nbin, density=False)
        ax[1,i].set_xlim(left=0, right=5)
        ax[1,i].set_ylim(bottom=0, top=60)
        ax[1,i].axvline(x=1, color='k')
        ax[1,i].xaxis.set_ticks(kurto_xticks)
        ax[1,i].set_xlabel(f'Excess kurtosis ({categories[i]})')
        ax[1,i].set_ylabel('Number of channels')
    return skewness, kurtosis

#%% Sliding window analysis

def sliding_ts(picks, proc='preproc', stage='_BP_montage_HFB_raw.fif', sub_id='DiAs',
               tmin=0, tmax=1.75, win_size=0.2, step = 0.050, detrend=True, sfreq=250):
    """
    Return sliced hfb into win_size time window between tmin and tmax,
    overlap determined by parameter step
    """
    window = list_window(tmin=tmin, tmax=tmax, win_size=win_size, step=step)
    nwin = len(window)
    ts = [0]*nwin
    time = [0]*nwin
    for i in range(nwin):
        tmin_crop=window[i][0]
        tmax_crop=window[i][1]
        ts[i], time[i] = chan_specific_category_ts(picks, sub_id= sub_id, proc= proc, 
                                                      stage= stage, tmin_crop=tmin_crop, 
                                                      tmax_crop=tmax_crop)
        if detrend==True:
            ts[i] = substract_AERA(ts[i], axis=2)
        else:
            continue
    ts = np.stack(ts, axis=3)
    time = np.stack(time, axis=-1)
    return ts, time

def sliding_lfp(picks, proc='preproc', stage='_BP_montage_preprocessed_raw.fif', sub_id='DiAs',
               tmin=0, tmax=1.75, win_size=0.2, step = 0.050, detrend=True, sfreq=250):
    """
    Return sliced lfp into win_size time window between tmin and tmax,
    overlap determined by parameter step
    """
    window = list_window(tmin=tmin, tmax=tmax, win_size=win_size, step=step)
    nwin = len(window)
    ts = [0]*nwin
    time = [0]*nwin
    for i in range(nwin):
        tmin_crop=window[i][0]
        tmax_crop=window[i][1]
        ts[i], time[i] = chan_specific_category_lfp(picks, sub_id= sub_id, proc= proc, 
                                                      stage= stage, tmin_crop=tmin_crop, 
                                                      tmax_crop=tmax_crop)
        if detrend==True:
            ts[i] = substract_AERA(ts[i], axis=2)
        else:
            continue
    ts = np.stack(ts, axis=3)
    time = np.stack(time, axis=-1)
    return ts, time

def list_window(tmin=0, tmax=1.75, win_size=0.2, step=0.050):
    """
    Sliding_ts subroutine. Returns sliced windows.
    """
    nwin = int(np.floor((tmax - win_size)/step))
    win_start = [0]*nwin
    win_stop = [0]*nwin
    window = [0]*nwin
    
    for i in range(nwin):
        win_start[i] = tmin + step*i
        win_stop[i] = win_start[i] + win_size
        window[i] = (win_start[i], win_stop[i])
    return window

#%% Sliding window on continuous data

def category_continous_sliding_ts(hfb, tmin=0, tmax=265, step=1, win_size=10):
    """
    Return category specific sliding ts from continuous hfb or lfp
    """
    # Extract resting state hfb and stimulus hfb
    hfb_rest = hfb.copy().crop(tmin=60, tmax=325)
    hfb_stim = hfb.copy().crop(tmin=425, tmax=690)
    hfb_cat = [hfb_rest, hfb_stim]
    ncat = len(hfb_cat)
    ts = [0]*ncat
    # Return sliding window ts
    for i in range(ncat):
        ts[i], time = make_continuous_sliding_ts(hfb_cat[i], tmin=tmin, tmax=tmax, step=step, 
                                              win_size=win_size)
    ts = np.stack(ts, axis=-1)
    return ts, time

def make_continuous_sliding_ts(hfb, tmin=0, tmax=265, step=1, win_size=10):
    """
    Return sliding window from continuous hfb or lfp
    """
    window = list_window(tmin=tmin, tmax=tmax, win_size=win_size, step=step)
    nwin = len(window)
    ts = [0]*nwin
    time = [0]*nwin
    # List windowed hfb
    for i in range(nwin):
        tmin_crop = window[i][0]
        tmax_crop = window[i][1]
        ts[i] = hfb.copy().crop(tmin=tmin_crop, tmax=tmax_crop).get_data()
        time[i] = hfb.copy().crop(tmin=tmin_crop, tmax=tmax_crop).times
    ts = np.stack(ts, axis=-1)
    time = np.stack(time, axis=-1)
    return ts, time