#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:32:06 2020

@author: guime
"""
#     
# Import relevant libraries
import pandas as pd
import os 
import mne 

from pathlib import Path, PurePath
from shutil import copy

# Define some useful functions that will be used in subject class
def cifar_ieeg_path(home='~'):
    home = Path(home).expanduser()
    ieeg_path = home.joinpath('projects', 'CIFAR', 'CIFAR_data', 'iEEG_10')
    return ieeg_path 

def all_visual_channels_path(home='~'):
    """Return all visual channels path"""
    all_visual_channels_path = cifar_ieeg_path()
    all_visual_channels_path = all_visual_channels_path.joinpath('visual_electrodes.csv')
    return all_visual_channels_path

def cifar_cohort_path(home='~'):
    ieeg_path = cifar_ieeg_path(home)
    cohort_path = ieeg_path.joinpath('subjects')
    return cohort_path

def chan2brodman(df_electrodes_info, electrode_name):
        brodman = df_electrodes_info['Brodman'].loc[df_electrodes_info['electrode_name']== electrode_name]
        brodman = brodman.values
        if len(brodman)==1: #known ROI
            brodman = brodman[0]
        else:
            brodman = 'unknown'
        return brodman

def chan2DK(df_electrodes_info, electrode_name):
    DK = df_electrodes_info['ROI_DK'].loc[df_electrodes_info['electrode_name']== electrode_name]
    DK = DK.values
    if len(DK)==1: #known ROI
        DK = DK[0]
    else:
        DK = 'unknown'
    return DK

def picks2brodman(df_electrodes_info, picks):
    ROI_brodman = []
    for chan in picks:
        brodman = chan2brodman(df_electrodes_info, chan)
        ROI_brodman.append(brodman)
    return ROI_brodman

def picks2DK(df_electrodes_info, picks):
    ROI_DK = []
    for chan in picks:
        DK = chan2DK(df_electrodes_info, chan)
        ROI_DK.append(DK)
    return ROI_DK

# Create subject class
class Subject:
    
    def __init__(self, name='DiAs', task='stimuli', run='1'):
        """Parameters: ].append(subject)
    visual_info['category'].append('Face')
    visual_info['Brodman'].append(chan2brodman(chan))

            - task: 'stimuli', 'rest_baseline', 'sleep' 
            - run : 1, 2 (run in the experiment) """
        self.name = name
        self.task = task
        self.run = run 
        
    def subject_path(self):
        """Return path of the subject"""    
        cohort_path = cifar_cohort_path() # home directory
        subject_path = cohort_path.joinpath(self.name)
        return subject_path

    def brain_path(self):
        """Return path were brain anatomical info are stored"""
        subject_path = self.subject_path()
        brain_path = subject_path.joinpath('brain')
        return brain_path
    
    def electrodes_file(self):
        """Return path of electrodes infos csv file"""
        brain_path = self.brain_path()
        electrodes_file = brain_path.joinpath('electrodes_info.csv')
        return electrodes_file
    
    def df_electrodes_info(self):
        """"Return electrode info as dataframe"""
        electrodes_file = self.electrodes_file()
        df_electrodes_info = pd.read_csv(electrodes_file)
        return df_electrodes_info
        
    def processing_stage_path(self, proc='raw_signal'):
        """Return data path at some processed stage"""
        subject_path = self.subject_path()
        proc_path = subject_path.joinpath('EEGLAB_datasets', proc)
        return proc_path 
    
    def dataset(self, suffix=''):
        """Return  dataset name """
        dataset = [self.name, "freerecall", self.task, self.run, 'preprocessed']
        if suffix  == '':
            dataset = dataset
        else:
            dataset = dataset +[suffix]
        dataset = "_".join(dataset)
        return dataset

    def dataset_ext(self, suffix='', ext='.set'):
        """Return  dataset name  with extension"""
        dataset = self.dataset(suffix)
        dataset_ext = dataset+ext
        return dataset_ext
    
    def dataset_path(self, proc='raw_signal', suffix='', ext='.set'):
        processing_stage_path = self.processing_stage_path(proc)
        dataset_ext = self.dataset_ext(suffix=suffix, ext=ext)
        dataset_path = processing_stage_path.joinpath(dataset_ext)
        dataset_path = os.fspath(dataset_path)
        return dataset_path
    
    
    def read_eeglab(self, proc= 'raw_signal', suffix = '', ext='.set',preload=True):
    
        fpath = self.dataset_path(proc = proc, suffix=suffix)
        raw = mne.io.read_raw_eeglab(fpath, preload=preload)
        
        return raw
    
    def load_data(self, proc= 'preproc', stage= '_BP_montage_HFB_raw.fif',
                      preload=True, epo=False):  
        datadir = self.processing_stage_path(proc=proc)
        sub = self.name
        fname = sub + stage
        fpath = datadir.joinpath(fname)
        if epo==False:
            data = mne.io.read_raw_fif(fpath, preload=preload)
        else:
            data = mne.read_epochs(fpath, preload=preload)
        return data 

    
    def brodman(self, chan_name):
        """Return Brodman area of a given channel"""
        df_electrodes_info = self.df_electrodes_info()
        brodman = chan2brodman(df_electrodes_info, chan_name)
        return brodman
    def ROI_DK(self, picks):
        """Return  ROI according of picked channels """
        df_electrodes_info = self.df_electrodes_info()
        ROI_DK = picks2DK(df_electrodes_info, picks)
        return ROI_DK
    
    def ROI_brodman(self, picks):
        """Return  Brodman area of picked channels"""
        df_electrodes_info = self.df_electrodes_info()
        ROI_brodman = picks2brodman(df_electrodes_info, picks)
        return ROI_brodman

    def pick_visual_chan(self, fname = 'visual_BP_channels.csv'):
        
        brain_path = self.brain_path()
        fpath = brain_path.joinpath(fname)
        visual_chan = pd.read_csv(fpath)
        # Not necessary but uncomment in case need sorting channels
        # visual_chan = visual_chan.sort_values(by='latency')
        return visual_chan
    
    def low_high_chan(self, fname = 'visual_channels_BP_montage.csv'):
        """Drop channels in other category to only keep retinotopic (low)  
        Face and Place selective channels (high)"""
        visual_chan = self.pick_visual_chan(fname)
        visual_chan = visual_chan[visual_chan.group != 'other']
        visual_chan = visual_chan.reset_index(drop=True)
        return visual_chan
    
    









