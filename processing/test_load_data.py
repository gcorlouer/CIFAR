#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 17:02:35 2021

@author: guime
"""

import mne
import os
import pandas as pd

from pathlib import Path, PurePath

cohort_path = Path('~','projects', 'CIFAR', 'CIFAR_data', 'iEEG_10', 
                   'subjects').expanduser()
print(cohort_path)
class Subject:
    """
    Class subject
    """
    def __init__(self, cohort_path, name='DiAs', task='stimuli', run='1', proc='preproc',
                 stage='_BP_montage_HFB_raw.fif', preload=True, epoch=False):
        """
        Parameters:
            - name: name of subject
            - task: 'stimuli', 'rest_baseline', 'sleep' 
            - run : 1, 2  
        """
        self.name = name
        self.task = task
        self.run = run
        self.proc = proc
        self.stage = stage
        self.preload=preload
        self.epoch = epoch

    def read_dataset(self):
        
        subject_path = cohort_path.joinpath(self.name)
        proc_path = subject_path.joinpath('EEGLAB_datasets', self.proc)
        if self.proc == 'preproc':
            fname = self.name + self.stage
            fpath = proc_path.joinpath(fname)
            if self.epoch==False:
                raw = mne.io.read_raw_fif(fpath, preload=self.preload)
            else:
                raw = mne.read_epochs(fpath, preload=self.preload)
        else:
            fname = [self.name, "freerecall", self.task, self.run, 'preprocessed']
            if self.proc == 'bipolar_montage':
                fname.append('BP_montage')
            fname = "_".join(fname)
            fname = fname + '.set'
            fpath = proc_path.joinpath(fname)
            raw = mne.io.read_raw_eeglab(fpath, preload=self.preload)
        return raw

    def read_channels_info(self, fname='electrodes_info.csv'):
        """
        Read subject specific channels information into a dataframe. 
        Channels info can be contain the subject sites or visually responsive 
        channels
        ----------
        Parameters
        ----------
        fname: file name of the channels
        fname= 'electrodes_info.csv', 'visual_BP_channels.csv'
        Note:
        If user wants to read visually responsive channels from all subjects in
        one table, look up 'visual_electrodes.csv' file in /iEEG_10 path.
        """
        subject_path = cohort_path.joinpath(self.name)
        brain_path = subject_path.joinpath('brain')
        channel_path = brain_path.joinpath(fname)
        channel_info = pd.read_csv(channel_path)
        return channel_info
    
#%% Test read data

subject = Subject(cohort_path, proc='bipolar_montage')
raw = subject.read_dataset()

#%% Test read channels info

subject = Subject(cohort_path, proc='bipolar_montage')
channels_info = subject.read_channels_info(fname='electrodes_info.csv')
