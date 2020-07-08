#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:32:06 2020

@author: guime
"""
# TODO: extract envelope, channels info.
    
import pandas as pd
import os 
import mne 

from pathlib import Path, PurePath
from shutil import copy

def cf_ieeg_path(home='~'):
    home = Path(home).expanduser()
    ieeg_path = home.joinpath('CIFAR_data', 'iEEG_10')
    return ieeg_path 

def visual_path(home='~'):
    path_visual = cf_ieeg_path()
    path_visual = path_visual.joinpath('visual_electrodes_1.csv')
    return path_visual

def cf_cohort_path(home='~'):
    ieeg_path = cf_ieeg_path(home)
    cohort_path = ieeg_path.joinpath('subjects')
    return cohort_path

def chan2brodman(dfelec, electrode_name):
        brodman = dfelec['Brodman'].loc[dfelec['electrode_name']== electrode_name]
        brodman = brodman.values
        if len(brodman)==1: #known ROI
            brodman = brodman[0]
        else:
            brodman = 'Unknown'
        return brodman

def picks2ROIs(dfelec, picks):
    ROIs = []
    for chan in picks:
        brodman = chan2brodman(dfelec, chan)
        ROIs.append(brodman)
    return ROIs
    
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
        cohort_path = cf_cohort_path() # home directory
        subject_path = cohort_path.joinpath(self.name)
        return subject_path

    def anatpath(self):
        subject_path = self.subject_path()
        anatpath = subject_path.joinpath('brain')
        return anatpath
    
    def elecfile(self):
        anatpath = self.anatpath()
        elecfile = anatpath.joinpath('electrodes_info.csv')
        return elecfile
    
    def dfelec(self):
        """"Return electrode info as dataframe"""
        elecfile = self.elecfile()
        dfelec = pd.read_csv(elecfile)
        return dfelec
        
    def procpath(self, proc='raw_signal'):
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

    def fname(self, suffix='', ext='.set'):
        dataset = self.dataset(suffix)
        fname = dataset+ext
        return fname
    
    def fpath(self, proc='raw_signal', suffix='', ext='.set'):
        procpath = self.procpath(proc)
        fname = self.fname(suffix=suffix, ext=ext)
        fpath = procpath.joinpath(fname)
        fpath = os.fspath(fpath)
        return fpath
    
    def import_data(self, fpath, preload=True):
        raw = mne.io.read_raw_eeglab(fpath, preload=preload)
        return raw
    
    def brodman(self, chan_name):
        dfelec = self.dfelec()
        brodman = chan2brodman(dfelec, chan_name)
        return brodman
    
    def ROIs(self, picks):
        dfelec = self.dfelec()
        ROIs = picks2ROIs(dfelec, picks)
        return ROIs

    









