#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:32:06 2020

@author: guime
"""

import pandas as pd
import os 

from pathlib import Path, PurePath
from shutil import copy

def cf_dataset(subject='DiAs', task='stimuli', run= '1'):
    """Return raw dataset name """
    """Parameters: 
        - task: 'stimuli', 'rest_baseline', 'sleep' 
        - run : 1, 2 (run in the experiment) """
    dataset = [subid, "freerecall", task, run, 'preprocessed']
    dataset = "_".join(dataset)
    return dataset

def cf_dataset_preproc(dataset, preproc='raw_signal'):
    """Return data set at some preprocessed stage"""
    dataset = dataset+preproc
    return dataset 

def cf_ieeg_path(home='~'):
    home = Path(home).expanduser()
    ieeg_path = home.joinpath('CIFAR_data', 'iEEG_10')
    return ieeg_path 

def cf_cohort_path(home='~'):
    ieeg_path = cf_ieeg_path(home)
    cohort_path = ieeg_path.joinpath('subjects')
    return cohort_path

def cf_subject_path(subject='DiAs'):
    """Return path of the subject"""    
    cohort_path = cf_cohort_path() # home directory
    subject_path = cohort_path.joinpath(subject)
    return subject_path

def cf_anatpath(subject='DiAS'):

    subject_path = cf_subject_path(subject=subject)
    anatpath = subject_path.joinpath('brain')
    return anatpath

def cf_elecpath(subject='DiAs'):
    
    anatpath = cf_anatpath(subject=subject)
    elecfile = anatpath.joinpath('electrodes_info.csv')
    return elecfile

def cf_proc_path(home=Path('~').expanduser, subject='DiAs', preproc='raw_signal'):
    """Return path at some preprocessed stage"""
    subpath = home.joinpath('CIFAR_data', 'iEEG_10','subjects', subject)
    cf_path = subpath.joinpath('EEGLAB_dataset', preproc)
    return cf_path



