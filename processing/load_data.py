#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:58:14 2021

@author: guime
"""

import HFB_process as hf
import cifar_load_subject as cf
import mne

sub_id = 'DiAs'
proc = 'preproc'
fname = sub_id + '_BP_montage_preprocessed_raw.fif'
stage= '_BP_montage_HFB_raw.fif'

#%%

subject = cf.Subject()
raw = subject.load_data(proc=proc, stage = stage) 
#%%

