#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:10:10 2021

@author: guime
"""
import cifar_load_subject as cf

#%%
sub_id = 'DiAs'
proc = 'preproc'
fname = sub_id + '_BP_montage_preprocessed_raw.fif'

subject = cf.Subject()
df = subject.df_electrodes_info()
df = df.sort_values(by= 'Y')

#%%