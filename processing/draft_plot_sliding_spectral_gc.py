#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 17:46:07 2021

@author: guime
"""

import cifar_load_subject as cf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import helper_functions as fun
import HFB_process as hf
import pandas as pd

from scipy.io import loadmat

#%%

sub_id = 'DiAs'
proc = 'preproc' 
categories = ['Rest', 'Face', 'Place']
pop = ['R', 'F']

subject = cf.Subject(name=sub_id)
datadir = subject.processing_stage_path(proc=proc)
fname = sub_id + 'sliding_sgc.mat'
fpath = datadir.joinpath(fname)
GC = loadmat(fpath)

#%%

f = GC['f']
time = GC['time']
sfreq = GC['sfreq']
sfreq = sfreq[0][0]
time = time[0,:]

#%%

S = f[0,1,:,:,1]

#%%

plt.pcolormesh(S,  shading='gouraud', cmap='magma', vmin=0, vmax=0.010)