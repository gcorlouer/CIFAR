#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 18:52:27 2020

@author: guime
"""


# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import h5py
import pandas as pd
from pathlib import Path
from scipy.io import savemat

connectivity_file = "2020-09-23_16-44_Connectivity.h5"
ts_file = '2020-09-23_16-44_TimeSeriesRegion.h5'
    
home = Path.home()
connectivity_path = home.joinpath('Downloads', connectivity_file)
ts_path = home.joinpath('Downloads', ts_file)

f_connectivity = h5py.File(connectivity_path, 'r')
f_ts = h5py.File(ts_path, 'r')

weights = f_connectivity['weights'] 
tracts = f_connectivity['tract_lengths'] 

X = f_ts['data']
time = f_ts['time']

X = X[:,0,:,0]

fsave = home.joinpath('tvb_test.mat')
d = {'data': X, 'time': time, 'connectivity': weights[:,:]}

savemat(fsave, d)
    