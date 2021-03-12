#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 08:50:47 2021

@author: guime
"""


import cifar_load_subject as cf
import pandas as pd
import numpy as np


subjects = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
#%%

for sub_id in subjects:

    subject = cf.Subject(name=sub_id)
    HFB = subject.load_raw_data()
    chan_BP = HFB.info['ch_names']
    dfelec = subject.df_electrodes_info()
    columns = dfelec.columns
    nchan = len(chan_BP)
    #%%
    
    df_BP = pd.DataFrame(index=np.arange(nchan), columns=columns)
    for idx, chan in enumerate(chan_BP):
        chan_name_split = chan.split('-')[0]
        for col in columns:
            df_BP[col].iloc[idx] = dfelec[col].loc[dfelec['electrode_name']==chan_name_split].values[0]
    
    df_BP['electrode_name'] = chan_BP
    #%%
    
    df_BP = df_BP.sort_values(by='Y')
    
    #%%
    
    brainpath  = subject.brain_path()
    fname = 'BP_channels.csv'
    fpath = brainpath.joinpath(fname)
    df_BP.to_csv(fpath, index='False')
