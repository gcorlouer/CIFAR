#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:31:02 2020

@author: guime
"""
# Copy electrodes info
import pandas as pd
import os 
import mne 

from pathlib import Path, PurePath
from shutil import copy
import cf_load

def elec_src(subid='0'):
    path = Path('~').expanduser()
    path = path.joinpath('projects','CIFAR','data_bids', f'sub-0{subid}', 'anat')
    file = 'electrodes_info.csv'
    fpath = path.joinpath(file)
    #elec_src = f'electrodes_info_subject_{subid}.csv'
    return fpath

def elecpath_src(elec_src, home='~'):
    home = Path(home).expanduser()
    elecpath_src = home.joinpath('CIFAR_electrodes_info')
    elecpath_src = elecpath_src.joinpath(elec_src)
    return elecpath_src


def cp_elecinfo(subid, subject):
    subject = cf_load.Subject(subject)
    elecfile_src = elec_src(subid)
    elecfile_dest = subject.elecfile()
    copy(elecfile_src, elecfile_dest)

cf_subjects = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
subjects_bids = [0, 1, 2, 3, 4, 5, 6, 7, 8]

for idx, subject in enumerate(cf_subjects):
    cp_elecinfo(subid=idx, subject=subject)

for sub in cf_subjects:
    subject = cf_load.Subject(sub)
    elecfile = subject.elecfile()
    df = pd.read_csv(elecfile)
    df = df.drop('Unnamed: 0',axis='columns')
    df.to_csv(elecfile, index= False)

brodman = cf_load.chan2brodman(dfelec, electrode_name='Lo9')
picks = dfelec['electrode_name']
ROIs = cf_load.picks2ROIs(dfelec, picks)