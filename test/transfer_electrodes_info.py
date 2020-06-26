#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:31:02 2020

@author: guime
"""
# Copy electrodes info

def elec_src(subid='0'):
    
    elec_src = f'electrodes_info_subject_{subid}.csv'
    return elec_src

def elecpath_src(elec_src, home='~'):
    home = Path(home).expanduser()
    elecpath_src = home.joinpath('CIFAR_electrodes_info')
    elecpath_src = elecpath_src.joinpath(elec_src)
    return elecpath_src


def cp_elecinfo(subid='0', subject='AnRa'):
    
    elecfile_src = elec_src(subid)
    elecfile_src = elecpath_src(elecfile_src)
    elecfile_dest = cf_elecpath(subject)
    copy(elecfile_src, elecfile_dest)

subjects = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']

for idx, subject in enumerate(subjects):
    cp_elecinfo(subid=idx, subject=subject)

for subject in subjects:
    anatpath = cf_anatpath(subject)
    elecfile = cf_elecpath(subject)
    df = pd.read_csv(elecfile)
    df = df.drop('Unnamed: 0',axis='columns')
    df = df.drop('Unnamed: 0.1',axis='columns')
    df.to_csv(elecfile)
