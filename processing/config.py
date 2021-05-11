#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 20:14:06 2021
Config file, contain all parameters for analysis
@author: guime
"""
import argparse

#%% Loading data parameters
subjects = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']

parser = argparse.ArgumentParser()
parser.add_argument("--subjects", type=list, default=subjects)
parser.add_argument("--sub_id", type=str, default='DiAs')
parser.add_argument("--proc", type=str, default='preproc')
parser.add_argument("--stage", type=str, default='_BP_montage_preprocessed_raw.fif')
# Preprocessing stages:

# _BP_montage_concatenated_bads_marked_raw.fif
# _BP_montage_HFB_db_raw.fif
# _BP_montage_HFB_raw.fif
# _BP_montage_preprocessed_raw.fif
# _hfb_db_epo.fif
# _hfb_db_raw.fif
# _preprocessed_raw.fif

#%% Filtering parameters

# Useful channel to test on: ichan = 6
parser.add_argument("--l_freq", type=float, default=60.0)
parser.add_argument("--band_size", type=float, default=20.0)
parser.add_argument("--l_trans_bandwidth", type=float, default=10.0)
parser.add_argument("--h_trans_bandwidth", type=float, default=10.0)
parser.add_argument("--filter_length", type=str, default='auto')
parser.add_argument("--phase", type=str, default='minimum')
parser.add_argument("--fir_window", type=str, default='blackman')

#%% Epoching parameters

parser.add_argument("--t_prestim", type=float, default=-0.5)
parser.add_argument("--t_postim", type=float, default=1.75)
parser.add_argument("--baseline", default=None) # No baseline from MNE
parser.add_argument("--preload", default=True)
parser.add_argument("--tmin_baseline", type=float, default=-0.4)
parser.add_argument("--tmax_baseline", type=float, default=-0.1)
parser.add_argument("--mode", type=str, default='logratio')

#%% Visually responsive channels classification parmeters

parser.add_argument("--tmin_prestim", type=float, default=-0.4)
parser.add_argument("--tmax_prestim", type=float, default=-0.1)
parser.add_argument("--tmin_postim", type=float, default=0.1)
parser.add_argument("--tmax_postim", type=float, default=0.5)
parser.add_argument("--alpha", type=float, default=0.05)
parser.add_argument("--zero_method", type=str, default='pratt')
parser.add_argument("--alternative", type=str, default='two-sided')

#%% Create category specific time series

parser.add_argument("--sfreq", type=float, default=250.0)
parser.add_argument("--tmin_crop", type=float, default=0.0)
parser.add_argument("--tmax_crop", type=float, default=1.5)

#%%

args = parser.parse_args()

