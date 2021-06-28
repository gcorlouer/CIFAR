#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 13:09:15 2021

@author: guime
"""


import mne
import pandas as pd
import HFB_process as hf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path, PurePath
from config import args
from scipy.io import savemat
from scipy.stats import spearmanr
#%%

ecog = hf.Ecog(args.cohort_path, subject=args.subject, proc=args.proc, 
                       stage = '_hfb_extracted_raw.fif', epoch=args.epoch)
hfb = ecog.read_dataset()

ecog = hf.Ecog(args.cohort_path, subject=args.subject, proc=args.proc, 
                       stage = '_bad_chans_removed_raw.fif', epoch=args.epoch)
lfp = ecog.read_dataset()

#%% Compute correlations in resting state


X = hfb.copy().crop(tmin=50, tmax = 150).get_data()
Y = lfp.copy().crop(tmin=50, tmax = 150).get_data()

corr = spearmanr(X,Y, axis =1)
rho, pval = corr