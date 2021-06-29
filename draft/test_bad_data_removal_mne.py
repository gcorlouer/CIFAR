#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 20:05:51 2021

@author: guime
"""

#%%

import mne
import pandas as pd
import HFB_process as hf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path, PurePath
from config import args

#%%

ecog = hf.Ecog(args.cohort_path, subject=args.subject, proc=args.proc, 
                       stage = args.stage, epoch=args.epoch)
raw = ecog.read_dataset(run=1, task='stimuli')

ecg_epochs = mne.preprocessing.find_ecg_events(raw)
ecg_epochs.plot_image(combine='mean')

#%%

