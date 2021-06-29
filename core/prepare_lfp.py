#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 12:58:01 2021

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
from scipy.io import savemat

#%%

ecog = hf.Ecog(args.cohort_path, subject=args.subject, proc=args.proc, 
                       stage = args.stage, epoch=args.epoch)
lfp = ecog.read_dataset()
