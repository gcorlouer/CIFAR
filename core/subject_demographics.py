#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 16:37:10 2021

@author: guime
"""
from pathlib import Path
import pandas as pd
# %%

home = Path.home()
ieeg_dir = home.joinpath('projects', 'CIFAR', 'CIFAR_data', 'iEEG_10')
demographic_path = ieeg_dir.joinpath('subjects_demographics.csv')
df = pd.read_csv(demographic_path)
# Compute useful statistics
total = df.sum(axis=0, skipna=False, numeric_only=True)
average = df.mean(axis=0, numeric_only=True)
std = df.std(axis=0, numeric_only=True)
# Append rows
df = df.append(total, ignore_index=True)
df = df.append(average, ignore_index=True)
df = df.append(std, ignore_index=True)
# %% Save file
fname = 'processed_demographics.csv'
fpath = ieeg_dir.joinpath(fname)
df.to_csv(fpath, index=False)
