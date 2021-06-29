#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 17:34:20 2021
Explore results from pairwise functional connectivity
@author: guime
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import HFB_process as hf
import pandas as pd

from scipy.io import loadmat
from config import args
from pathlib import Path, PurePath

#%%

result_path = Path('~','projects', 'CIFAR','CIFAR_data', 'results').expanduser()
fname = 'all_subjects_fc_results.csv'
fpath = result_path.joinpath(fname)
df = pd.read_csv(fpath)

pcgc = df['pcgc']

#%%

pair = ['RR','RF','FR','FF']
df = df[df['pair'].isin(pair)]

g = sns.FacetGrid(df, row = "pair", ylim = (0,0.003))
g.map_dataframe(sns.boxplot, x="cdt", y="pcgc")
g.map_dataframe(sns.stripplot, x="cdt", y="pcgc")

#%%


