#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 12:06:38 2021

@author: guime
"""

import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

from pathlib import Path
# %% Read all presented stimuli into numpy array

nstim = 29  # 14 faces, 14 places and 1 fixation screen
stimuli = [0]*nstim
home = Path.home()
# Directory containing presented stimuli
stimuli_dir = home.joinpath('projects', 'CIFAR', 'CIFAR_data', 'iEEG_10',
                            'presented_stimuli')
# Read stimuli into numpy array
for i, fname in enumerate(stimuli_dir.iterdir()):
    fpath = stimuli_dir.joinpath(fname)
    stimuli[i] = img.imread(fpath, format='jpg')

stimuli = np.stack(stimuli)

# %% Plot all presented stimuli

f, ax = plt.subplots(6,5)
for i in range(6):
    for j in range(5):
        if 5*i + j <= nstim-1:
            ax[i, j].imshow(stimuli[5*i + j, ...])
            ax[i, j].axis('off')

ax[5,4].axis('off')
