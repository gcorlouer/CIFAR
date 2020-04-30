# %% ToDo
# Exploratory analysis: Gaussianity, Mean, var, std, kurtosis, violin/box plot
# Time freq analysis
# Epoching and analysis on epochs
# Remove line noise

# %% Necessary modules
from mne_bids import make_bids_basename
from pathlib import Path, PurePath
import mne
import os
import matplotlib.pyplot as plt
import pandas as pd
import matlab.engine
import numpy as np
import scipy as sp
import scipy.io

# Enable the table_schema option in pandas,
# data-explorer makes this snippet available with the `dx` prefix:
pd.options.display.html.table_schema = True
pd.options.display.max_rows = None
# %% Import data
%matplotlib auto
CIFAR_dir = Path('~', 'projects', 'CIFAR').expanduser()
bids_dir = CIFAR_dir.joinpath('data_bids')
subject_dir = bids_dir.joinpath('sub-00')
elecinfo = subject_dir.joinpath('anat','electrodes_info.csv')
bids_basename = make_bids_basename(subject='00',
                                   task='rest', run='01', suffix='ieeg.vhdr')
fname = subject_dir.joinpath('ieeg', bids_basename)
raw = mne.io.read_raw_brainvision(fname, preload=True)
# %% Pick ROIs (will be more useful for analysis)
# Import electrodes
dfelec = pd.read_csv(elecinfo)
dfelec
ROIs = dfelec['Brodman'].unique()
nROIs = len(ROIs)
ROIs
# Select electrodes in specific ROIs
dfelec['electrode_name'].loc[dfelec['Brodman'] == ROIs[1]]

# %% Drop bad channels
raw.set_channel_types({'ECG': 'ecg'})
bads = ['TRIG', 'ECG', 'P3', 'T5', 'T3', 'F3', 'F7']
raw.plot(duration=50, n_channels=60, scalings=4e-4, color='b')
raw.info['bads'] = bads
raw_bad = raw.copy().pick_types(eeg=True, ecog=True, exclude = 'bads')
raw_bad.info
# %% Exploratory data anlysis
# psd
raw_bad.plot_psd(xscale='log', spatial_colors=False, color='blue')
raw_bad
#  %% Line noise removal via outlier removal
# Start matlab and add noisetool to path
# Add relevant path
eng = matlab.engine.start_matlab()
noisetool = eng.fullfile('~', 'toolboxes', 'NoiseTools')
eng.addpath(eng.genpath(noisetool))
eng.rmpath(eng.fullfile(noisetool, 'COMPAT'))
# Get data in numpy array:
data = raw_bad.get_data()
datapath = CIFAR_dir.joinpath('temp', 'data.mat')
datalist = data.tolist()
scipy.io.savemat(datapath, dict(data))
