from pathlib import Path, PurePath
from mne_bids import make_bids_basename
from mne.time_frequency import (tfr_multitaper, tfr_stockwell, tfr_morlet,
                                tfr_array_morlet)
from mne.time_frequency import csd_fourier, csd_multitaper, csd_morlet
from mne.time_frequency import psd_multitaper
from statsmodels.stats.multitest import fdrcorrection, multipletests
from numpy import inf

import mne 
import numpy as np 
import os
import pandas as pd
import seaborn
import re
import scipy.stats as spstats
import matplotlib.pyplot as plt 

# %% Data manip
def import_data(subject='DiAs', subject_id='04', run='1', task='rest_baseline', montage='bipolar_montage', proc='BP', suffix='.set', load=True):
    
    cfsubdir = Path('~','CIFAR_data','iEEG_10','subjects', subject, 'EEGLAB_datasets', montage).expanduser()
    bids_subpath = Path('~','projects','CIFAR','data_bids', f'sub-{subject_id}' ).expanduser()
    elecinfo = bids_subpath.joinpath('anat','electrodes_info.csv')
    dfelec = pd.read_csv(elecinfo)
    ROIs = dfelec['Brodman'].unique()
    nROIs = len(ROIs)
    fname = CIFAR_filename(subid=subject,task=task,proc=proc, run=run, suffix=suffix)
    fpath = cfsubdir.joinpath(fname)
    fpath = os.fspath(fpath)
    raw = mne.io.read_raw_eeglab(fpath, preload=load)
    return raw, dfelec

def BIDS_filename(sub_id="sub-00", task="rest", run="01",
                   ext=".set"):
    """Return filename given subject id, task, run, and datatype """
    dataset = [sub_id,  task, "baseline", f"run-{run}", 'ieeg']
    dataset = "_".join(dataset)
    dataset = dataset + ext
    return dataset

def CIFAR_filename(subid="JuRo", task="sleep", run="1", proc="raw",
                      suffix=".set"):
    """Return filename given subject id, task, run, and datatype """
    if task == 'sleep':
        if proc == 'raw':
            dataset = [subid, "freerecall", task, 'preprocessed']
            dataset = "_".join(dataset)
            dataset = dataset + suffix
        else:
            dataset = [subid, "freerecall", task, 'preprocessed', 'BP', 'montage']
            dataset = "_".join(dataset)
            dataset = dataset + suffix
    else:
        if proc == 'raw':
            dataset = [subid, "freerecall", task, run, 'preprocessed']
            dataset = "_".join(dataset)
            dataset = dataset + suffix
        else:
            dataset = [subid, "freerecall", task, run, 'preprocessed', 'BP', 'montage']
            dataset = "_".join(dataset)
            dataset = dataset + suffix       
    return dataset

def subject_path(data_dir=Path('~','CIFAR_data').expanduser(), subid='JuRo', 
                 montage='bipolar_montage'):
    
    cfsubdir = data_dir.joinpath('iEEG_10','subjects', subid)
    data_subdir = cfsubdir.joinpath('EEGLAB_datasets', montage)
    return cfsubdir, data_subdir

#%% Anatomical info 
def electrodes_info(sub_num = '0'):
    subject_path = Path('~','projects','CIFAR','data_bids', f'sub-0{sub_num}')
    elecinfo = subject_path.joinpath('anat','electrodes_info.csv')
    dfelec = pd.read_csv(elecinfo)
    return dfelec

def ch_info(picks, dfelec, epochs):
    ROI_pick = []
    for i in range(len(picks)):
        ROI_pick.extend(dfelec['Brodman'].loc[dfelec['electrode_name']==picks[i].split('-')[0]])

    ch_index = mne.pick_channels(epochs.info['ch_names'], include=picks)
    ch_names = epochs.info['ch_names']
    return ROI_pick, ch_index, ch_names

#%% Envelope extraction and normalisation

def HFB_raw(raw, l_freq=60, nband=6, band_size=20):
    # Extract High frequency broadband envelope 
    bands = [l_freq+i*band_size for i in range(1, nband)]
    nobs = len(raw.times)
    nchan = len(raw.info['ch_names'])
    HFB_norm = np.zeros(shape=(nchan, nobs))
    HFB_mean_amplitude = np.zeros(shape=(nchan,))

    for band in bands:
        raw_band = raw.copy().filter(l_freq=band, h_freq=band+band_size, phase='minimum', filter_length='auto', l_trans_bandwidth= 10, h_trans_bandwidth= 10, 
                                     fir_window='blackman')
        HFB = raw_band.copy().apply_hilbert(envelope=True).get_data()
        HFB_mean_amplitude = HFB_mean_amplitude + np.mean(HFB, axis=1) # mean amplitude
        HFB = np.divide(HFB, np.mean(HFB, axis=1)[:,np.newaxis]) 
        HFB_norm = HFB_norm + HFB # normalise amplitude

    HFB_norm = HFB_norm/nband # average across band
    HFB_mean_amplitude = HFB_mean_amplitude/nband # Average of mean amplitude across bands
    HFB_norm = HFB_norm * HFB_mean_amplitude[:,np.newaxis] # (not sure if needed)
    HFB_norm = np.nan_to_num(HFB_norm) # replace nan with zeros
    raw_HFB = mne.io.RawArray(HFB_norm, raw.info) # create raw structure from HFB
    return HFB_norm, raw_HFB

def HFB_epochs(epochs, l_freq=60, nband=6, band_size=20):
    # Extract High frequency broadband envelope 
    bands = [l_freq+i*band_size for i in range(1, nband)]
    nobs = len(epochs.times)
    nchan = len(epochs.info['ch_names'])
    nepochs = len(epochs)
    HFB_norm = np.zeros(shape=(nepochs, nchan, nobs))
    HFB_mean_amplitude = np.zeros(shape=(nepochs, nchan))

    for band in bands:
        epochs_band = epochs.copy().filter(l_freq=band, h_freq=band+band_size, phase='minimum', filter_length='auto', l_trans_bandwidth= 10, h_trans_bandwidth= 10, 
                                     fir_window='blackman')
        HFB = epochs_band.copy().apply_hilbert(envelope=True).get_data()
        HFB_mean_amplitude = HFB_mean_amplitude + np.mean(HFB, axis=2) # mean amplitude
        HFB = np.divide(HFB, np.mean(HFB, axis=2)[:,:,np.newaxis]) 
        HFB_norm = HFB_norm + HFB # normalise amplitude

    HFB_norm = HFB_norm/nband # average across band
    HFB_mean_amplitude = HFB_mean_amplitude/nband # Average of mean amplitude across bands
    #HFB_norm = HFB_norm * HFB_mean_amplitude (not sure if needed)
    HFB_norm = np.nan_to_num(HFB_norm) # replace nan with zeros
    epochs_HFB = mne.EpochsArray(HFB_norm, epochs.info) # create raw structure from HFB
    return HFB_norm, epochs_HFB


def HFB_norm(epochs, events, tmin):
    # Normalise high frequency amplitude relative to prestimulus baseline 
    baseline = epochs.copy().crop(tmin=-0.4, tmax=-0.1) # Extract prestimulus baseline
    HFB_epoch = epochs.get_data() 
    baseline = baseline.get_data()
    baseline = spstats.mstats.gmean(baseline,axis=2)
    HFB_norm = np.zeros_like(HFB_epoch)
    for i in range(len(epochs)):
        for j in range(len(epochs.info['ch_names'])):
            HFB_norm[i,j,:] = np.divide(HFB_epoch[i,j,:], baseline[i,j])# divide by baseline
        HFB_norm = np.nan_to_num(HFB_norm)
        HFB_db = np.zeros_like(HFB_norm) 
    for i in range(len(epochs)):
        for j in range(len(epochs.info['ch_names'])):
            HFB_db[i,j,:] = 10*np.log10(HFB_norm[i,j,:]) # transform into normal distribution
        HFB_db = np.nan_to_num(HFB_db)
    HFB_db = mne.EpochsArray(HFB_db, epochs.info, events=events[1:], tmin=tmin) # create epochs structure
    # Note: Haven't add events_id because had to drop one event
    return HFB_db

#%% ERP inspection

def stim_id(events_id):
    
    p_face = re.compile('Face')
    p_place = re.compile('Place')
    place_id = []
    face_id = []
    #p_face.match(events_id.keys())
    for key in events_id.keys():
        if p_place.match(key):
            place_id.append(str(events_id[key]))
        if p_face.match(key):
            face_id.append(str(events_id[key]))
    return place_id, face_id 

def plot_stim_response(picks, HFB_db, cdt, label):
        evok = HFB_db[cdt].copy().pick(picks).average()
        evok_std = HFB_db[cdt].copy().pick(picks).standard_error()
        ERP = evok.data
        ERP_std = evok_std.data
        time = HFB_db.times
        plt.plot(time, ERP[0,:], label=label)
        plt.fill_between(time, ERP[0,:]-1.96*ERP_std[0,:], ERP[0,:]+1.96*ERP_std[0,:],
                         alpha=0.3)
        
# %% Channel statistics

def detect_visual(A_pr, A_po, HFB_db, alpha):
    
    M1 = np.mean(A_pr,axis=2)
    M2 = np.mean(A_po,axis=2)
    # Get rid of infinity 
    M1[M1==-inf] = 0
    M2[M2 == -inf] = 0
    # Compute inflated p values
    pval = [0]*len(HFB_db.info['ch_names'])
    freedom_degree = [0]*len(HFB_db.info['ch_names'])
    tstat = [0]*len(HFB_db.info['ch_names'])
    for i in range(0,len(HFB_db.info['ch_names'])):
        tstat[i], pval[i] = spstats.wilcoxon(M1[:,i], M2[:,i], zero_method='zsplit') # Non normal distrib 
    # Correct multiplt testing    
    reject, pval_correct = fdrcorrection(pval, alpha=alpha)
    
    # Compute effect size: Cohen d 
    MC1 = np.mean(M1, axis=0)
    MC2 = np.mean(M2, axis=0)
    std1 = np.std(M1, axis=0)
    std2 = np.std(M2, axis=0)
    n1 = M1.shape[1]
    n2 = M2.shape[1]
    std = np.sqrt(np.divide((n1-1)*std1**2+(n2-1)*std2**2,(n1+n2-2)))
    cohen = np.divide(MC1-MC2, std)
    # Return visual channels
    idx = np.where(reject==True)
    idx = idx[0]
    visual_chan = []
    visual_cohen = []
    for i in list(idx):
        visual_chan.append(HFB_db.info['ch_names'][i])
        visual_cohen.append(np.abs(cohen[i]))
    return reject, pval_correct, visual_chan, visual_cohen