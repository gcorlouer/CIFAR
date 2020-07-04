#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:02:51 2020

@author: guime
"""
import cf_load
import HFB_test
import mne
import matplotlib.pyplot as plt
import pandas as pd

%matplotlib

plt.rcParams.update({'font.size': 17})

# %% Parameters 
t_pr= -0.5
t_po=1.75
baseline = None
preload = True 
tmin_pr = -0.5
tmax_pr = -0.1 
tmin_po=0.1 
tmax_po=0.5


sub_id = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
functional_group = {'subject_id': [], 'chan_name':[], 'category': [], 'brodman': []}

for sub in sub_id: 
    #%% Import data
    subject = cf_load.Subject(sub)
    fpath = subject.fpath(suffix='lnrmv')
    raw = subject.import_data(fpath)
    dfelec = subject.dfelec()
    
    # %% Extract HFB
    
    bands = HFB_test.freq_bands() # Select Bands of interests 
    HFB_db = HFB_test.extract_HFB_db(raw, bands)
    
    # place and face id
    events, event_id = mne.events_from_annotations(raw)
    face_id = HFB_test.extract_stim_id(event_id)
    place_id = HFB_test.extract_stim_id(event_id, cat='Place')
    
    # Check test looks ok
    # HFB_test.plot_stim_response(HFB_db, picks='LTo6', stim_id=face_id)
    
    
    # %% Detect face, place and bicat channels
    
    visual = HFB_test.make_visual_cat(HFB_db, face_id, place_id)
    
    # %% Dictionary
    
    for key in visual.keys():
        cat = visual[key]
        functional_group['subject_id'].extend([subject.name]*len(cat))
        functional_group['chan_name'].extend(cat)
        functional_group['category'].extend([key]*len(cat))
        functional_group['brodman'].extend(subject.ROIs(cat))
            
# %% save in csv file

df_visual = pd.DataFrame(functional_group)

df_visual = df_visual[df_visual.brodman != 'Unknown']
fpath = Path('~', 'CIFAR_data', 'iEEG_10').expanduser()
fname = 'visual_electrodes.csv'
fpath = fpath.joinpath(fname)
df_visual.to_csv(fpath, index= False)

df = df_visual

tot_face = df['category'].loc[df['category']=='Face']
tot_place = df['category'].loc[df['category']=='Place']
tot_bicat = df['category'].loc[df['category']=='Bicat']