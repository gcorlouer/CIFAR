#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:12:47 2020

@author: guime
"""
import cf_load
import HFB_test
import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp 

from pathlib import Path

%matplotlib

plt.rcParams.update({'font.size': 34})

# %% Parameters 
t_pr= -0.5
t_po=1.75
baseline = None
preload = True 
tmin_pr = -0.5
tmax_pr = -0.1 
tmin_po=0.1 
tmax_po=0.5
preproc = 'preproc'
task = 'stimuli'
run = '1'
nepochs = 28
nobs = 1126

#HFB_tot_face = np.empty(shape =( nepochs, 0, nobs))
#HFB_tot_place = np.empty(shape =( nepochs, 0, nobs))

HFB_tot_pref = np.empty(shape =( nepochs, 0, nobs)) # need to define time before
HFB_tot_npref = np.empty(shape =( nepochs, 0, nobs))

sub_id = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']

path_visual = cf_load.visual_path()
df_visual = pd.read_csv(path_visual)

for sub in sub_id:
    subject = cf_load.Subject(name=sub, task= task, run = run)
    fpath = subject.fpath(proc = preproc, suffix='lnrmv')
    raw = subject.import_data(fpath)
    
    face_chan = list(df_visual['chan_name'].loc[df_visual['subject_id']==sub].loc[df_visual['category']=='Face'])
    place_chan = list(df_visual['chan_name'].loc[df_visual['subject_id']==sub].loc[df_visual['category']=='Place'])
    
    bands = HFB_test.freq_bands() # Select Bands of interests 
    HFB_db = HFB_test.extract_HFB_db(raw, bands)
    HFB_db = HFB_db.drop_channels(ch_names = 'TRIG')
    time = HFB_db.times
    # place and face id
    events, event_id = mne.events_from_annotations(raw)
    face_id = HFB_test.extract_stim_id(event_id)
    place_id = HFB_test.extract_stim_id(event_id, cat='Place')
    
    if face_chan == []: 
        HFB_face = np.empty(shape =( nepochs, 0, nobs))
    else: 
        HFB_face = HFB_db[face_id].copy().pick(face_chan).get_data()
        #HFB_face = HFB_face.average().data
        
    if place_chan == []:
        HFB_place = np.empty(shape =( nepochs, 0, nobs))
    else : 
        HFB_place = HFB_db[place_id].copy().pick(place_chan).get_data()
        #HFB_place = HFB_place.average().data
    
    HFB_pref = np.append(HFB_face, HFB_place, axis = 1)
    
    if face_chan == []: 
        HFB_face = np.empty(shape =( nepochs, 0, nobs))
    else: 
        HFB_face = HFB_db[place_id].copy().pick(face_chan).get_data()
        #HFB_face = HFB_face.average().data
        
    if place_chan == []:
        HFB_place = np.empty(shape =( nepochs, 0, nobs))
    else : 
        HFB_place = HFB_db[face_id].copy().pick(place_chan).get_data()
        #HFB_place = HFB_place.average().data
    
    HFB_npref = np.append(HFB_face, HFB_place, axis = 1)
    
    HFB_tot_pref = np.append(HFB_tot_pref, HFB_pref, axis=1)
    HFB_tot_npref = np.append(HFB_tot_npref, HFB_npref, axis=1)
    #time = HFB_db.times
    
    #HFB_tot_face = np.append(HFB_tot_face, HFB_face, axis = 1 )
    #HFB_tot_place = np.append(HFB_tot_place, HFB_place, axis =1)


newshape = (28*71, nobs)
HFB_new_pref = np.reshape(HFB_tot_pref, newshape)
HFB_new_npref = np.reshape(HFB_tot_npref, newshape)

HFB_average_pref = np.mean(HFB_new_pref, 0)
HFB_average_npref = np.mean(HFB_new_npref, 0)

HFB_SE_pref = sp.stats.sem(HFB_new_pref, 0)
HFB_SE_npref = sp.stats.sem(HFB_new_npref, 0)


# %%
plt.plot(time, HFB_average_pref, label='Prefered stimulus')
plt.fill_between(time, HFB_average_pref-1.96*HFB_SE_pref, HFB_average_pref+1.96*HFB_SE_pref,
                         alpha=0.3)
plt.plot(time, HFB_average_npref, label= 'Non preferred stimulus')
plt.fill_between(time, HFB_average_npref-1.96*HFB_SE_npref, HFB_average_npref+1.96*HFB_SE_npref,
                         alpha=0.3)
plt.axhline(y=0)
plt.axvline(x=0)
plt.legend()
plt.title('Grand average of categroy selective HFB channels amplitude, n=71')
plt.xlabel('Time from stimulus onset (s)')
plt.ylabel('Amplitude (dB)')