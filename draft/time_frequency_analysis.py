import HFB_process as hf
import cifar_load_subject as cf
import pandas as pd
import numpy as np 
import helper_functions as fun 
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_multitaper, tfr_array_multitaper, tfr_array_morlet

from scipy.io import savemat

# %matplotlib
#%% TODO
# compute psd over all chans and then average over populations (compare to averaging over
# populations first)
#%% 

subjects = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
proc = 'preproc' 
sfreq = 500;
fmin = 0.1
fmax = 150;
bandwidth = 3
tmin_crop = -0.5
tmax_crop = 1.75
fname = 'visual_channels_BP_montage.csv'
ieeg_path = cf.cifar_ieeg_path(home='~')
visual_chan_table_path = ieeg_path.joinpath(fname)
df = pd.read_csv(visual_chan_table_path)
df_sorted = pd.DataFrame(columns=df.columns)

ts = [0]*len(subjects)
for s in range(len(subjects)):
    sub_id = subjects[s]  
    subject = cf.Subject(name=sub_id)
    datadir = subject.processing_stage_path(proc=proc)
    visual_chan = subject.pick_visual_chan()
    HFB = hf.visually_responsive_HFB(sub_id = sub_id)
    ts[s], time = fun.ts_all_categories(HFB, sfreq=sfreq, tmin_crop=tmin_crop, tmax_crop=tmax_crop)
    df_sorted = df_sorted.append(df.loc[df['subject_id']==sub_id].sort_values(by='latency'))

df_sorted = df_sorted.reset_index(drop=True)
X = np.concatenate(tuple(ts), axis=0)
X = np.transpose(X, (3,2,0,1))
#%%
nstate = X.shape[-1]
time_freq = [0]*nstate
freqs = np.arange(5., 100., 3.)
for i in range(nstate):
    epoch_data = X[i,...]
    time_freq[i] = tfr_array_morlet(epoch_data, sfreq, freqs, n_cycles=freqs/2., 
                                        output='complex')