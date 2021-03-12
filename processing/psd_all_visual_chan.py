import HFB_process as hf
import cifar_load_subject as cf
import pandas as pd
import numpy as np 
import helper_functions as fun 
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_multitaper

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
#%% Concatenate populations

X = np.concatenate(tuple(ts), axis=0)
pop_to_idx = hf.parcellation_to_indices(df_sorted, parcellation='group')
pop = df_sorted['group'].unique()
npop = len(pop)
ts_pop = [0]*npop

for i, p in enumerate(pop):
    idx = pop_to_idx[p]
    X_p = np.take(X, idx, axis = 0)
    ts_pop[i] = np.average(X_p, axis=0)

X_pop = np.stack(ts_pop)
ntrial = X_p.shape[2]
(n, m, T, c) = X_pop.shape


#%% Compute psd
newshape = (n, T, c, m)
X = np.transpose(X_pop, (0,2,3,1))
psd, freqs = psd_array_multitaper(X, sfreq, fmin=fmin, fmax=fmax,
                                             bandwidth=bandwidth)
psd_mean = np.mean(psd, axis=1)  # average over channels
psd_std = np.std(psd, axis=1)



#%% Plot trial based psd 

nstate = X_pop.shape[-1]
for i in range(nstate):
    plt.subplot(3,1,i+1)
    for j in range(npop):
        confidence_up = psd_std[j,i,:] + 1.96*psd_std[j,i,:]/np.sqrt(ntrial)
        confidence_down = psd_std[j,i,:] - 1.96*psd_std[j,i,:]/np.sqrt(ntrial)
        plt.plot(freqs, psd_std[j,i,:], label = pop[j])
        plt.fill_between(freqs, confidence_down, confidence_up, alpha=0.2)
        plt.ylim((0.01, 100))
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.ylabel('PSD')

plt.xlabel('Frequency (Hz)')

