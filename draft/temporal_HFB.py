import HFB_process as hf
import cifar_load_subject as cf
import pandas as pd
import numpy as np 
import helper_functions as fun 
import matplotlib.pyplot as plt

from scipy.io import savemat

# %matplotlib
#%% TODO

# -Check that there are output visual_data X is correct with HFB_visual (i.e. check that 
# permutation works)
# - Create a module for category specific electrodes
# - Rearrange HFB module consequently
#%% 

subjects = ['AnRa',  'AnRi',  'ArLa',  'BeFe',  'DiAs',  'FaWa',  'JuRo', 'NeLa', 'SoGi']
proc = 'preproc' 
sfreq = 250;
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
    ts[s], time = fun.ts_all_categories(HFB, tmin_crop=tmin_crop, tmax_crop=tmax_crop)
    df_sorted = df_sorted.append(df.loc[df['subject_id']==sub_id].sort_values(by='latency'))

df_sorted = df_sorted.reset_index(drop=True)
#%% 

X = np.concatenate(tuple(ts), axis=0)

#%% 

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
std_pop = np.std(X_pop, axis=2)
X_pop = np.average(X_pop, axis=2)
#%% Plot result

nstate = X_pop.shape[-1]
for i in range(nstate):
    plt.subplot(3,1,i+1)
    for j in range(npop):
        confidence_up = X_pop[j,:,i] + 1.96*std_pop[j,:,i]/np.sqrt(ntrial)
        confidence_down = X_pop[j,:,i] - 1.96*std_pop[j,:,i]/np.sqrt(ntrial)
        plt.plot(time, X_pop[j,:,i], label = pop[j])
        plt.fill_between(time, confidence_down, confidence_up, alpha=0.2)
        plt.ylim((-0.5, 2))
        plt.axvline(x=0)
        plt.legend()
        plt.ylabel('Amplitude (dB)')

plt.xlabel('Time (s)')