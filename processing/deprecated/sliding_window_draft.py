#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 16:12:19 2021

@author: guime
"""

import numpy as np
import matplotlib.pyplot as plt
 
eps = np.random.normal(0,1, 1000)

window_size = 100
step = 10
start = 0
nobs = eps.shape[0]
nwin = round((nobs - window_size)/step)
win_start = [0]*nwin
win_end = [0]*nwin
eps_win = [0]*nwin
for k in range(nwin):
    win_start[k] = k*step
    win_end[k] = win_start[k] + window_size
    eps_win[k] = eps[win_start[k]:win_end[k]]
eps_win = np.stack(eps_win)


def slide_window(X, start=None, stop=None, step=None, window_size=None):
    nobs = stop - start + 1
    nwin = round((nobs - 1 - window_size)/step)
    win_start = [0]*nwin
    win_end = [0]*nwin
    X_win = [0]*nwin
    for k in range(nwin):
        win_start[k] = k*step + start
        win_end[k] = win_start[k] + window_size
        X_win[k] = X[win_start[k]:win_end[k]]
    X_win = np.stack(eps_win)
    return X_win

# %%

eps_win = slide_window(eps, start=10, stop=100, step=step, window_size=window_size)
#%%

obs = range(0,window_size)
eps[0:window_size]

plt.plot(obs, eps[step:step + window_size])
plt.plot(obs, eps_win[1,:])
