#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 13:52:48 2021

@author: guime
"""


import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.0, 5.0, 100)
y = np.cos(2*np.pi*x) * np.exp(-x)

plt.plot(x, y, 'k')
plt.title('Damped exponential decay')
plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$')
plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
