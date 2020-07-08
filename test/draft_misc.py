#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 09:56:30 2020

@author: guime
"""

import numpy as np

nface = 48
times = 1251
A = np.empty(shape =(0, times))

B = np.random.randn(5, times)
C = np.random.randn(10, times)

D = np.append(A, B, axis=0)

