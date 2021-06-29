#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 16:27:41 2021

@author: guime
"""


import matlab.engine
import os
from pathlib import Path, PurePath

matlab_path = Path('~', 'projects', 'CIFAR', 'code_matlab').expanduser()
analysis_path = matlab_path.joinpath('analysis')
os.chdir(matlab_path)
eng = matlab.engine.start_matlab()
eng.startup_mat(nargout=0)


