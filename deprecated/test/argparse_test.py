#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 15:56:24 2020

@author: guime
"""


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--x', type=int, default=3)
parser.add_argument('--y', type =int, default=2)
args = parser.parse_args()
x = args.x
y = args.y
print(x**2)
print(x+y)