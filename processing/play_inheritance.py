#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 12:16:12 2021

@author: guime
"""

class A():
    def __init__(self, a=0.1):
        self.a = a

class B(A):
    def __init__(self, a):
        super().__init__(a)

a = 0.3
parent = A(a=a)
child = B(a=a)

print(f'Parent is {parent.a}\n')
print(f'Child is {child.a}')