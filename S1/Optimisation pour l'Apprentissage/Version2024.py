#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:17:49 2024

@author: mberar
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
plt.close("all")

def mesh(x_min,x_max,y_min,y_max, h = 0.1):
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

H = np.array([[1,-0.5],[-0.5,4]])
b = np.array([2,1])
c = 3

def cout(x):
    return 0.5*x@(H@x) + x@b + c

# Z contient tout les points de la grille -5,5
xx, yy  = mesh(-5,5,-5,5)
Cost = np.zeros(xx.shape)

for z in itertools.product(range(len(xx)),range(len(yy))):
    Cost[z] = cout(np.array([xx[z],yy[z]]))

plt.figure(1)
plt.contour(xx, yy, Cost, 20, cmap=plt.cm.Paired)
plt.show()

