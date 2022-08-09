#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:25:43 2022

@author: michael
"""
from sklearn.neighbors import KernelDensity
import pandas as pd
from itertools import product


def SC(X, U, B, SA):
    n = len(B)
    covset = {B[i] for i in range(n) if any([x for x in SA(X) if U * (i-1)/n < x < U * i/n])}
    return len(covset)/n


def SC_verbose(X, U, B, SA):
    n = len(B)
    sax = SA(X)
    print(sax)
    
    for i in range(n):
        for x in sax:
            print(f"{U * (i-1)/n} < {x} < {U * i/n}")
        
    covset = {B[i] for i in range(n) if any([x for x in SA(X) if U * (i-1)/n < x < U * i/n])}
    return len(covset)/n


train_data = pd.read_csv("data/random/data_random_2000.csv", index_col=0)[['width', 'height', 'intensity']].to_numpy()
kde = KernelDensity(kernel='tophat', bandwidth=0.2).fit(train_data)

X = pd.read_csv("data/random/data_random_1000.csv", index_col=0)[['width', 'height', 'intensity']].to_numpy()[:10]
U = 2
B = product(range(U+1), repeat=3)

print(SC_verbose(X, U, list(B), kde.score_samples))
