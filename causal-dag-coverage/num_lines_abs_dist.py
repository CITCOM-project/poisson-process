#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:17:29 2022

@author: michael
"""
from fitter import Fitter, get_common_distributions, get_distributions
import pandas as pd
import os
import scipy

# data = pd.concat([pd.read_csv(f"../data/random/{csv}") for csv in os.listdir("../data/random")])
data = pd.read_csv("../data/random/data_random_5000.csv")

treatment = "num_lines_abs"

print(get_common_distributions())
f = Fitter(data[treatment], distributions=get_distributions())
f.fit()
(dist, params) = list(f.get_best(method="sumsquare_error").items())[0]
params = ",".join([f"{k}={v}" for k, v in params.items()])
dist = eval(f"scipy.stats.{dist}({params})")
print(f.summary())
print(scipy.stats.kstest(data[treatment], dist.cdf))
