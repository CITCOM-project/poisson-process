#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:24:25 2022

@author: michael
"""
from poisson_line_process import run_data
import pandas as pd
import lhsmdu
from scipy.stats import uniform
import numpy as np
from scipy.stats import kstest


values = [lhsmdu.sample(3, 100).T for _ in range(40)]
values = np.concatenate(values)
runs = pd.DataFrame(values, columns=["width", "height", "intensity"])

runs["width"] = lhsmdu.inverseTransformSample(uniform(0, 10), runs["width"])
runs["height"] = lhsmdu.inverseTransformSample(uniform(0, 10), runs["height"])
runs["intensity"] = lhsmdu.inverseTransformSample(uniform(0, 16), runs["intensity"])

print(kstest(runs['width'], uniform(0, 10).cdf))

run_data(runs, f"data/random/data_random_{len(runs)}.csv")
