#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:37:46 2022

@author: michael
"""
import lhsmdu
import pandas as pd
import sys
sys.path.append(".")

from poisson_line_process import run

samples = pd.DataFrame(lhsmdu.sample(3, 200).T, columns=["width", "height", "intensity"])
for col in samples:
    coefficient = 16
    if col == "intensity":
        coefficient = 32
    samples[col] = samples[col] * coefficient

num_lines, num_shapes = zip(*[run(row['width'], row['height'], row['intensity']) for _, row in samples.iterrows()])

samples["num_lines_abs"] = num_lines
samples["num_shapes_abs"] = num_shapes
samples.to_csv("data.csv")
