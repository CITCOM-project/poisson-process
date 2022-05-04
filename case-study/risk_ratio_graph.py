#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:20:44 2022

@author: michael
"""
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("intensity_num_shapes_results_random_1000.csv", index_col=0)

colors = ["red", "orange", "yellow", "green", "blue"]

for w, df in data.groupby("width"):
    # if w % 2 == 1:
    plt.plot(df['treatment'], df['smt_risk_ratio'], label=f"{w}")
plt.plot(df['treatment'], df['obs_risk_ratio'], label="1000 random", color="purple")

plt.legend(ncol=3, title="Width")
plt.xlabel("Intensity (treatment)")
plt.ylabel("Risk Ratio")
plt.show()
