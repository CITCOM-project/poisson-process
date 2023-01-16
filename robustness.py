#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 09:15:03 2023

@author: michael
"""
import pandas as pd
import math
from scipy.stats import t
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# DATAPATH = "data/random/data_random_1000.csv"


def estimate_robustness(model, q=1, alpha=1):
    """Calculate the robustness of a linear regression model. This allow
    the user to identify how large an unidentified confounding variable
    would need to be to nullify the causal relationship under test."""

    dof = model.df_resid
    t_values = model.tvalues

    fq = q * abs(t_values / math.sqrt(dof))
    f_crit = abs(t.ppf(alpha / 2, dof - 1)) / math.sqrt(dof - 1)
    fqa = fq - f_crit

    rv = 0.5 * (np.sqrt(fqa**4 + (4 * fqa**2)) - fqa**2)

    return rv


data = pd.concat([pd.read_csv(f"data/random/data_random_{n}.csv", index_col=0) for n in [1000, 2000, 3000]])

iw_robustness = {k:[] for k in range(2, 17)}
for i in range(2, 17):
    for _ in range(30):
        df = data.where(data['width'] < i).dropna()
        df = df.where(df['height'] < i).dropna()
        df = df.where(df['intensity'] < i).dropna()
        df = df.sample(15)
    
        Y = df[['num_lines_abs']]
        # X = df[['intensity', 'width', 'height']]
        X = pd.DataFrame()
        X["iw"] = df['intensity'] * df['width']
        X["ih"] = df['intensity'] * df['height']
        # X['width*height'] = df['width'] * df['height']
        # X['intensity'] = X['intensity'] ** 2
        # sm.add_constant(X, prepend=False)
        # X['intercept'] = 1
        
        model = sm.OLS(Y, X).fit()
        robustness = estimate_robustness(model)
        iw_robustness[i].append(robustness['iw'])

data = pd.DataFrame().from_dict(iw_robustness)
plt.scatter(data.mean().index, data.mean())
plt.fill_between(data.mean().index, data.max(), data.min(), color=[(0, 0, 1, 0.5)])
plt.xlabel("i, h, and w less than")
plt.ylabel("Robustness")
