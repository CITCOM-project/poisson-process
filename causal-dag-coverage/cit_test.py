#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:46:45 2022

@author: michael
"""
import pycit
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt


# df = pd.read_csv("../data/random/data_random_5000.csv")
data = pd.concat([pd.read_csv(f"../data/random/data_random_{i}000.csv", index_col=0) for i in range(1, 7)])

df = data[['width', 'height', 'intensity', 'num_shapes_unit', 'num_lines_abs']].copy()
df['intensity^2'] = df['intensity'] ** 2

# pc_raw = pg.partial_corr(data=df, x='intensity', y='num_shapes_unit', covar=[], method='spearman')
# pc_adj = pg.partial_corr(data=df, x='intensity', y='num_shapes_unit', covar=["num_shapes_abs"], method='spearman')

# print(pc_raw)
# print(pc_adj)
print(df.partial_corr("intensity", "num_shapes_unit", []))
print(df.partial_corr("intensity", "num_shapes_unit", ["width", "height", "num_lines_abs"]))
print(df.pcorr()['intensity'])
# df.pcorr().to_csv("pcorr_2.csv")


