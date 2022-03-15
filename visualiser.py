#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 08:21:43 2022

@author: michael
"""
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import os

cwd = os.getcwd()

datapath = "data.csv"
if "tests" not in os.getcwd():
    datapath = "tests/"+datapath

data = pd.read_csv(datapath)

area = data['width'] * data['height']
data['num_shapes_unit'] = data['num_shapes_abs']/area
assert all([s > 0 for s in data['num_shapes_abs']])
assert all([s > 0 for s in data['num_shapes_unit']])

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter(data['width']+data['height'], data['intensity'], data['num_lines_abs'])
ax.set_xlabel("width+height")
ax.set_ylabel("intensity")
ax.set_zlabel("num_lines_abs")

wh = np.linspace(0, 32)
i = np.linspace(0, 32)

WH, I = np.meshgrid(wh, i)

S = 2*I*WH

ax.plot_surface(WH, I, S, color="yellow")
plt.show()

# X = data[['intensity']].copy()
# X['intercept'] = 1
# X['intensity^2'] = X['intensity'] ** 2
# Y = data['num_shapes_unit']


# rlm_model = sm.RLM(Y, X, M=sm.robust.norms.HuberT())

# rlm_results = rlm_model.fit()
# print(rlm_results.params)

# x = pd.DataFrame()
# x['intensity'] = np.linspace(0, 32)
# x['intercept'] = 1
# x['intensity^2'] = x['intensity'] ** 2
# y = rlm_results.predict(x)
# plt.plot(x['intensity'], y, color='r')

# test_case = pd.DataFrame()
# test_case["intensity"] = [1.32, 0.66]
# test_case['intercept'] = 1
# test_case['intensity^2'] = test_case['intensity'] ** 2
# y = rlm_results.predict(test_case)
# plt.scatter(test_case['intensity'], y, color='orange')
# print(y)

# plt.savefig("/tmp/graph.svg")

# risk_ratio = y.iloc[0]/y.iloc[1]
# print(risk_ratio)


