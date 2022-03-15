#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:10:54 2022

@author: michael
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
from fitter import Fitter, get_common_distributions, get_distributions


datapaths = ["data.csv", "data1.csv"]
if "tests" not in os.getcwd():
    datapaths = ["tests/" + datapath for datapath in datapaths]

data = pd.concat([pd.read_csv(datapath, index_col=0) for datapath in datapaths])

area = data["width"] * data["height"]
data["num_shapes_unit"] = data["num_shapes_abs"] / area
assert all([s > 0 for s in data["num_shapes_abs"]])
assert all([s > 0 for s in data["num_shapes_unit"]])


for column in ["num_lines_abs"]:
    print(column)
    # plt.hist(data['num_shapes_abs'], bins=25)

    f = Fitter(data[column], distributions=get_common_distributions())
    f.fit()
    print(f.summary())
    print(f.get_best(method="sumsquare_error"))
