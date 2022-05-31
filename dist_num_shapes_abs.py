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
from fcit import fcit
import numpy as np

# datapaths = ["data.csv", "data1.csv"]
# if "tests" not in os.getcwd():
#     datapaths = ["tests/" + datapath for datapath in datapaths]

# data = pd.concat([pd.read_csv(datapath, index_col=0) for datapath in datapaths])

data = pd.concat([pd.read_csv(f"data/random/data_random_{num}.csv") for num in [1000, 2000, 3000, 4000, 5000, 6000]])


# area = data["width"] * data["height"]
# data["num_shapes_unit"] = data["num_shapes_abs"] / area
# assert all([s > 0 for s in data["num_shapes_abs"]])
# assert all([s > 0 for s in data["num_shapes_unit"]])

# print(data[['intensity']].to_numpy().shape)


# for column in ["num_lines_abs"]:
#     print(column)
#     # plt.hist(data['num_shapes_abs'], bins=25)

#     f = Fitter(data[column], distributions=get_common_distributions())
#     f.fit()
#     print(f.summary())
#     print(f.get_best(method="sumsquare_error"))



for _ in range(10):
    unadjusted = fcit.test(
            data[["intensity"]].to_numpy(),
            data[["num_shapes_unit"]].to_numpy(),
        )
    
    adjusted = fcit.test(
            data[["intensity"]].to_numpy(),
            data[["num_shapes_unit"]].to_numpy(),
            data[["height", "width", "num_shapes_abs"]].to_numpy(),
        )
    
    print(unadjusted, adjusted)

# plt.scatter(data['intensity'], data['num_lines_abs'])
# plt.hist(data['num_lines_abs'])
