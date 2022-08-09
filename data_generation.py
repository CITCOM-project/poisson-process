#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:56:26 2022

@author: michael
"""
import pandas as pd
from poisson_line_process import run_data
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import uniform
import lhsmdu


num_repeats = 100
run_again = False

for wh in range(1, 11):
    rct_samples = pd.DataFrame({"width": wh, "height": wh, "intensity": [1, 2, 4, 8, 16]})
    rct_datapath = f"data/smt_{num_repeats}/data_smt_wh{rct_samples['width'][0]}_{num_repeats}.csv"

    if os.path.exists(rct_datapath):
        print(f"RCT data already exists at {rct_datapath}")
        data = pd.read_csv(rct_datapath)
        plt.scatter(data['intensity'], data['num_shapes_unit'])
        intensities = sorted(list(set(data['intensity'])))
        columns = [data.where(data["intensity"] == i).dropna() for i in intensities]
        means = [col["num_shapes_unit"].mean() for col in columns]
        means_4 = np.array(means) * 4
        means_3 = np.array(means) * 3
        plt.scatter(intensities, means, color="red", marker="x", label="Empirical mean")
        plt.scatter(intensities[1:], means_4[:-1], color="orange", marker="x", label="4 * previous_mean")
        plt.scatter(intensities[1:], means_3[:-1], color="yellow", marker="x", label="3 * previous_mean")
        plt.xlabel("area")
        plt.ylabel("num_shapes_unit")
        plt.title(f"width = height = {wh}")
        plt.legend()
        # plt.show()
    else:
        print(f"Running RCT for {rct_datapath}")
        repeats = [rct_samples.copy() for i in range(num_repeats)]
        rct_data = run_data(pd.concat(repeats), rct_datapath)

for i in range(1, 11):
    rct_samples = pd.DataFrame({"width": range(1, 11), "height": range(1, 11), "intensity": i})
    rct_datapath = f"data/smt_wh_{num_repeats}/data_smt_i{i}_{num_repeats}.csv"

    if os.path.exists(rct_datapath):
        print(f"RCT data already exists at {rct_datapath}")
        data = pd.read_csv(rct_datapath)
        plt.scatter(data['width']*data['height'], data['num_shapes_unit'])
        plt.xlabel("area")
        plt.ylabel("num_shapes_unit")
        plt.title(f"intensity = {i}")
        # plt.show()
    else:
        print(f"Running RCT for {rct_datapath}")
        repeats = [rct_samples.copy() for i in range(num_repeats)]
        rct_data = run_data(pd.concat(repeats), rct_datapath)

for num_repeats in [100, 200, 300, 400]:
    random_datapath = f"data/random/data_random_{num_repeats*10}.csv"
    if os.path.exists(random_datapath) and not run_again:
        print(f"Random data already exists at {random_datapath}")
        random_data = pd.read_csv(random_datapath)
    else:
        print(f"Running random for {random_datapath}")
        random_samples = pd.concat([pd.DataFrame(
            lhsmdu.sample(3, 10).T, columns=["width", "height", "intensity"],
        ) for _ in range(num_repeats)])

        random_samples["width"] = lhsmdu.inverseTransformSample(
            uniform(0, 10), random_samples["width"]
        )

        random_samples["height"] = lhsmdu.inverseTransformSample(
            uniform(0, 10), random_samples["height"]
        )
        random_samples["intensity"] = lhsmdu.inverseTransformSample(
            uniform(0, 16), random_samples["intensity"]
        )
        # assert all([x >= 0 for x in random_samples['width']])
        # assert all([x >= 0 for x in random_samples['height']])

        print("Starting processing")
        random_data = run_data(random_samples, random_datapath)

# intensities = sorted(list(set(data["intensity"])))

# columns = [data.where(data["intensity"] == i).dropna() for i in intensities]

# # plt.boxplot([col['num_shapes_unit'] for col in columns])
# random_data = random_data.where(random_data['num_shapes_unit'] < 600).dropna()

# Alpha = (random_data["width"] + random_data["height"]) / (random_data["width"].max() + random_data["height"].max())
# assert all([alpha <= 1 for alpha in Alpha])
# plt.scatter(random_data["intensity"], random_data["num_shapes_unit"], alpha=Alpha)

# means = [col["num_shapes_unit"].mean() for col in columns]
# means_4 = np.array(means) * 4
# means_3 = np.array(means) * 3
# plt.scatter(intensities, means, color="red", marker="x")
# plt.scatter(intensities[1:], means_4[:-1], color="orange", marker="x")
# plt.scatter(intensities[1:], means_3[:-1], color="yellow", marker="x")
# plt.xlabel("Intensity")
# plt.ylabel("No. Polygons Unit")
# plt.title(f"width = height = {data['width'].mean()}")

# results = pd.DataFrame(
#     [means_3[:-1], means[1:], means_4[:-1]],
#     columns=[f"i={i}" for i in intensities[1:]],
#     index=["prev * 3", "actual", "prev * 4"],
# )
# print(results)
# plt.savefig(f"/tmp/plot_wh{data['width'].mean()}.png")
