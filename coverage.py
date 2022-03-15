#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 13:58:25 2022

@author: michael
"""
from scipy import stats
from numpy.random import rand
import matplotlib.pyplot as plt
import lhsmdu
import pandas as pd


distribution = stats.lognorm(
    s=2.0576536945440047, loc=-37.17739278721696, scale=8356.72250029186
)


def plot_samples(num_strata=5, num_cycles=500, threshold=0.05):
    data_points = []
    samples = []
    for x in range(num_cycles):
        k = lhsmdu.sample(1, num_strata)
        samples += lhsmdu.inverseTransformSample(distribution, k[0]).tolist()[0]
        test = stats.kstest(samples, distribution.cdf)
        data_points.append((len(samples), test.statistic))
        if test.statistic < threshold:
            break

    X, Y = zip(*data_points)

    plt.scatter(X, Y)
    plt.show()
    print(samples)


# stat = 1
# samples = []
# while stat > 0.05:
#     sample = pd.DataFrame(
#         lhsmdu.sample(1, 5).T,
#         columns=["sample"],
#     )['sample']
#     sample = sample * 10
#     samples += sample.to_list()
#     print(len(samples), stat)
#     samples.append(rand()*10)
#     stat = stats.kstest(samples, stats.uniform(loc=0.0, scale=10.0).cdf).statistic

# print(len(samples), stat)

plot_samples()
