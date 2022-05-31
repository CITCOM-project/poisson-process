#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 08:38:35 2022

@author: michael
"""
import json
import networkx as nx
import pandas as pd
import scipy
import os
from scipy.stats import uniform, kstest

from fitter import Fitter, get_common_distributions

from causal_testing.specification.variable import Input, Output
from causal_testing.specification.scenario import Scenario
from dag_tools import all_total_effects, all_direct_effects, direct_effect_is_total

import matplotlib.pyplot as plt

# 1. Read the DAG and data
dot_path = "../case-study/dag.dot"
dag = nx.drawing.nx_agraph.read_dot(dot_path)
# data = pd.read_csv("../data/random/data_random_2000.csv")
data = pd.concat([pd.read_csv(f"../data/random/data_random_{num}.csv") for num in [1000, 2000, 3000, 4000]])
# data = pd.concat([pd.read_csv(f"../data/random/{csv}") for csv in os.listdir("../data/random")])

plt.hist(data["height"])

# 2. Create variables
width = Input("width", float, uniform(0, 10))
height = Input("height", float, uniform(0, 10))
intensity = Input("intensity", float, uniform(0, 16))

num_lines_abs = Output("num_lines_abs", float)
num_lines_unit = Output("num_lines_unit", float)
num_shapes_abs = Output("num_shapes_abs", float)
num_shapes_unit = Output("num_shapes_unit", float)

# 3. Create scenario by applying constraints over a subset of the input variables
scenario = Scenario(
    variables={
        width,
        height,
        intensity,
        num_lines_abs,
        num_lines_unit,
        num_shapes_abs,
        num_shapes_unit,
    }
)

# 4. Check all inputs have distributions
for ip in scenario.inputs():
    assert ip.distribution is not None, f"Distribution of input {ip} must be specified"

# 5. For variables with specified distributions, check the data matches these
for v in scenario.variables.values():
    if v.distribution is not None:
        pvalue = kstest(data[v.name], v.distribution.cdf).pvalue
        assert pvalue >= 0.95, (
            f"P-value {pvalue} is less than 0.95. "
            + f"Data distribution for {v.name} possibly does not match specified distribution. "
            + "Consider getting more data values."
        )


# 6. For variables without specified distributions, infer one
def fit_distribution(data, treatment):
    f = Fitter(data[treatment], distributions=get_common_distributions() + ["exponpow"])
    f.fit()
    # print(treatment)
    # print(f.summary())
    (dist, params) = list(f.get_best(method="sumsquare_error").items())[0]
    params = ",".join([f"{k}={v}" for k, v in params.items()])
    return dist, params


for v in scenario.variables.values():
    if v.distribution is None:
        dist, params = fit_distribution(data, v.name)
        v.distribution = eval(f"scipy.stats.{dist}({params})")
        pvalue = kstest(data[v.name], v.distribution.cdf).pvalue
        if 0.05 < pvalue < 0.95:
            print(
                f"P-value {pvalue} is outside 95% confidence. "
                + f"Inferred distribution {dist}({params}) for {v.name} does not fit it very well. "
                + "Consider getting more data values."
            )


# 7. Load the tests from JSON (or wherever)
with open("concrete_tests.json") as f:
    test_suite = json.load(f)

# 8. Allocate each test to an edge in the DAG
total_effects = {(a, b): [] for a, b in all_total_effects(dag, scenario)}
direct_effects = {(a, b): [] for a, b in all_total_effects(dag, scenario) if not direct_effect_is_total(dag, a, b)}

# Let's focus on total causality for now...
for test in test_suite:
    treatment_vars = list(test["control_input_configuration"])[0]
    outcome_vars = list(test["outcome_variables"])[0]
    if test["effect"] == "total" and (treatment_vars, outcome_vars) in total_effects:
        total_effects[(treatment_vars, outcome_vars)].append(test)
    elif test["effect"] == "direct" and (treatment_vars, outcome_vars) in direct_effects:
        direct_effects[(treatment_vars, outcome_vars)].append(test)
    else:
        raise ValueError(
            "Invalid test {test['effect']} effect with treatment {treatment_vars} and outcome {outcome_vars}."
        )


def score_edge(dag, data, treatment, tests):
    control_values = [test["control_input_configuration"][treatment.name] for test in tests]
    if control_values == []:
        return 0
    distribution = None
    if hasattr(treatment, "distribution"):
        distribution = treatment.distribution
    if distribution is None:
        distribution = fit_distribution(data, treatment.name)
    return 100 - (kstest(control_values, distribution.cdf).pvalue * 100)

print("Total Effect Scores")
total_effect_scores = {edge: None for edge in total_effects}
for edge in total_effects:
    treatment, _ = edge
    score = score_edge(dag, data, scenario.variables[treatment], total_effects[edge])
    total_effect_scores[edge] = score
    print(edge, score)
print("Direct Effect Scores")
direct_effect_scores = {edge: None for edge in direct_effects}
for edge in direct_effects:
    treatment, _ = edge
    score = score_edge(dag, data, scenario.variables[treatment], direct_effects[edge])
    direct_effect_scores[edge] = score
    print(edge, score)
