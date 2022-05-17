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
from scipy.stats import uniform, kstest

from fitter import Fitter, get_common_distributions

from causal_testing.specification.variable import Input, Output
from causal_testing.specification.scenario import Scenario
from dag_tools import all_total_effects, all_direct_effects, direct_effect_is_total

# 1. Read the DAG and data
dot_path = "../case-study/dag.dot"
dag = nx.drawing.nx_agraph.read_dot(dot_path)
data = pd.read_csv("../data/random/data_random_1000.csv")

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

# 4. Load the tests from JSON (or wherever)
with open("concrete_tests.json") as f:
    test_suite = json.load(f)

# 5. Allocate each test to an edge in the DAG
total_effects = {(a, b): [] for a, b in all_total_effects(dag, scenario)}
direct_effects = {(a, b): [] for a, b in all_direct_effects(dag, 3) if not direct_effect_is_total(dag, a, b)}
# Let's focus on direct causality for now...
# edges.update({interaction: [] for interaction in interaction_tests(dag)})
# print(total_effects)
# All our tests are testing total effects, but we need a way to distinguish between direct and total effects
for test in test_suite:
    treatment_vars = list(test['control_input_configuration'])[0]
    outcome_vars = list(test['outcome_variables'])[0]
    total_effects[(treatment_vars, outcome_vars)].append(test)


def fit_distribution(data, treatment):
    f = Fitter(data[treatment], distributions=get_common_distributions())
    f.fit()
    (dist, params) = list(f.get_best(method="sumsquare_error").items())[0]
    params = ",".join([f"{k}={v}" for k, v in params.items()])
    return eval(f"scipy.stats.{dist}({params})")


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


total_effect_scores = {edge: None for edge in total_effects}
for edge in total_effects:
    treatment, _ = edge
    total_effect_scores[edge] = score_edge(dag, data, scenario.variables[treatment], total_effects[edge])
print(total_effect_scores)
