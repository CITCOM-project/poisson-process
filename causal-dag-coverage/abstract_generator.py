#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:14:38 2022

@author: michael
"""
import networkx as nx
from itertools import product
import json

from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from dag_tools import should_test

# 1. Read the DAG
dot_path = "../case-study/dag.dot"
dag = nx.drawing.nx_agraph.read_dot(dot_path)

# 2. Create variables
width = Input("width", float)
height = Input("height", float)
intensity = Input("intensity", float)

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


# 4. Generate the pairs of inputs to test
tests = {
    (a, b): []
    for a, b in product(["Input", "Output", "Meta"], ["Input", "Output", "Meta"])
    if not (a == "Output" and b == "Input")
}
for treatment, outcome in product(dag.nodes, dag.nodes):
    if not should_test(treatment, outcome, dag, scenario):
        continue
    abstract_test = {
        "name": f"{treatment}__{outcome}",
        "mutations": {treatment: "??"},
        "estimator": "??",
        "estimate_type": "??",
        "expectedEffect": {outcome: "??" if dag.has_edge(treatment, outcome) else "NoEffect()"},
        "skip": True,
    }
    tests[scenario.variables[treatment].__class__.__name__, scenario.variables[outcome].__class__.__name__].append(
        abstract_test
    )

print("DAG has", len(dag.edges), "edges")
print(len([v for k in tests for v in tests[k]]), "abstract tests required")
json_tests = {f"{k[0]}-{k[1]}": v for k, v in tests.items()}
with open("abstract_tests.json", "w") as f:
    print(json.dumps(json_tests, indent=2), file=f)
