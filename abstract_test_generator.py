#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:16:24 2022

@author: michael
"""
import networkx as nx
from itertools import product
import json

dag_path = "tests/dag.dot"

dag = nx.drawing.nx_pydot.read_dot(dag_path)


def should_test(n1, n2, dag):
    assert nx.algorithms.dag.is_directed_acyclic_graph(dag), "DAG must be a DAG."
    dag_ = dag.copy()
    dag_.add_edge(n1, n2)
    if nx.algorithms.dag.is_directed_acyclic_graph(dag_):
        return True
    else:
        return False


total_pairs = list(product(dag.nodes, dag.nodes))
print("total_pairs:", len(total_pairs))

testable_pairs = [(n1, n2, dag.has_edge(n1, n2)) for n1, n2 in total_pairs if should_test(n1, n2, dag)]
print("testable_pairs:", len(testable_pairs))
print("no_effect:", sum([x for _, _, x in testable_pairs]))

abstract_tests = []
for n1, n2, has_edge in testable_pairs:
    abstract_test = {
      "name": f"{n1}__{n2}",
      "mutations": {n1: "??"},
      "estimator": "??",
      "estimate_type": "??",
      "effect_modifiers": [],
      "expectedEffect": {n2: "??" if has_edge else "NoEffect()"},
      "skip": True
    }
    abstract_tests.append(abstract_test)

with open('tests/abstract_tests.json', 'w') as f:
    print(json.dumps(abstract_tests, indent=2), file=f)
