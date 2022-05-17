#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:07:07 2022

@author: michael
"""
import networkx as nx
from causal_testing.specification.variable import Input, Output
from itertools import product


def ancestors(dag, n1):
    dag = dag.copy()
    nodes = list(dag.nodes)
    for node in nodes:
        if not nx.has_path(dag, node, n1):
            dag.remove_node(node)
    return dag.nodes


def should_test(treatment, outcome, dag, scenario):
    assert nx.algorithms.dag.is_directed_acyclic_graph(dag), "DAG must be a DAG."
    # If the treatment -> outcome edge is in the DAG, we need to test it.
    if dag.has_edge(treatment, outcome):
        return True
    # We want to skip testing if we can get from node b (outcome) to node a (treatment)
    # as this goes directly against what is in the DAG.
    # This also covers the situation where a = b since a graph has a path from x to x.
    if nx.has_path(dag, outcome, treatment):
        return False
    # We also want to skip Output -> Input tests as this is nonsense (assuming inputs precede outputs).
    if isinstance(treatment, Output) and isinstance(outcome, Input):
        return False
    # Remove relationships where the treatment depends on outputs and the outcome depends on inputs
    # as this is basically an Output-Input test, which is nonsense. This will mostly apply to metavariables
    if any([isinstance(scenario.variables[p], Output) for p in ancestors(dag, treatment)]) and (
        any([isinstance(scenario.variables[p], Input) for p in ancestors(dag, outcome)])
    ):
        return False
    dag_ = dag.copy()
    dag_.add_edge(treatment, outcome)
    if nx.algorithms.dag.is_directed_acyclic_graph(dag_):
        return True
    else:
        return False


def all_total_effects(dag, scenario):
    return ((a, b) for a, b in product(dag.nodes, dag.nodes) if should_test(a, b, dag, scenario))


def paths_of_length_from(dag, u, n):
    if n == 0:
        return [[u]]
    paths = [
        [u] + path
        for neighbor in dag.successors(u)
        for path in paths_of_length_from(dag, neighbor, n - 1)
        if u not in path
    ]
    return paths


def all_paths_of_length(dag, length):
    allpaths = []
    for node in dag:
        allpaths += paths_of_length_from(dag, node, length)
    return allpaths


def all_paths_up_to_length(dag, length):
    allpaths = []
    for n in range(1, length):
        allpaths += all_paths_of_length(dag, n)
    return allpaths


def all_direct_effects(dag, length):
    direct_effects = []
    for path in all_paths_up_to_length(dag, length):
        for n1, n2 in zip(path, path[1:]):
            if (n1, n2) not in direct_effects:
                direct_effects.append((n1, n2))
    return direct_effects


def interaction_tests(dag):
    colliders = []
    for node in dag.nodes():
        parents = tuple(dag.predecessors(node))
        if len(parents) > 1:
            colliders.append((parents, (node,)))
    return colliders


def direct_effect_is_total(dag, treatment, outcome):
    all_paths = list(nx.all_simple_paths(dag, treatment, outcome))
    return all_paths == [[treatment, outcome]]
