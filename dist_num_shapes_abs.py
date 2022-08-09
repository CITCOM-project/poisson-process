#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:10:54 2022

@author: michael
"""
import io
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Node import Node
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.Helper import list_union, powerset


def draw_dag_with_labels(causal_dag, labels):
    """Draw the given causal DAG with the given set of labels.
    
    Based on causal-learn test here: https://github.com/cmu-phil/causal-learn/
    blob/e4e73f8b58510a3cd5a9125ba50c0ac62a425ef3/tests/
    TestGraphVisualization.py#L106
    
    :param causal_dag: The Causal DAG to draw.
    :param labels: An ordered list of labels for the variables of the dag.
    """
    pydot_causal_dag = GraphUtils.to_pydot(causal_dag.G,
                                           labels=labels)
    tmp_png = pydot_causal_dag.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')
    plt.axis('off')
    plt.imshow(img)
    plt.show()

data = pd.concat([pd.read_csv(f"data/random/data_random_{num}.csv", index_col=0) for num in [1000, 2000, 3000, 4000, 5000, 6000]])
cg = pc(data.to_numpy(), 0.01, fisherz, False, 0, -1, show_progress=False)  # Run PC and obtain the estimated graph (CausalGraph object)
draw_dag_with_labels(cg, labels=data.columns)