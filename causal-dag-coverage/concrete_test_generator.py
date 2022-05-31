#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 08:16:53 2022

@author: michael
"""
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.specification.variable import Input, Output
from causal_testing.testing.causal_test_outcome import ExactValue, Positive

import json


def test_to_dict(causal_test_case):
    {
        "control_input_configuration": {k.name: v for k, v in causal_test_case.control_input_configuration.items()},
        "expected_causal_effect": str(causal_test_case.expected_causal_effect),
        "outcome_variables": [v.name for v in causal_test_case.outcome_variables],
        "treatment_input_configuration": {k.name: v for k, v in causal_test_case.treatment_input_configuration.items()},
        "estimate_type": causal_test_case.estimate_type,
        "effect_modifier_configuration": {k.name: v for k, v in causal_test_case.effect_modifier_configuration.items()},
        "effect": "total"
      }


# 2. Create variables
width = Input('width', float)
height = Input('height', float)
intensity = Input('intensity', float)

num_lines_abs = Output('num_lines_abs', float)
num_lines_unit = Output('num_lines_unit', float)
num_shapes_abs = Output('num_shapes_abs', float)
num_shapes_unit = Output('num_shapes_unit', float)

causal_tests = []

for wh in range(1, 11):
    smt_data_path = f"../data/smt_100/data_smt_wh{wh}_100.csv"
    for control_value, treatment_value in [(1, 2), (2, 4), (4, 8), (8, 16)]:
        # 5. Create a causal test case
        causal_test_case = CausalTestCase(control_input_configuration={intensity: control_value},
                                          treatment_input_configuration={intensity: treatment_value},
                                          expected_causal_effect=ExactValue(4, tolerance=0.5),
                                          outcome_variables={num_shapes_unit},
                                          estimate_type="risk_ratio",
                                          )
        causal_tests.append(test_to_dict(causal_test_case))

for i in range(17):
    for w in range(1, 10):
        # 5. Create a causal test case
        control_value = w
        treatment_value = w+1
        causal_test_case = CausalTestCase(control_input_configuration={width: control_value},
                                          treatment_input_configuration={width: treatment_value},
                                          expected_causal_effect=Positive(),
                                          outcome_variables={num_shapes_unit},
                                          estimate_type="ate_calculated",
                                          effect_modifier_configuration={intensity: i}
                                          )
        causal_tests.append(test_to_dict(causal_test_case))


with open("concrete_tests.json", 'w') as f:
    print(json.dumps(causal_tests, indent=2), file=f)
