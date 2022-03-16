from causal_testing.specification.variable import Input, Output, Meta
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.generation.abstract_causal_test_case import AbstractCausalTestCase
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.testing.causal_test_outcome import ExactValue, Positive, Negative, CausalTestResult, NoEffect
from causal_testing.testing.causal_test_engine import CausalTestEngine
from causal_testing.testing.estimators import LinearRegressionEstimator, Estimator, CausalForestEstimator

import json
import scipy
import numpy as np
import pandas as pd

from fitter import Fitter, get_common_distributions

from custom_outcomes import PoissonIntensity, PoissonWidthHeight
from custom_estimators import WidthHeightEstimator

def populate_width_height(data):
    data['width_plus_height'] = data['width'] + data['height']


def populate_num_lines_unit(data):
    area = data['width'] * data['height']
    data['num_lines_unit'] = data['num_lines_abs'] / area


def populate_num_shapes_unit(data):
    area = data['width'] * data['height']
    data['num_shapes_unit'] = data['num_lines_abs'] / area


with open("causal_tests.json") as f:
    test_plan = json.loads("".join(f.readlines()))

inputs = [Input(i['name'], eval(i['type']), eval(i['distribution'])) for i in test_plan['inputs']]
outputs = [Output(i['name'], eval(i['type'])) for i in test_plan['outputs']]
metas = [Meta(i['name'], eval(i['type']), eval(i['populate'])) for i in test_plan['metas']] if "metas" in test_plan else []

data = pd.read_csv(test_plan['data'])

for meta in metas:
    meta.populate(data)

for var in metas+outputs:
    f = Fitter(data[var.name], distributions=get_common_distributions())
    f.fit()
    (dist, params) = list(f.get_best(method="sumsquare_error").items())[0]
    params = ",".join([f"{k}={v}" for k, v in params.items()])
    var.distribution = eval(f"scipy.stats.{dist}({params})")
    print(var.name, f"{dist}({params})")

# TODO: Work out how best to get constraints in there
modelling_scenario = Scenario(inputs+outputs+metas, [])
modelling_scenario.setup_treatment_variables()
causal_specification = CausalSpecification(scenario=modelling_scenario, causal_dag=CausalDAG(test_plan['dag']))

mutate = {
    "Increase": lambda x: modelling_scenario.treatment_variables[x].z3 > modelling_scenario.variables[x].z3,
    "ChangeByFactor(2)": lambda x: modelling_scenario.treatment_variables[x].z3 == modelling_scenario.variables[x].z3 * 2
}

def execute_test_case(causal_test_case, estimator):
    data_collector = ObservationalDataCollector(modelling_scenario, test_plan['data'])
    causal_test_engine = CausalTestEngine(causal_test_case, causal_specification, data_collector)
    minimal_adjustment_set = causal_test_engine.load_data(index_col=0)
    treatment_vars = list(causal_test_case.treatment_input_configuration)
    minimal_adjustment_set = minimal_adjustment_set - set([v.name for v in treatment_vars])
    # @andrewc19, why can we only have atomic control/treatment values?
    # I think it'd be good to pass it in as two dicts instead of vars, control, treatment lists
    estimation_model = estimator((list(treatment_vars)[0].name,),
                                             [causal_test_case.treatment_input_configuration[v] for v in treatment_vars][0],
                                             [causal_test_case.control_input_configuration[v] for v in treatment_vars][0],
                                             minimal_adjustment_set,
                                             (list(causal_test_case.outcome_variables)[0].name,),
                                             causal_test_engine.scenario_execution_data_df,
                                             effect_modifiers=causal_test_case.effect_modifier_configuration
                                             )
    if "intensity" in [v.name for v in treatment_vars] and hasattr(estimation_model, "add_squared_term_to_df"):
        estimation_model.add_squared_term_to_df("intensity")
    if isinstance(estimation_model, WidthHeightEstimator):
        estimation_model.add_product_term_to_df("width", "intensity")
        estimation_model.add_product_term_to_df("height", "intensity")
    causal_test_result = causal_test_engine.execute_test(estimation_model, estimate_type=causal_test_case.estimate_type)
    test_passes = causal_test_case.expected_causal_effect.apply(causal_test_result)

    result_string = ""
    if causal_test_result.ci_low() and causal_test_result.ci_high():
        result_string  = f"{causal_test_result.ci_low()} < {causal_test_result.ate} <  {causal_test_result.ci_high()}"
    else:
        result_string = causal_test_result.ate
    assert test_passes, f"{causal_test_case}\n    FAILED - expected {causal_test_case.expected_causal_effect}, got {result_string}"
    if not test_passes:
        global failures
        failures += 1
        print(f"    FAILED - expected {causal_test_case.expected_causal_effect}, got {causal_test_result.ate}")

failures = 0
executed_tests = 0

# abstract_tests = [AbstractCausalTestCase(
#     scenario=modelling_scenario,
#     intervention_constraints=[mutate[v](k) for k, v in test['mutations'].items()],
#     treatment_variables={modelling_scenario.variables[v] for v in test['mutations']},
#     expected_causal_effect={modelling_scenario.variables[variable]: eval(effect) for variable, effect in test["expectedEffect"].items()},
#     effect_modifiers={modelling_scenario.variables[v] for v in test['effect_modifiers']} if "effect_modifiers" in test else {},
#     estimate_type=test['estimate_type']
# )
# for test in test_plan['tests']]

for test in test_plan['tests']:
    if "skip" in test and test['skip']:
        continue
    abstract_test = AbstractCausalTestCase(
        scenario=modelling_scenario,
        intervention_constraints=[mutate[v](k) for k, v in test['mutations'].items()],
        treatment_variables={modelling_scenario.variables[v] for v in test['mutations']},
        expected_causal_effect={modelling_scenario.variables[variable]: eval(effect) for variable, effect in test["expectedEffect"].items()},
        effect_modifiers={modelling_scenario.variables[v] for v in test['effect_modifiers']} if "effect_modifiers" in test else {},
        estimate_type=test['estimate_type']
    )
    concrete_tests, runs = abstract_test.generate_concrete_tests(5, 0.5)
    print(abstract_test)
    print([(v.name, v.distribution) for v in abstract_test.treatment_variables])
    print(len(concrete_tests))
    for concrete_test in concrete_tests:
        print(concrete_test)
        executed_tests += 1
        execute_test_case(concrete_test, eval(test['estimator']))
print(f"{failures}/{executed_tests} failed")
