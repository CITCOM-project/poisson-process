from causal_testing.specification.variable import Input, Output, Meta
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.generation.abstract_causal_test_case import AbstractCausalTestCase
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.testing.causal_test_outcome import ExactValue, Positive, Negative, CausalTestResult, NoEffect
from causal_testing.testing.causal_test_engine import CausalTestEngine
import argparse
import scipy
import json
import pandas as pd
from fitter import Fitter, get_common_distributions

import causal_test_setup as cts


def populate_metas(metas, outputs, data):

    for meta in metas:
        meta.populate(data)

    for var in metas + outputs:
        f = Fitter(data[var.name], distributions=get_common_distributions())
        f.fit()
        (dist, params) = list(f.get_best(method="sumsquare_error").items())[0]
        params = ",".join([f"{k}={v}" for k, v in params.items()])
        var.distribution = eval(f"scipy.stats.{dist}({params})")
        print(var.name, f"{dist}({params})")


def json_parse(file) -> tuple[list, list, list, dict, pd.DataFrame]:
    """Parse a JSON input file into inputs, outputs, metas and a test plan

    :param file: path to JSON input file to be parsed
    :returns:
        - inputs -  variables that are input parameters to the system
        - outputs -  variables that are outputs from the system
        - metas - variables that are not directly observable but are relevant to system testing, e.g. a model may take
         a location parameter and expand this out into average_age and household_size variables "under the hood".
          These parameters can be made explicit by instantiating them as metavariables.
        - test_plan - json object containing execution data for the system-under-test
    """
    with open(file) as f:
        test_plan = json.load(f)

    type_dict = {"float": float, "str": str, "int": int}

    inputs = [Input(i['name'], type_dict[i['type']], cts.distributions[i['distribution']]) for i in test_plan['inputs']]
    outputs = [Output(i['name'], type_dict[i['type']]) for i in test_plan['outputs']]
    metas = [Meta(i['name'], type_dict[i['type']], getattr(cts, i['populate'])) for i in
             test_plan['metas']] if "metas" in test_plan else []

    data = pd.read_csv(test_plan['data'])
    print(type(data))
    return inputs, outputs, metas, test_plan, data


# TODO: Work out how best to get constraints in there


def execute_test_case(causal_test_case, estimator, modelling_scenario, test_plan: dict, causal_specification) -> bool():
    """
    Takes a given test case and runs it using the CausalTestingFramework.

    :param causal_test_case:
    :param estimator:
    :param modelling_scenario:
    :param test_plan:
    :param causal_specification:
    :return: True if test failed, returns false otherwise.
    """
    failed = False
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
    if isinstance(estimation_model, cts.WidthHeightEstimator):
        estimation_model.add_product_term_to_df("width", "intensity")
        estimation_model.add_product_term_to_df("height", "intensity")
    causal_test_result = causal_test_engine.execute_test(estimation_model, estimate_type=causal_test_case.estimate_type)
    test_passes = causal_test_case.expected_causal_effect.apply(causal_test_result)

    result_string = str()
    if causal_test_result.ci_low() and causal_test_result.ci_high():
        result_string = f"{causal_test_result.ci_low()} < {causal_test_result.ate} <  {causal_test_result.ci_high()}"
    else:
        result_string = causal_test_result.ate
    assert test_passes, f"{causal_test_case}\n    FAILED - expected {causal_test_case.expected_causal_effect}, got {result_string}"
    if not test_passes:
        failed = True
        print(f"    FAILED - expected {causal_test_case.expected_causal_effect}, got {causal_test_result.ate}")
    return failed


def main():
    parser = argparse.ArgumentParser(
        description='A script for parsing json config files for the Causal Testing Framework')
    parser.add_argument("--json_path", help="path to the json file containing the causal tests",
                        default="causal_tests.json")
    args = parser.parse_args()
    json_path = args.json_path

    inputs, outputs, metas, test_plan, data = json_parse(json_path)
    populate_metas(metas, outputs, data)

    modelling_scenario = Scenario(inputs + outputs + metas, None)
    modelling_scenario.setup_treatment_variables()
    causal_specification = CausalSpecification(scenario=modelling_scenario, causal_dag=CausalDAG(test_plan['dag']))

    mutate = {
        "Increase": lambda x: modelling_scenario.treatment_variables[x].z3 > modelling_scenario.variables[x].z3,
        "ChangeByFactor(2)": lambda x: modelling_scenario.treatment_variables[x].z3 == modelling_scenario.variables[
            x].z3 * 2
    }


    executed_tests = 0
    failures = 0

    for test in test_plan['tests']:
        if "skip" in test and test['skip']:
            continue

        abstract_test = AbstractCausalTestCase(
            scenario=modelling_scenario,
            intervention_constraints=[mutate[v](k) for k, v in test['mutations'].items()],
            treatment_variables={modelling_scenario.variables[v] for v in test['mutations']},
            expected_causal_effect={modelling_scenario.variables[variable]: cts.effects[effect] for variable, effect in
                                    test["expectedEffect"].items()},
            effect_modifiers={modelling_scenario.variables[v] for v in
                              test['effect_modifiers']} if "effect_modifiers" in test else {},
            estimate_type=test['estimate_type']
        )

        concrete_tests, runs = abstract_test.generate_concrete_tests(5, 0.05)
        print(abstract_test)
        print([(v.name, v.distribution) for v in abstract_test.treatment_variables])
        print(len(concrete_tests))
        for concrete_test in concrete_tests:
            executed_tests += 1
            failed = execute_test_case(concrete_test, cts.estimators[test['estimator']], modelling_scenario, test_plan,
                                       causal_specification)
            if failed:
                failures += 1

    print(f"{failures}/{executed_tests} failed")


if __name__ == '__main__':
    main()
