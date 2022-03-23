from causal_testing.specification.variable import Input, Output, Meta
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.generation.abstract_causal_test_case import AbstractCausalTestCase
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.testing.causal_test_outcome import ExactValue, Positive, Negative, CausalTestOutcome, \
    CausalTestResult
from causal_testing.testing.causal_test_engine import CausalTestEngine
from causal_testing.testing.estimators import LinearRegressionEstimator, Estimator, CausalForestEstimator

import json
import scipy
import numpy as np
import pandas as pd

from fitter import Fitter, get_common_distributions


class WidthHeightEstimator(LinearRegressionEstimator):
    def add_modelling_assumptions(self):
        """
        Add modelling assumptions to the estimator. This is a list of strings which list the modelling assumptions that
        must hold if the resulting causal inference is to be considered valid.
        """
        self.modelling_assumptions += 'The output varies according to 2i(w+h)'

    def estimate_ate(self) -> (float, [float, float], float):
        """ Estimate the conditional average treatment effect of the treatment on the outcome. That is, the change
        in outcome caused by changing the treatment variable from the control value to the treatment value.
        :return: The conditional average treatment effect and the 95% Wald confidence intervals.
        """
        assert self.effect_modifiers, f"Must have at least one effect modifier to compute CATE - {self.effect_modifiers}."
        x = pd.DataFrame()
        x[self.treatment[0]] = [self.treatment_values, self.control_values]
        x['Intercept'] = 1
        for k, v in self.effect_modifiers.items():
            self.adjustment_set.add(k)
            x[k] = v
        if hasattr(self, "product_terms"):
            for a, b in self.product_terms:
                x[f"{a}*{b}"] = x[a] * x[b]

        x.drop(['width', 'height'], axis=1, inplace=True)
        self.adjustment_set = {"width*intensity", "height*intensity"}

        print(x)
        print(self.adjustment_set)

        model = self._run_linear_regression()
        print(model.summary())
        y = model.predict(x)
        treatment_outcome = y.iloc[0]
        control_outcome = y.iloc[1]

        return treatment_outcome - control_outcome, None


class PoissonWidthHeight(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be positive."""

    def __init__(self, tolerance=0.5):
        self.tolerance = tolerance
        self.i2c = None

    def apply(self, res: CausalTestResult) -> bool:
        # TODO: confidence intervals?
        print("=== APPLYING ===")
        print("effect_modifier_configuration", res.effect_modifier_configuration)
        effect_modifier_configuration = {k.name: v for k, v in res.effect_modifier_configuration.items()}
        c = res.treatment_value - res.control_value
        i = effect_modifier_configuration['intensity']
        self.i2c = i * 2 * c
        print("2ic:", f"2*{i}*{c}={self.i2c}")
        print("ate:", res.ate)
        return np.isclose(res.ate, self.i2c, atol=self.tolerance)

    def __str__(self):
        if self.i2c is None:
            return f"PoissonWidthHeight±{self.tolerance}"
        else:
            return f"PoissonWidthHeight:{self.i2c}±{self.tolerance}"


def populate_width_height(data):
    data['width_plus_height'] = data['width'] + data['height']


def populate_num_lines_unit(data):
    area = data['width'] * data['height']
    data['num_lines_unit'] = data['num_lines_abs'] / area


def populate_num_shapes_unit(data):
    area = data['width'] * data['height']
    data['num_shapes_unit'] = data['num_lines_abs'] / area


def json_parse(file):
    with open(file) as f:
        test_plan = json.loads("".join(f.readlines()))

    inputs = [Input(i['name'], eval(i['type']), eval(i['distribution'])) for i in test_plan['inputs']]
    outputs = [Output(i['name'], eval(i['type'])) for i in test_plan['outputs']]
    metas = [Meta(i['name'], eval(i['type']), eval(i['populate'])) for i in
             test_plan['metas']] if "metas" in test_plan else []

    data = pd.read_csv(test_plan['data'])

    for meta in metas:
        meta.populate(data)

    for var in metas + outputs:
        f = Fitter(data[var.name], distributions=get_common_distributions())
        f.fit()
        (dist, params) = list(f.get_best(method="sumsquare_error").items())[0]
        params = ",".join([f"{k}={v}" for k, v in params.items()])
        var.distribution = eval(f"scipy.stats.{dist}({params})")
        print(var.name, f"{dist}({params})")

    return inputs, outputs, metas, test_plan


# TODO: Work out how best to get constraints in there


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
    if "intensity" in [v.name for v in treatment_vars]:
        estimation_model.add_squared_term_to_df("intensity")
    if "width" in [v.name for v in treatment_vars] or "height" in [v.name for v in treatment_vars]:
        estimation_model.add_product_term_to_df("width", "intensity")
        estimation_model.add_product_term_to_df("height", "intensity")
    causal_test_result = causal_test_engine.execute_test(estimation_model, estimate_type=causal_test_case.estimate_type)
    test_passes = causal_test_case.expected_causal_effect.apply(causal_test_result)
    # assert test_passes, f"{causal_test_case}\n    FAILED - expected {causal_test_case.expected_causal_effect}, got {causal_test_result.ate}"
    if not test_passes:
        global failures
        failures += 1
        print(f"    FAILED - expected {causal_test_case.expected_causal_effect}, got {causal_test_result.ate}")


# abstract_tests = [AbstractCausalTestCase(
#     scenario=modelling_scenario,
#     intervention_constraints=[mutate[v](k) for k, v in test['mutations'].items()],
#     treatment_variables={modelling_scenario.variables[v] for v in test['mutations']},
#     expected_causal_effect={modelling_scenario.variables[variable]: eval(effect) for variable, effect in test["expectedEffect"].items()},
#     effect_modifiers={modelling_scenario.variables[v] for v in test['effect_modifiers']} if "effect_modifiers" in test else {},
#     estimate_type=test['estimate_type']
# )
# for test in test_plan['tests']]


if __name__ == '__main__':
    inputs, outputs, metas, test_plan = json_parse("causal_tests.json")
    modelling_scenario = Scenario(inputs + outputs + metas, [])
    modelling_scenario.setup_treatment_variables()
    causal_specification = CausalSpecification(scenario=modelling_scenario, causal_dag=CausalDAG(test_plan['dag']))

    failures = 0
    executed_tests = 0

    mutate = {
        "Increase": lambda x: modelling_scenario.treatment_variables[x].z3 > modelling_scenario.variables[x].z3,
        "ChangeByFactor(2)": lambda x: modelling_scenario.treatment_variables[x].z3 == modelling_scenario.variables[
            x].z3 * 2
    }

    for test in test_plan['tests']:
        abstract_test = AbstractCausalTestCase(
            scenario=modelling_scenario,
            intervention_constraints=[mutate[v](k) for k, v in test['mutations'].items()],
            treatment_variables={modelling_scenario.variables[v] for v in test['mutations']},
            expected_causal_effect={modelling_scenario.variables[variable]: eval(effect) for variable, effect in
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
            print(concrete_test)
            executed_tests += 1
            execute_test_case(concrete_test, eval(test['estimator']))
    print(f"{failures}/{executed_tests} failed")
