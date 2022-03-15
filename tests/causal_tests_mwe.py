from causal_testing.specification.variable import Input, Output
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.testing.causal_test_case import CausalTestCase
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.testing.causal_test_outcome import Positive
from causal_testing.testing.causal_test_engine import CausalTestEngine
from causal_testing.testing.estimators import LinearRegressionEstimator

# define the inputs
width = Input("width", float)
height = Input("height", float)
intensity = Input("intensity", float)

# define the outputs
num_lines = Output("num_lines", float)
num_shapes = Output("num_shapes", float)

modelling_scenario = Scenario(
    [width, height, intensity, num_lines, num_shapes],
    {width > 0, height > 0, intensity > 0}
)

dag_path = "dag.dot"
observational_data_csv_path = "data.csv"
causal_specification = CausalSpecification(scenario=modelling_scenario, causal_dag=CausalDAG(dag_path))

control_input_configuration = {intensity: 1}
treatment_input_configuration = {intensity: 2}
outcome_var = num_shapes

causal_test_case = CausalTestCase(
    control_input_configuration=control_input_configuration,
    treatment_input_configuration=treatment_input_configuration,
    expected_causal_effect=Positive(),
    outcome_variables=[outcome_var],
)

data_collector = ObservationalDataCollector(modelling_scenario, observational_data_csv_path)
causal_test_engine = CausalTestEngine(causal_test_case, causal_specification, data_collector)
minimal_adjustment_set = causal_test_engine.load_data(index_col=0)
treatment_vars = list(causal_test_case.treatment_input_configuration)
minimal_adjustment_set = minimal_adjustment_set - set([v.name for v in treatment_vars])
# @andrewc19, why can we only have atomic control/treatment values?
# I think it'd be good to pass it in as two dicts instead of vars, control, treatment lists
estimation_model = LinearRegressionEstimator((list(treatment_vars)[0].name,),
                                         [causal_test_case.treatment_input_configuration[v] for v in treatment_vars][0],
                                         [causal_test_case.control_input_configuration[v] for v in treatment_vars][0],
                                         minimal_adjustment_set,
                                         (list(causal_test_case.outcome_variables)[0].name,),
                                         causal_test_engine.scenario_execution_data_df,
                                         )
causal_test_result = causal_test_engine.execute_test(estimation_model)
test_passes = causal_test_case.expected_causal_effect.apply(causal_test_result)
assert test_passes, f"{causal_test_case}\n    FAILED - Expected an increase in the number of shapes per unit area"
