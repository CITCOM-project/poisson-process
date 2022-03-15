# Poisson Line Process
A simple model of a poisson line process `poisson_line_process.py` kindly for us written by [Benedikt Prifling](https://www.uni-ulm.de/mawi/mawi-stochastik/mitarbeiter/mitarbeiter/benedikt-prifling/).
The file `poisson_line_data.py` runs the model to collect data in `tests/data.csv`, which can then be used to evaluate the tests.

The file `tests/causal_tests.json` describes the abstract tests. This can be run by calling `python process_causal_tests_json.py`. This will then generate and execute concrete tests using the data in `data.csv`.
