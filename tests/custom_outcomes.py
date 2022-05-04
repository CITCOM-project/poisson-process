from causal_testing.testing.causal_test_outcome import CausalTestOutcome, CausalTestResult

import numpy as np

class PoissonWidthHeight(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be positive."""

    def __init__(self, tolerance=0.5):
        self.tolerance = tolerance
        self.i2c = None

    def apply(self, res: CausalTestResult) -> bool:
        # TODO: confidence intervals?
        effect_modifier_configuration = {k.name: v for k, v in res.effect_modifier_configuration.items()}
        c = res.treatment_value - res.control_value
        i = effect_modifier_configuration['intensity']
        self.i2c = i*2*c
        return np.isclose(res.ate, self.i2c, atol=self.tolerance)

    def __str__(self):
        if self.i2c is None:
            return f"PoissonWidthHeight±{self.tolerance}"
        else:
            return f"PoissonWidthHeight:{self.i2c}±{self.tolerance}"


class PoissonIntensity(CausalTestOutcome):
    """An extension of TestOutcome representing that the expected causal effect should be positive."""

    def __init__(self, tolerance=0.5):
        self.tolerance = tolerance
        self.chw2 = None

    def apply(self, res: CausalTestResult) -> bool:
        # TODO: confidence intervals?
        effect_modifier_configuration = {k.name: v for k, v in res.effect_modifier_configuration.items()}
        c = res.treatment_value - res.control_value
        h_w = effect_modifier_configuration['height'] + effect_modifier_configuration['width']
        self.chw2 = 2*c*h_w
        return np.isclose(res.ate, self.chw2, atol=self.tolerance)

    def __str__(self):
        if self.chw2 is None:
            return f"PoissonIntensity±{self.tolerance}"
        else:
            return f"PoissonIntensity: {self.chw2}±{self.tolerance}"
