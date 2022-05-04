from causal_testing.testing.estimators import LinearRegressionEstimator, Estimator

import pandas as pd

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

        width = x['width']
        height = x['height']
        intensity = x['intensity']
        # x.drop(['width', 'height', "intensity"], axis=1, inplace=True)

        self.adjustment_set = {"width*intensity", "height*intensity"}

        model = self._run_linear_regression()

        x_ = x[[self.treatment[0], "height*intensity", "width*intensity", "Intercept"]]
        y = model.predict(x_)
        treatment_outcome = y.iloc[0]
        control_outcome = y.iloc[1]

        x['predicted_outcome'] = y
        x['expected_outcome'] = 2 * intensity * (width + height)

        return treatment_outcome - control_outcome, None
