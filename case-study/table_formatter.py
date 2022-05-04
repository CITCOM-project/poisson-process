import pandas as pd

data = pd.read_csv("intensity_num_shapes_results_random_1000.csv", index_col=0)
# data = data[(data.width == 1) | (data.width == 10)]
data['intervention'] = [f"${c} \\to {t}$" for c, t in zip(data['control'], data['treatment'])]
data = data.drop(['control', "treatment"], axis=1)
smt_data = data.pivot_table('smt_risk_ratio', ['width', "height"], 'intervention')
smt_data.to_latex("intensity_num_shapes_results_random_1000.tex", escape=False, float_format="%.4f")

# Take one width=height value. Observational is all the same
obs_risk_ratio = data.pivot_table('obs_risk_ratio', ["width", "height"], 'intervention')
obs_risk_ratio.to_latex("intensity_num_shapes_results_random_1000_obs.tex", escape=False, float_format="%.4f")

wh_data = pd.read_csv("width_num_shapes_results_random_1000.csv", index_col=0)
wh_data = wh_data.where(wh_data['intensity'] == 1).dropna()
wh_data = wh_data[['control', 'treatment', 'ci_low', 'ate', 'ci_high']]
wh_data['control'] = wh_data['control'].astype(int)
wh_data['treatment'] = wh_data['treatment'].astype(int)
wh_data.to_latex("width_num_shapes_results_random_1000_obs.tex", index=False, escape=False, float_format="%.4f")
print(wh_data)
# wh_data['intervention'] = [f"${c} \\to {t}$" for c, t in zip(wh_data['control'], wh_data['treatment'])]
# wh_data['result'] = [f"${round(cl, 4)} < {round(a, 4)} < {round(ch, 4)}$" for cl, a, ch in zip(wh_data['ci_low'], wh_data['ate'], wh_data['ci_high'])]
# wh_data = wh_data.drop(['control', "treatment"], axis=1)
# print(wh_data)
#
# pivot_data = wh_data.pivot_table('result', ['intensity'], 'intervention', aggfunc=lambda x: x)
# pivot_data.to_latex("width_num_shapes_results_random_1000_obs.tex", escape=False, float_format="%.4f")
#
# print(pivot_data)
