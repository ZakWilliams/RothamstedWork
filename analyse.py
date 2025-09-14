# Open the precipitation
# Open the flow
# Prove theres a link/find which catchment has the smallest 'lag' time?

# Also, come up with some questions/plans based on the skim

import numpy as np
import pandas as pd
from termcolor import cprint, colored
from fit_utils import fit_DCB, fit_EXP

flow_measures_address = 'data/flows_2020.csv'
precipitation_address = 'data/precipitations_2020.csv'

flow_df = pd.read_csv(flow_measures_address)
flow_keys = list(flow_df.keys())
flow_df = flow_df[[flow_key for flow_key in flow_keys if 'Quality Last Modified' not in flow_key]]

precipitation_df = pd.read_csv(precipitation_address)

y_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
catchment_number = 4

for df in [flow_df, precipitation_df]:
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y/%m/%d %H:%M:%S')

# Preprocess the flow df
cprint('Fraction "Acceptable" data quality in:', 'magenta')
for catchment_num in range(1, 16):
    flow_key = [key for key in flow_df.columns if ((f'Catchment {catchment_num}' in key) and ('Quality' not in key))][0]
    quality_key = [key for key in flow_df.columns if ((f'Catchment {catchment_num}' in key) and ('Quality' in key))][0]
    flow_df[f'C{catchment_num}_FLOW'] = flow_df[flow_key]
    flow_df[f'C{catchment_num}_DQ'] = (flow_df[quality_key] == 'Acceptable')
    catchment_str = f'Catchment {catchment_num} Flow:'
    # Isolate quality into only selected month(s)
    month_DQ = np.mean(flow_df[flow_df['Datetime'].dt.month.isin(y_months)][f'C{catchment_num}_DQ'])

    cprint(f'{catchment_str:<22} {month_DQ:.4f}', 'magenta')
    # Drop flow_key and quality_key from df
    flow_df = flow_df.drop(columns=[flow_key, quality_key])

# Do something else with the data down here
precipitation_df['PRECIP'] = precipitation_df['Precipitation (mm) [Site]']
precipitation_df['PRECIP_DQ'] = (precipitation_df['Precipitation (mm) [Site] Quality'] == 'Acceptable')
precipitation_df = precipitation_df[['Datetime', 'PRECIP', 'PRECIP_DQ']]
cprint(f'Precipitation Dataset: {np.mean(precipitation_df["PRECIP_DQ"]):.4f}', 'cyan')

# Check all the datetimes line up between the dfs
datetimes_flow = flow_df['Datetime'].dt.to_pydatetime()
datetimes_precip = precipitation_df['Datetime'].dt.to_pydatetime()
if np.any(datetimes_flow != datetimes_precip):
    raise ValueError(colored("Datetimes do not match between flow and precipitation datasets. Failing...", "red"))

# join the dfs
df = pd.merge(flow_df, precipitation_df, on='Datetime')
quit()


fit_params_dict, cost_recorder = fit_EXP(df, catchment_number=catchment_number, pad_days=4, y_months=y_months, refit=False)

# plot the result of the basic exponential model down here
lam = fit_params_dict['lambda']
mag = fit_params_dict['mag']


# plot the result of the minimisation down here







