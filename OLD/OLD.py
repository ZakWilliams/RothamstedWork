import pandas as pd
import numpy as np
from lmfit import Model
from termcolor import cprint
from scipy.signal import fftconvolve




# Example: fake data







def fit_DCB(df, catchment_number, y_months, pad_days=7):

    df['Datetime_as_yearfrac'] = (df['Datetime'].dt.month - 1) / 12 + (df['Datetime'].dt.day - 1) / 365 + (df['Datetime'].dt.hour) / 8760 + (df['Datetime'].dt.minute) / 525600

    # y_time_mask
    y_time_mask = df['Datetime'].dt.month.isin(y_months)
    # x_time_mask
    min_y_datetime = df['Datetime'][y_time_mask].dt.to_pydatetime().min()
    max_y_datetime = df['Datetime'][y_time_mask].dt.to_pydatetime().max()
    # Subtract
    min_x_datetime = min_y_datetime - pd.Timedelta(days=pad_days)
    max_x_datetime = max_y_datetime

    time = df['Datetime_as_yearfrac'][(df['Datetime'] >= min_x_datetime) & (df['Datetime'] <= max_x_datetime)].values
    x_data = df['PRECIP'][(df['Datetime'] >= min_x_datetime) & (df['Datetime'] <= max_x_datetime)].values # x_data is precipitation
    y_data = df[f'C{catchment_number}_FLOW'][(df['Datetime'] >= min_y_datetime) & (df['Datetime'] <= max_y_datetime)].values # y data is flow

    def crystal_ball_kernel(N, mu, sigma, alpha, n):
        lags = np.arange(N)
        z = (lags - mu) / sigma
        A = (n / abs(alpha))**n * np.exp(-0.5 * alpha**2)
        B = n / abs(alpha) - abs(alpha)
        gauss = np.exp(-0.5 * z**2)
        tail = A * np.power(np.maximum(B - z, 1e-12), -n)  # clamp to avoid NaNs
        cb = np.where(z > -alpha, gauss, tail)
        cb /= cb.sum() if cb.sum() != 0 else 1
        return cb

    def convolved_model(idx_array, mu, sigma, alpha, n, ampl):
        Nx = len(x_data)
        Ny = len(y_data)
        kernel = crystal_ball_kernel(Nx - Ny + 1, mu, sigma, alpha, n) * ampl
        conv_full = fftconvolve(x_data, kernel, mode='full')
        offset = Nx - Ny
        return conv_full[offset : offset + Ny]

    Nx = len(x_data)
    Ny = len(y_data)
    # When creating the lmfit Model, set param bounds
    model = Model(convolved_model, independent_vars=['idx_array'])
    params = model.make_params(mu=0, sigma=3, alpha=2, n=3, ampl=1.0)
    params['sigma'].min = 1e-6     # avoid division by zero
    params['alpha'].min = 1e-6     # avoid division by zero
    params['n'].min = 0.1          # keep exponent well-defined
        
    result = model.fit(y_data, params, idx_array=np.arange(Ny))
    print(result.fit_report())

    return