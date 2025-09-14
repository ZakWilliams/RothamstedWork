import os
import pickle 
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from termcolor import cprint

def EXP(x, log_lam, mag):
    lam = tf.exp(log_lam)  # force positivity
    return mag * lam * tf.exp(-lam * x)

def EXP_np(x, lam, mag):
    return mag * lam * np.exp(-lam * x)

def CrystalBall(x, mu, std, n, alpha, mag):
    xR = (x - mu) / std

    A = tf.pow(n / tf.abs(alpha), n) * tf.exp(-0.5 * alpha**2)
    B = (n / tf.abs(alpha)) - tf.abs(alpha)

    y = tf.where(
        xR >= -alpha,
        tf.exp(-0.5 * xR**2),
        A * tf.pow(B - xR, -n)
    )

    return y * mag

def calc_cost_DCB(x_PRECIP, y_FLOW, fit_params):
    mu, std, n, alpha, mag = fit_params

    Nx = len(x_PRECIP)
    Ny = len(y_FLOW)

    x_lin = tf.cast(tf.linspace(0, 10, Nx-Ny), dtype=tf.float32)

    weights = CrystalBall(x_lin, mu, std, n, alpha, mag)

    weighted_sum_list = []
    for idx in range(Ny):
        x_slice = x_PRECIP[idx:idx+len(weights)]
        weighted_sum_list.append(tf.reduce_sum(weights * x_slice))

    weighted_sum = tf.stack(weighted_sum_list)

    cost = tf.reduce_sum((y_FLOW - weighted_sum)**2)

    return cost

def calc_cost_EXP(x_PRECIP, y_FLOW, fit_params):
    lam, mag, lam2, mag2, C = fit_params
    Nx = len(x_PRECIP)
    Ny = len(y_FLOW)
    
    x_lin = tf.cast(tf.linspace(0, 10, Nx-Ny), dtype=tf.float32)
    
    weights = EXP(x_lin, lam, mag)
    weights += EXP(x_lin, lam2, mag2)
    weighted_sum_list = []
    for idx in range(Ny):
        x_slice = x_PRECIP[idx:idx+len(weights)]
        weighted_sum_list.append(tf.reduce_sum(weights * x_slice))

    weighted_sum = tf.stack(weighted_sum_list)
    weighted_sum += C

    cost = tf.reduce_sum((y_FLOW - weighted_sum)**2)

    return cost

def fit_DCB(df, catchment_number, y_months, pad_days=7, refit=True):
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.01)
    
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
    x_PRECIP = df['PRECIP'][(df['Datetime'] >= min_x_datetime) & (df['Datetime'] <= max_x_datetime)].values # x_data is precipitation
    y_FLOW = df[f'C{catchment_number}_FLOW'][(df['Datetime'] >= min_y_datetime) & (df['Datetime'] <= max_y_datetime)].values # y data is flow


    print_per_epoch = 100
    training_epochs = 1000
    
    fit_params_dict = {
        'mu' : tf.Variable(2.0),
        'std' : tf.Variable(1.0),
        'n' : tf.Variable(1.0),
        'alpha' : tf.Variable(0.1),
        'mag' : tf.Variable(1.0)
    }
    fit_params = [fit_params_dict[key] for key in fit_params_dict]

    pickle_address = f'fit_params_{catchment_number}_{pad_days}_{'.'.join([str(month) for month in y_months])}.pkl'

    cost_recorder = []
    if (not os.path.exists(pickle_address)) or refit:
        for epoch in range(training_epochs):
            with tf.GradientTape() as tape:
                cost = calc_cost_DCB(x_PRECIP, y_FLOW, fit_params)

                if epoch % print_per_epoch == 0:
                    epoch_frac = f'{epoch}/{training_epochs}'
                    cprint(f'Epoch {epoch_frac:<{16}} {"":<{4}} Cost: {cost}', 'light_grey')

                # Apply gradients
                gradients = tape.gradient(cost, fit_params)
                optimiser.apply_gradients(zip(gradients, fit_params))

                cost_recorder.append(cost.numpy())
    else:
        with open(pickle_address, 'rb') as f:
            fit_params_dict, cost_recorder = pickle.load(f)
        return fit_params_dict, cost_recorder

    # save fit_params_dict, cost_recorder to pickle
    with open(pickle_address, 'wb') as f:
        pickle.dump((fit_params_dict, cost_recorder), f)

    return fit_params_dict, cost_recorder

def fit_EXP(df, catchment_number, y_months, pad_days=7, refit=True):
    optimiser = tf.keras.optimizers.Adam(learning_rate=0.01)
    
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
    x_PRECIP = df['PRECIP'][(df['Datetime'] >= min_x_datetime) & (df['Datetime'] <= max_x_datetime)].values # x_data is precipitation
    y_time = df['Datetime'][(df['Datetime'] >= min_y_datetime) & (df['Datetime'] <= max_y_datetime)].values
    y_FLOW = df[f'C{catchment_number}_FLOW'][(df['Datetime'] >= min_y_datetime) & (df['Datetime'] <= max_y_datetime)].values # y data is flow




    print_per_epoch = 100
    training_epochs = 1500
    
    fit_params_dict = {
        'lambda' : tf.Variable(0.1), # Maybe this needs tweaking?
        'mag' : tf.Variable(1.0),
        'C' : tf.Variable(1.0),
        'lambda2' : tf.Variable(0.1),
        'mag2' : tf.Variable(1.0)
    }
    fit_params = [fit_params_dict[key] for key in fit_params_dict]
    
    cost_recorder = []
    if (not os.path.exists(f'fit_params_{catchment_number}.pkl')) or refit:
        for epoch in range(training_epochs):
            with tf.GradientTape() as tape:
                cost = calc_cost_EXP(x_PRECIP, y_FLOW, fit_params)

                if epoch % print_per_epoch == 0:
                    epoch_frac = f'{epoch}/{training_epochs}'
                    cprint(f'Epoch {epoch_frac:<{16}} {"":<{4}} Cost: {cost}', 'light_grey')

                # Apply gradients
                gradients = tape.gradient(cost, fit_params)
                optimiser.apply_gradients(zip(gradients, fit_params))

                cost_recorder.append(cost.numpy())
    else:
        with open(f'fit_params_{catchment_number}.pkl', 'rb') as f:
            fit_params_dict, cost_recorder = pickle.load(f)

    # save fit_params_dict, cost_recorder to pickle
    with open(f'fit_params_{catchment_number}.pkl', 'wb') as f:
        pickle.dump((fit_params_dict, cost_recorder), f)

    # Print key, value pairs
    cprint(f'Fit Values:')
    for key, value in fit_params_dict.items():
        cprint(f'{key}: {value.numpy()}', 'magenta')

    # plot the result/prediction
    p_FLOW = []
    Ny = len(y_FLOW)
    Nx = len(x_PRECIP)
    x_PRECIP_plotting = x_PRECIP[Nx-Ny-1:-1]
    x_lin = np.linspace(0, 10, Nx-Ny)
    weights = EXP_np(x_lin, fit_params_dict['lambda'], fit_params_dict['mag'])
    weights += EXP_np(x_lin, fit_params_dict['lambda2'], fit_params_dict['mag2'])
    for idx, y_val in enumerate(y_FLOW):
        x_slice = x_PRECIP[idx:idx+len(weights)]
        weighted_sum = np.sum(weights * x_slice)
        p_FLOW.append(weighted_sum)
    p_FLOW = tf.stack(p_FLOW)
    p_FLOW += fit_params_dict['C']

    fig = plt.figure(figsize=(12, 10))
    ax_precip = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax = ax_precip.twinx()
    ax_precip.fill_between(y_time, x_PRECIP_plotting, color='blue', alpha=0.9, lw=0.)

    sum_y = np.sum(y_FLOW)
    sum_pred = np.sum(p_FLOW)
    
    p_FLOW *= (sum_y / sum_pred)

    ax.plot(y_time, y_FLOW, label='Observed', color='red', zorder=0.4)
    ax.plot(y_time, p_FLOW, label='Predicted', color='orange', zorder=0.5, linestyle='--')
    ax.set_xlabel('Time')
    ax_precip.set_ylabel(r'Precipitation [mm/hour]')
    ax.set_ylabel(r'Flow [L/s]')
    ax.set_xlim([y_time[0], y_time[-1]])
    ax.legend()
    ax.set_title(f'Catchment {catchment_number} - Pad Days: {pad_days}')
    fig.savefig(f'fit_results_{catchment_number}.pdf')

    return fit_params_dict, cost_recorder