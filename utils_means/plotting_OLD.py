import os
import mplhep
mplhep.style.use('LHCb2')
import matplotlib.pyplot as plt

latex_dict_catchments = {
    '1' : r'$1 : \text{PP}\underset{2015}{\to}\text{HSG}\underset{2019}{\to}\text{Wheat}$',
    '2' : r'$2 : \text{HSG}\underset{2019}{\to}\text{Wheat}$',
    '3' : r'$3 : \text{HSG}\underset{2019}{\to}\text{Wheat}$',
    '4' : r'$4 : \text{PP}$',
    '5' : r'$5 : \text{PP}$',
    '6' : r'$6 : \text{PP}$',
    '7' : r'$7 : \text{PP}\underset{2015}{\to}\text{HSG-C}$',
    '8' : r'$8 : \text{HSG-C}$',
    '9' : r'$9 : \text{HSG-C}$',
    '10' : r'$10 : \text{PP}\underset{2015}{\to}\text{HSG}\underset{2019}{\to}\text{Wheat}$',
    '11' : r'$11 : \text{PP}\underset{2015}{\to}\text{HSG-C}$',
    '12' : r'$12 : \text{PP}$',
    '13' : r'$13 : \text{PP}$',
    '14' : r'$14 : \text{DRG-C}$',
    '15' : r'$15 : \text{DRG}\underset{2019}{\to}\text{Wheat}$',
}

units_dict = {
    'N' : 'kg',
    'P2O5' : 'kg',
    'K2O' : 'kg',
    'SO3' : 'kg',
    'MANURE' : 'kg',
    'LIME' : 'kg',
    'NitriteANDNitrate' : 'mg/l',
    'Nitrogen' : 'mg/l',
    'Ammonia' : 'mg/l',
    'Ammonium' : 'mg/l',
    'Conductivity' : 'uS/cm',
    'Dissolved Oxygen' : '%',
    'pH' : '',
    'Turbidity' : 'FNU',
    'Fluorescent Dissolved Organic Matter (ug/l QSU)' : 'ug/l QSU',
}

LaTeX_dict = {
    # event chemicals
    'N' : r'Nitrogen',
    'P2O5' : r'P$_{2}$O$_{5}$',
    'K2O' : r'K$_{2}$O',
    'SO3' : r'SO$_{3}$',
    'MANURE' : r'Manure',
    'LIME' : r'Lime',
    # runoff chemicals
    'NitriteANDNitrate' : r'NO$_{2}^-$ & NO$_{3}^-$ Runoff',
    'Ammonia' : r'NH$_{3}$ Runoff',
    'Ammonium' : r'NH$_{4}^+$ Runoff',
    'Nitrogen' : r'NO$_{2}^-$ & NO$_{3}^-$ & NH$_{3}$ & NH$_{4}^+$ Runoff',
    'Conductivity' : r'Conductivity',
    'Dissolved Oxygen' : r'Dissolved Oxygen',
    'pH' : r'pH',
    'Turbidity' : r'Turbidity',
    'Fluorescent Dissolved Organic Matter (ug/l QSU)' : r'Fluorescent Dissolved Organic Matter',
}

month_LaTeX = {
    1: r'Jan.',
    2: r'Feb.',
    3: r'Mar.',
    4: r'Apr.',
    5: r'May',
    6: r'Jun.',
    7: r'Jul.',
    8: r'Aug.',
    9: r'Sep.',
    10: r'Oct.',
    11: r'Nov.',
    12: r'Dec.'
}

def plot_results(start_year, end_year, results_dict, runoff_chemical, valid_months):
    means_dict = results_dict['means']
    stds_dict = results_dict['stds']
    errs_upper_dict = results_dict['std_upper']
    errs_lower_dict = results_dict['std_lower']

    if valid_months is None:
        title_str = f'Months: Jan. - Dec.'
    elif valid_months[0] == valid_months[1]:
        title_str = f'Months: {month_LaTeX[valid_months[0]]}'
    else:
        title_str = f'Months: {month_LaTeX[valid_months[0]]} - {month_LaTeX[valid_months[1]]}'

    # make the years strings
    if valid_months is None:
        years = [year for year in range(start_year, end_year + 1)]
    else:
        if valid_months[1] < valid_months[0]:
            years = [f'{year}/{year+1-2000}' for year in range(start_year, end_year + 1)][:-1]
        else:
            years = [year for year in range(start_year, end_year + 1)]

    plotting_folders = ['plotting/means_stds', 'plotting/means', 'plotting/stds']
    for folder in plotting_folders:
        os.makedirs(folder, exist_ok=True)

    fig_savename = f'{runoff_chemical}_{".".join(means_dict.keys())}_{start_year}-{end_year}.pdf'

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    for catchment in means_dict.keys():
        y = means_dict[catchment]
        yerr = stds_dict[catchment]
        yerr_upper = errs_upper_dict[catchment]
        yerr_lower = errs_lower_dict[catchment]
        # set cap length of 20
        
        # assume symmetric errors
        #ax.errorbar(years, y, yerr=yerr, label=latex_dict_catchments[catchment], fmt='D', linestyle='none', capsize=20)

        # assuming asymmetric errors
        ax.errorbar(years, y, yerr=[yerr_lower, yerr_upper], label=latex_dict_catchments[catchment], fmt='D', linestyle='none', capsize=20)

        if units_dict[runoff_chemical] != '':
            ax.set_ylabel(f'{LaTeX_dict[runoff_chemical]} [{units_dict[runoff_chemical]}]', loc='center')
        else:
            ax.set_ylabel(f'{LaTeX_dict[runoff_chemical]}', loc='center')
        
        ax.set_xlabel('Year', loc='center')

    ax.set_title(title_str)
    ax.legend()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
    fig.savefig(f'plotting/means_stds/{fig_savename}')

    return