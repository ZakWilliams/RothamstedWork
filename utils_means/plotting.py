import numpy as np
import os
import mplhep
import warnings
mplhep.style.use('LHCb2')
import matplotlib.pyplot as plt
from termcolor import cprint

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

def plot_results(start_year, end_year, percentiles_dict, runoff_chemical, valid_months, quality_assessment):






    # title string
    if valid_months is None:
        title_str = f'Months: Jan. - Dec.'
    elif valid_months[0] == valid_months[1]:
        title_str = f'Months: {month_LaTeX[valid_months[0]]}'
    else:
        title_str = f'Months: {month_LaTeX[valid_months[0]]} - {month_LaTeX[valid_months[1]]}'

    plotting_folders = ['plotting/percentiles']
    for folder in plotting_folders:
        os.makedirs(folder, exist_ok=True)

    fig_savename = f'{runoff_chemical}_{".".join(percentiles_dict.keys())}_{start_year}-{end_year}.pdf'

    fig = plt.figure(figsize=(12, 10))

    if len(list(quality_assessment.keys())) == 2:
        ax = fig.add_axes([0.1, 0.3, 0.8, 0.9])
        ax_QUAL = fig.add_axes([0.1, 0.1, 0.8, 0.2])
    else:
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])



    # unpack percentiles into a more plottable fillbetween format
    # make fillbetweens
    fillbetween_lower_dict = {}
    fillbetween_upper_dict = {}
    xlims_to_match_fillbetween = {}
    
    for catchment in percentiles_dict.keys():
        fillbetween_lower_dict[catchment] = {}
        fillbetween_upper_dict[catchment] = {}
        xlims_to_match_fillbetween[catchment] = {}

        for year in percentiles_dict[catchment].keys():
            percs = percentiles_dict[catchment][year]
            if np.isnan(percs).any():
                cprint(f'Skipping NaN...', 'yellow')
                fillbetween_lower_dict[catchment][year] = np.array([])
                fillbetween_upper_dict[catchment][year] = np.array([])
                xlims_to_match_fillbetween[catchment][year] = np.array([])
            else:
                #cprint(percs, 'cyan')
                #cprint(len(percs), 'red')
                length = len(percs)
                first_half = percs[:int(length/2)]
                second_half = percs[int(length/2):]

                fillbetween_lower_dict[catchment][year] = np.concatenate([first_half[::-1], first_half])
                fillbetween_upper_dict[catchment][year] = np.concatenate([second_half, second_half[::-1]])
                xlims_to_match_fillbetween[catchment][year] = np.linspace(year-0.5, year+0.5, num=len(fillbetween_lower_dict[catchment][year]))

    color_list = ['blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color_dict = {}
    # build a catchment color dict due to how color assignment works in matplotlib
    for i, catchment in enumerate(percentiles_dict.keys()):
        color_dict[catchment] = color_list[i % len(color_list)]

    # do the actual plotting
    potential_start_year = []
    for catchment in percentiles_dict.keys():
        for year in percentiles_dict[catchment].keys():
            if year == list(percentiles_dict[catchment].keys())[0]:
                ax.fill_between(xlims_to_match_fillbetween[catchment][year],
                                fillbetween_lower_dict[catchment][year],
                                fillbetween_upper_dict[catchment][year],
                                color=color_dict[catchment], alpha=0.5, lw=0.,
                                label=latex_dict_catchments[catchment])
            else:
                ax.fill_between(xlims_to_match_fillbetween[catchment][year],
                                fillbetween_lower_dict[catchment][year],
                                fillbetween_upper_dict[catchment][year],
                                color=color_dict[catchment], alpha=0.5, lw=0.)
            potential_start_year.append(np.mean(xlims_to_match_fillbetween[catchment][year]))

    potential_start_year = np.array(potential_start_year)
    potential_start_year = potential_start_year[~np.isnan(potential_start_year)]
    start_year = round(np.min(potential_start_year))
    end_year = round(np.max(potential_start_year))
    
    # for the quality assesment masks, find the overlap and the XORs for each catchment
    quality_plot_dict = {}
        
    # this will only be plotted when there are 2 catchments only
    for year in quality_assessment[catchment].keys():
        quality_plot_dict[year] = {}
        masks_temp = {catchment : quality_assessment[catchment][year] for catchment in quality_assessment.keys()}
        pass_mask = np.ones_like(masks_temp[catchment], dtype=bool)
        none_pass = np.zeros_like(masks_temp[catchment], dtype=bool)
        for catchment in masks_temp.keys():
            pass_mask &= masks_temp[catchment]
            none_pass |= masks_temp[catchment]
        none_pass = ~none_pass
        ONLY_one_pass_mask = {catchment : masks_temp[catchment] & ~pass_mask for catchment in masks_temp.keys()}
        quality_plot_dict[year]['none'] = none_pass
        quality_plot_dict[year]['overlap'] = pass_mask
        quality_plot_dict[year]['only_one'] = ONLY_one_pass_mask

    for year in quality_assessment[catchment].keys():
        cprint(np.mean(quality_plot_dict[year]['overlap']), 'green')
        cprint(np.mean(quality_plot_dict[year]['none']), 'red')
        for catchment in quality_plot_dict[year]['only_one'].keys():
            cprint(f'{catchment} : {np.mean(quality_plot_dict[year]['only_one'][catchment])}', 'blue')





    xtick_values = range(start_year, end_year + 1)
    # build the xtick_labels:
    if valid_months is None:
        xtick_labels = [str(year) for year in xtick_values]
    else:
        if valid_months[1] > valid_months[0]:
            xtick_labels = [str(year) for year in xtick_values]
        else:
            xtick_labels = [f'{year}/{year-1999}' for year in xtick_values]

    # fetch ax lims:

    if units_dict[runoff_chemical] != '':
        ax.set_ylabel(f'{LaTeX_dict[runoff_chemical]} [{units_dict[runoff_chemical]}]', loc='center')
    else:
        ax.set_ylabel(f'{LaTeX_dict[runoff_chemical]}', loc='center')
        
    if len(list(quality_assessment.keys())) == 2:
        xlims = [xtick_values[0]-0.5, xtick_values[-1]+0.5]
        ax.set_xlim(xlims)
        ax_QUAL.set_xlim(xlims)
        
        ax.set_xticks(xtick_values)
        ax.set_xticklabels([])
        ax_QUAL.set_xticks(xtick_values)
        ax_QUAL.set_xticklabels(xtick_labels)
        ax_QUAL.set_xticklabels(ax_QUAL.get_xticklabels(), rotation=90, ha="center")
        ax_QUAL.set_ylabel('Q.T.', loc='center')
        ax_QUAL.set_yticklabels([])
        for year in percentiles_dict[catchment].keys():
            lower_val = 0.
            # fill between
            ax_QUAL.fill_between([year-0.5, year+0.5], [lower_val, lower_val], [np.mean(quality_plot_dict[year]['overlap']), np.mean(quality_plot_dict[year]['overlap'])],
                            color='green', lw=0., alpha=1.0)
            lower_val += np.mean(quality_plot_dict[year]['overlap'])
            for catchment in quality_plot_dict[year]['only_one'].keys():
                ax_QUAL.fill_between([year-0.5, year+0.5],
                                [lower_val, lower_val],
                                [lower_val+np.mean(quality_plot_dict[year]['only_one'][catchment]), lower_val+np.mean(quality_plot_dict[year]['only_one'][catchment])],
                                color=color_dict[catchment], lw=0., alpha=1.0)
                lower_val += np.mean(quality_plot_dict[year]['only_one'][catchment])
            ax_QUAL.fill_between([year-0.5, year+0.5], [lower_val, lower_val], [1.0, 1.0], color='red', lw=0., alpha=1.0)

        ax_QUAL.set_ylim([0., 1.])



    else:
        ax.set_xticks(xtick_values)
        ax.set_xticklabels(xtick_labels)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

        xlims = [xtick_values[0]-0.5, xtick_values[-1]+0.5]
        ax.set_xlim(xlims)
    
    
    

    ax.set_title(title_str)
    ax.legend()
    ax.set_ylim([0., None])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(f'plotting/percentiles/{fig_savename}')

    return