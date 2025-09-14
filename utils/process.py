# Take the by-year split data

# build a by-catchment dict with compact column names

import os
import warnings
import numpy as np
import pandas as pd
from termcolor import cprint
from matplotlib.ticker import NullLocator
from utils.bank import conversion_dict

import matplotlib.pyplot as plt
#'Soil Moisture @ 10cm Depth (%) [Catchment 1]', 'Soil Moisture @ 10cm Depth (%) [Catchment 1] Quality', 'Soil Moisture @ 10cm Depth (%) [Catchment 1] Quality Last Modified'


def safe_read_csv(filepath):
    try:
        return pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(filepath, encoding="latin-1")

def build_catchment_dict(catchments, start_month, end_month):
    start_year = int(start_month[3:])
    end_year = int(end_month[3:])
    years = list(range(start_year, end_year + 1))
    
    cprint(f'Building catchment dict...', 'cyan')
    # put an empty list in each which will be populated with dataframes, which will then be concatenated at the end of the years loop
    catchment_dict = {str(catchment): [] for catchment in catchments}
    for year in years:
        cprint(f'Processing year {year}...', 'cyan')
        df = safe_read_csv(f'store/{year}.csv')
        # Process the DataFrame to create a compact representation split by catchment, rather than by year
        for catchment in catchments:
            catchment_columns = [col for col in df.columns if f'[Catchment {catchment}]' in col]
            keep_catchment_columns = [col for col in catchment_columns if np.any([col.startswith(key) for key in conversion_dict.keys()])]
            catchment_df = df[['Datetime'] + keep_catchment_columns].copy()
            # build the column renamer using conversion_dict
            renaming_dict = {}
            for key, value in conversion_dict.items():
                renaming_dict[f'{key} [Catchment {catchment}]'] = f'{value}'
                renaming_dict[f'{key} [Catchment {catchment}] Quality'] = f'{value} DQ'
                renaming_dict[f'{key} [Catchment {catchment}] Quality Last Modified'] = f'{value} DQLM'

            catchment_df.rename(columns=renaming_dict, inplace=True)

            catchment_dict[str(catchment)].append(catchment_df)

            #cprint(f'{list(catchment_df.columns)}\n', 'red')

    cprint(f'Concatenating years...', 'cyan')
    for catchment in catchments:
        catchment_dict[str(catchment)] = pd.concat(catchment_dict[str(catchment)], ignore_index=True)

    # Convert datetime strings to datetime objects
    cprint(f'Converting datetime strings to datetime objects...', 'cyan')
    for catchment in catchments:
        catchment_dict[str(catchment)]['Datetime'] = pd.to_datetime(catchment_dict[str(catchment)]['Datetime'], format='%Y/%m/%d %H:%M:%S')

    cprint(f'Time-united catchment-split dict created.', 'green')

    return catchment_dict

def build_weight_multipliers(df):
    weight_multipliers = np.zeros(len(df))
    # if df['Units'] == 'kg':
    weight_multipliers = np.where(df['Units'] == 'kg', 1, weight_multipliers)
    # if df['Units'] == 't':
    weight_multipliers = np.where(df['Units'] == 't', 1000, weight_multipliers)
    # if df['Units'] == 'l', I think % refer to 'by fertiliser' - check this later
    weight_multipliers = np.where(df['Units'] == 'l', 1, weight_multipliers)

    # elona - https://levitycropscience.co.uk/wp-content/uploads/2023/07/Elona-UK-Label.pdf - density approximated as 1.3444 kg/l
    #weight_multipliers = np.where((df['Units'] == 'l') & (df['Product_Name'].str.contains('Elona')), 1.3444, weight_multipliers)
    # EfficieNt - https://www.agro-vital.co.uk/efficient28 - density approximated as 1.25 kg/l
    #weight_multipliers = np.where((df['Units'] == 'l') & (df['Product_Name'].str.contains('EfficieNt')), 1.25, weight_multipliers)
    # KuruS - https://data.fmc-agro.co.uk/wp-content/uploads/KURUS-10L-UK-R-11Mar20.pdf - density approximated as 1.0 kg/l
    #weight_multipliers = np.where((df['Units'] == 'l') & (df['Product_Name'].str.contains('KuruS')), 1.0, weight_multipliers)
    # Fielder Magnesium - https://assets.vp-cdn.co.uk/BXPwmxGo37XaVOe1Mzgr/g16AIJGQdRUISLt3KJ9bNqsaQnjv7eFyS5UWlaxj.pdf - density approximated as 1.38 kg/l
    #weight_multipliers = np.where((df['Units'] == 'l') & (df['Product_Name'].str.contains('Fielder Magnesium')), 1.38, weight_multipliers)
    # Profol plus - https://bfsfertiliserservices.uk/media/content/files/application-charts/profol-plus1.pdf - density calculated as 1.13378684807 kg/l
    #weight_multipliers = np.where((df['Units'] == 'l') & (df['Product_Name'].str.contains('Profol plus')), 1.13378684807, weight_multipliers)
    # Root 66 - https://www.agro-vital.co.uk/root-66 - density approximated as 1.35 kg/l
    #weight_multipliers = np.where((df['Units'] == 'l') & (df['Product_Name'].str.contains('Root 66')), 1.35, weight_multipliers)

    return weight_multipliers


def build_catchment_fertilisation_dict(catchments, start_month, end_month):
    start_year = int(start_month[3:])
    end_year = int(end_month[3:])
    years = list(range(start_year, end_year + 1))
    
    print(f'Building fertilisation dict...', 'cyan')
    catchment_dict = {str(catchment): [] for catchment in catchments}
    catchment_field_dict = {
        '1' : ['NW001', 'NW038', 'NW047'],
        '2' : ['NW002'],
        '3' : ['NW003', 'NW004'],
        '10' : ['NW015'],
        '15' : ['NW019']
    }
    
    catchment_fertilisation_dict = {str(catchment): [] for catchment in catchments}
    for year in years:
        cprint(f'Processing year {year}...', 'cyan')
        df = safe_read_csv(f'store/{year}_EVENTS.csv')
        fertiliser_applied_mask = df['Field_Operation'].values == 'Apply Fertiliser'
        df = df[fertiliser_applied_mask]

        keep_cols = ['Datetime', 'Field', 'Catchment', 'Manure', 'Lime', 'N', 'P2O5', 'K2O', 'SO3']

        # Build Datetime
        time_in = pd.to_timedelta(df['Time_IN'].fillna("12:00")+":00") # assumed that nan ttimes happened at midday
        time_out = pd.to_timedelta(df['Time_Out'].fillna("12:00")+":00") # assumed that nan ttimes happened at midday
        df['Datetime'] = pd.to_datetime(df['Event_Date'], format='%d/%m/%Y')
        time_average = (time_in + time_out) / 2
        df['Datetime'] = df['Datetime'] + time_average
        #cprint(df['Datetime'], 'blue')
        
        
        # amount of each chemical in inorganic
        # amount of manure (assume that this is roughly consistent in its chemical makeup?)
        open_bracket = [str(_).find('(') for _ in df['Application_Info'].values]
        close_bracket = [str(_).find(')') for _ in df['Application_Info'].values]
        
        fertiliser_info = [str(_[open_bracket[i]+1:close_bracket[i]]) if open_bracket[i] != -1 and close_bracket[i] != -1 else 'OTHER' for i, _ in enumerate(df['Application_Info'].values)]
        fertiliser_info = [_.replace('SO3%', 'SO3').replace('Mgo', 'MgO') for _ in fertiliser_info]
        # where fertiliser info reads other, if Product_Name == Elona or Elona Top, replace that OTHER with '9% N; 1.66% MgO'
        elona_mask = (df['Product_Name'].values == 'Elona') | (df['Product_Name'].values == 'Elona Top')
        fertiliser_info = [_.replace('OTHER', '9% N; 1.66% MgO') if elona_mask[i] else _ for i, _ in enumerate(fertiliser_info)]
        # Replace the faulty Kieserite chemical composition with the correct one
        fertiliser_info = ['27% MgO, 50% SO3' if df['Product_Name'].values[i] == 'Kieserite' else _ for i, _ in enumerate(fertiliser_info)]
        # replace any of the manure configurations with MANURE
        manure_mask = df['Product_Name'].values == 'Farmyard Manure'
        fertiliser_info = ['MANURE' if manure_mask[i] else _ for i, _ in enumerate(fertiliser_info)]
        # replace any of the liming configurations with LIMING
        lime_mask = (df['Product_Name'].values == 'Lime') | (df['Product_Name'].values == 'Granulated lime')
        fertiliser_info = ['LIME' if lime_mask[i] else _ for i, _ in enumerate(fertiliser_info)]
        fertiliser_info = [_.replace(',', ';') for _ in fertiliser_info]
        # Add in options for manure and Liming
        chemicals = ['N', 'P2O5', 'K2O', 'SO3']
        # I want 6 columns, one for each of the 'chemicals' displaying the total kg added
        # Manure is either 1.0 or 0.0
        df['MANURE'] = np.where(df['Product_Name'].values == 'Farmyard Manure', 1.0, 0.0)
        df['LIME'] = np.where((df['Product_Name'].values == 'Lime') | (df['Product_Name'].values == 'Granulated lime'), 1.0, 0.0)
        # Other chemicals are more complex, broken down by specific fertiliser composition
        fertiliser_info_split = [_.split(';') for _ in fertiliser_info]
        for chemical in chemicals:
            # isolate the numeric string
            chemical_strs = []
            for chemical_set in fertiliser_info_split:
                # find which of the individual strings in the array features the chemical
                found_flag = False
                for chemical_info in chemical_set:
                    if (chemical in chemical_info) and ('MANURE' not in chemical_info) and ('LIME' not in chemical_info):
                        # isolate the numeric string
                        chemical_str = chemical_info.replace(chemical, '').strip().replace('%', '').replace(' ', '')
                        chemical_strs.append(np.float64(chemical_str)/100)
                        found_flag = True
                        
                # if none appended, then add a 0
                if not found_flag:
                    chemical_strs.append(np.float64(0))

            df[f'{chemical}_frac'] = chemical_strs

        #cprint(df['K2O_frac'], 'blue') 
        #cprint(list(df.columns), 'blue')


        # Build catchment masks and add to correct dict
        Field_vals = df['Field'].values
        for catchment in catchments:
            pass
        
        df['weight_multipliers'] = build_weight_multipliers(df)

        for chemical in chemicals:
            df[f'{chemical}_weight'] = df['weight_multipliers'] * df[f'{chemical}_frac'] * df['Total_application_in_Units']
        df['MANURE_weight'] = df['weight_multipliers'] * df['MANURE'] * df['Total_application_in_Units']
        df['LIME_weight'] = df['weight_multipliers'] * df['LIME'] * df['Total_application_in_Units']
        
        # drop columns where total application is not specified - Add clculation of later with application rate and known area
        df = df[df['Total_application_in_Units'].notna()]

        # trim to specific columns
        df = df[['Datetime', 'Field', 'MANURE_weight', 'LIME_weight'] + [f'{chemical}_weight' for chemical in chemicals]]

        # for each possible catchment, create a mask and append
        for catchment in catchments:
            catchment_mask = np.zeros(len(df), dtype=bool)
            for field in catchment_field_dict[str(catchment)]:
                catchment_mask |= (df['Field'].values == field)

            # append to the by-catchment dict lists
            catchment_fertilisation_dict[str(catchment)].append(df[catchment_mask])

    # concatenate all dataframes for each catchment
    for catchment in catchments:
        catchment_fertilisation_dict[str(catchment)] = pd.concat(catchment_fertilisation_dict[str(catchment)], axis=0)

    #cprint(catchment_fertilisation_dict, 'green')

    return catchment_fertilisation_dict


def assess_data_quality(by_catchment_dict):

    long_names_dict = {value: key for key, value in conversion_dict.items()}

    # Make plots showcasing the data quality for each metric.
    metrics = list(conversion_dict.values())
    metrics_fullnames = list(conversion_dict.keys())

    shortened_metrics = metrics.copy()#['Ammonia', 'NitriteANDNitrate', 'pH']
    for metric in shortened_metrics:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        datetime_vals = by_catchment_dict[list(by_catchment_dict.keys())[0]]['Datetime']
        plot_time_start = np.min(datetime_vals)
        plot_time_end = np.max(datetime_vals)

        cprint(f'Metric: {metric}', 'cyan')
        catchment_xdict = {}
        acceptable_percentage_xdict = {}
        for catchment_pseudonum, catchment in enumerate(by_catchment_dict.keys()):
            ax.fill_between(x=[catchment_pseudonum-0.4, catchment_pseudonum+0.4], y1=plot_time_start, y2=plot_time_end, color='red', lw=0., zorder=0.)

            # Assess the data quality for every catchment
            try:
                #cprint(sorted(np.unique(by_catchment_dict[catchment][f'{metric} DQ'])), 'yellow')
                pass
            except Exception as e:
                # convert all values to str by force
                pass
                #by_catchment_dict[catchment][f'{metric} DQ'] = by_catchment_dict[catchment][f'{metric} DQ'].astype(str)
                #cprint(np.unique(by_catchment_dict[catchment][f'{metric} DQ']), 'red')

            vals = by_catchment_dict[catchment][f'{metric} DQ'].astype(str).values
            # for vals, generate a list of the start and end of each 'Acceptable' datatype block
            acceptable_mask = vals == 'Acceptable'
            # Return indexes
            first_in_block = np.where(np.concatenate(([acceptable_mask[0]], acceptable_mask[1:] & ~acceptable_mask[:-1])))[0]
            last_in_block = np.where(np.concatenate((acceptable_mask[:-1] & ~acceptable_mask[1:], [acceptable_mask[-1]])))[0]

            for start_idx, end_idx in zip(first_in_block, last_in_block):
                ax.fill_between(x=[catchment_pseudonum-0.4, catchment_pseudonum+0.4], y1=datetime_vals[start_idx], y2=datetime_vals[end_idx], color='green', lw=0., zorder=0.2)

            #cprint(first_in_block, 'cyan')
            #cprint(last_in_block, 'cyan')
            catchment_xdict[catchment_pseudonum] = f'{catchment}'
            acceptable_percentage_xdict[catchment_pseudonum] = f'{np.mean(acceptable_mask):.2%}'

        ax.set_xticks(list(catchment_xdict.keys()))
        ax.set_xticklabels(catchment_xdict.values())
        # hide minor x ticks
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(f'Catchment', loc='center')
        ax.set_ylabel(f'Date-Time', loc='center')
        
        
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())  # sync with bottom axis
        ax_top.set_xticks(list(acceptable_percentage_xdict.keys()))
        ax_top.set_xticklabels(acceptable_percentage_xdict.values())
        ax_top.tick_params(axis='x', pad=2)  # positive = further from axis, negative = closer

        ax_top.xaxis.set_minor_locator(NullLocator())
        #ax_top.set_xlabel("Catchment (top)", loc="center")
        
        ax.set_title(f'{long_names_dict[metric]} Data Quality', pad=50)
        
        save_folder = f'plots/data_quality/'
        os.makedirs(save_folder, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.savefig(f'{save_folder}/{metric}_data_quality.pdf')
        plt.close(fig)
    
    return