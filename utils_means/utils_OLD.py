import os
import numpy as np
import pandas as pd
from termcolor import cprint
from utils_means.bank import conversion_dict


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
        csv_names = os.listdir(f'store2/MSMWFD_{year}/')
        csv_name = [csv_ for csv_ in csv_names if csv_.startswith('measurement')][0]


        df = safe_read_csv(f'store2/MSMWFD_{year}/{csv_name}')
        # Process the DataFrame to create a compact representation split by catchment, rather than by year
        for catchment in catchments:
            catchment_columns = [col for col in df.columns if (f'[Catchment {catchment}]' in col) or (f'[Catchment {catchment} ' in col)]
            keep_catchment_columns = [col for col in catchment_columns if np.any([col.startswith(key) for key in conversion_dict.keys()])]
            catchment_df = df[['Datetime'] + keep_catchment_columns].copy()
            catchment_df['Datetime'] = pd.to_datetime(catchment_df['Datetime'], format='%Y/%m/%d %H:%M:%S')

            # build the column renamer using conversion_dict
            renaming_dict = {}
            for key, value in conversion_dict.items():
                if catchment != 4:
                    renaming_dict[f'{key} [Catchment {catchment}]'] = f'{value}'
                    renaming_dict[f'{key} [Catchment {catchment}] Quality'] = f'{value} DQ'
                    renaming_dict[f'{key} [Catchment {catchment}] Quality Last Modified'] = f'{value} DQLM'
                    catchment_df.rename(columns=renaming_dict, inplace=True)

                else:
                    if year < 2013:
                        renaming_dict[f'{key} [Catchment {catchment} Prior to  2013/08/13]'] = f'{value}'
                        renaming_dict[f'{key} [Catchment {catchment} Prior to  2013/08/13] Quality'] = f'{value} DQ'
                        renaming_dict[f'{key} [Catchment {catchment} Prior to  2013/08/13] Quality Last Modified'] = f'{value} DQLM'
                        catchment_df.rename(columns=renaming_dict, inplace=True)
                        
                    elif year > 2013:
                        renaming_dict[f'{key} [Catchment {catchment} After  2013/08/13]'] = f'{value}'
                        renaming_dict[f'{key} [Catchment {catchment} After  2013/08/13] Quality'] = f'{value} DQ'
                        renaming_dict[f'{key} [Catchment {catchment} After  2013/08/13] Quality Last Modified'] = f'{value} DQLM'
                        catchment_df.rename(columns=renaming_dict, inplace=True)
                        
                    else:
                        # create datetime object for 2013-08-13 00::00::00
                        boundary_value = pd.Timestamp('2013-08-13 00:00:00')
                        pre_date_mask = catchment_df['Datetime'] < boundary_value
                        catchment_df[f'{value}'] = np.where(pre_date_mask,
                                                            catchment_df[f'{key} [Catchment {catchment} Prior to  2013/08/13]'],
                                                            catchment_df[f'{key} [Catchment {catchment} After  2013/08/13] Quality'])
                        catchment_df[f'{value} DQ'] = np.where(pre_date_mask,
                                                                catchment_df[f'{key} [Catchment {catchment} Prior to  2013/08/13] Quality'],
                                                                catchment_df[f'{key} [Catchment {catchment} After  2013/08/13] Quality'])
                        catchment_df[f'{value} DQLM'] = np.where(pre_date_mask,
                                                                  catchment_df[f'{key} [Catchment {catchment} Prior to  2013/08/13] Quality Last Modified'],
                                                                  catchment_df[f'{key} [Catchment {catchment} After  2013/08/13] Quality Last Modified'])

                        # drop the replaced columns
                        drop_columns = [
                            f'{key} [Catchment {catchment} Prior to  2013/08/13]',
                            f'{key} [Catchment {catchment} After  2013/08/13]',
                            f'{key} [Catchment {catchment} Prior to  2013/08/13] Quality',
                            f'{key} [Catchment {catchment} After  2013/08/13] Quality',
                            f'{key} [Catchment {catchment} Prior to  2013/08/13] Quality Last Modified',
                            f'{key} [Catchment {catchment} After  2013/08/13] Quality Last Modified'
                        ]
                        catchment_df.drop(columns=drop_columns, inplace=True)

                    # special renaming and combining due to catchement 4 name specs
                    # if year is > 2013 or < 2013, then do blah, else apply mask
                    # could just apply mask, actually




                catchment_dict[str(catchment)].append(catchment_df)

            #cprint(f'{list(catchment_df.columns)}\n', 'red')

    cprint(f'Concatenating years...', 'cyan')
    for catchment in catchments:
        catchment_dict[str(catchment)] = pd.concat(catchment_dict[str(catchment)], ignore_index=True)

    # Convert datetime strings to datetime objects
    #cprint(f'Converting datetime strings to datetime objects...', 'cyan')
    #for catchment in catchments:
    #    catchment_dict[str(catchment)]['Datetime'] = pd.to_datetime(catchment_dict[str(catchment)]['Datetime'], format='%Y/%m/%d %H:%M:%S')

    cprint(f'Time-united catchment-split dict created.', 'green')

    return catchment_dict

def combine_DQs(chem_DQ1, chem_DQ2, chem_DQ3):
    # if both acceptable, use acceptable, if not then attach bespoke, 'Unacceptable'
    combined_DQ = pd.Series(index=chem_DQ1.index, dtype=str)
    all_acceptable_mask = (chem_DQ1 == 'Acceptable') & (chem_DQ2 == 'Acceptable') & (chem_DQ3 == 'Acceptable')
    combined_DQ[all_acceptable_mask] = 'Acceptable'
    combined_DQ.fillna('Unacceptable', inplace=True)
    return combined_DQ

def process_data(start_year, end_year, catchments, runoff_chemical, months_range):

    if months_range is None:
        months_range = [1, 12]

    # build start and end month objects
    start_month = f'01/{start_year}'
    end_month = f'12/{end_year}'
    # build the split-by-catchment concatenated-by-year dict
    catchment_dict = build_catchment_dict(catchments, start_month, end_month)

    if months_range is None:
        list_valid_months = list(range(1, 13))
    else:
        list_valid_months = []
        if months_range[0] < months_range[1]:
            list_valid_months = list(range(months_range[0], months_range[1] + 1))
        elif months_range[1] < months_range[0]:
            list_valid_months = list(range(months_range[1], 12 + 1)) + list(range(1, months_range[0] + 1))
        elif months_range[0] == months_range[1]:
            list_valid_months = [months_range[0]]

    # make a mask for each month in months range
    for catchment in catchment_dict.keys():
        df = catchment_dict[catchment]
        Datetime_month = df['Datetime'].dt.month
        catchment_dict[catchment] = catchment_dict[catchment][Datetime_month.isin(list_valid_months)]
        

    years = [year for year in range(start_year, end_year + 1)]
    means_dict = {}
    stds_dict = {}
    std_upper_dict = {}
    std_lower_dict = {}

    for catchment in catchment_dict.keys():
        means_dict[catchment] = []
        stds_dict[catchment] = []
        std_upper_dict[catchment] = []
        std_lower_dict[catchment] = []
        datetime_vals = catchment_dict[catchment]['Datetime']
        #cprint(list(catchment_dict[catchment].keys()), 'red')
        cprint(list(catchment_dict[catchment].keys()), 'blue')
        if runoff_chemical != 'Nitrogen':
            chem_vals = catchment_dict[catchment][runoff_chemical]
            chem_DQ = catchment_dict[catchment][f'{runoff_chemical} DQ']
            chem_DQ = chem_DQ.astype(str)
        else:
            chem_vals = catchment_dict[catchment]['NitriteANDNitrate'] + catchment_dict[catchment]['Ammonium'] + catchment_dict[catchment]['Ammonia']
            chem_DQ = combine_DQs(catchment_dict[catchment][f'NitriteANDNitrate DQ'], catchment_dict[catchment][f'Ammonium DQ'], catchment_dict[catchment][f'Ammonia DQ'])
            chem_DQ = chem_DQ.astype(str)

        cprint(catchment, 'white', 'on_cyan')
        
        
        # edit this bit below which collects the values properly when the valid months is year bridging
        


        if months_range[1] >= months_range[0]:
            years_loop = [year for year in years if year in datetime_vals.dt.year.values]
        else:
            # take off last year
            years_loop = [year for year in years if year in datetime_vals.dt.year.values][:-1]
            
            
            
        for year in years_loop:
            if months_range[1] >= months_range[0]:
                time_mask = (datetime_vals.dt.year == year)
            else:
                # current year, along with 
                time_mask_PT1 = (datetime_vals.dt.year == year) & (datetime_vals.dt.month >= months_range[0])
                time_mask_PT2 = (datetime_vals.dt.year == year+1) & (datetime_vals.dt.month <= months_range[1])
                time_mask = time_mask_PT1 | time_mask_PT2

            #year_mask = (datetime_vals.dt.year == year)
            good_quality_mask = (chem_DQ == 'Acceptable')
            pass_chem_mask = (chem_vals.astype(str) != 'nan')
            # make a cast_float mask, which is true when pass_chem_mask can be successfully cast into a float, and false where not
            numeric_chem_vals = pd.to_numeric(chem_vals, errors="coerce")
            cast_float_mask = pass_chem_mask & numeric_chem_vals.notna()

            #pass_chem_mask2 = (chem_vals.astype(str) != 'Acceptable')
            # cast all chem_DQs into strings
            pass_chem_vals = chem_vals[time_mask & good_quality_mask & pass_chem_mask & cast_float_mask]

            # clip extreme mask - a crude way of eliminating the most extreme, likely erroneous values that are causing very large stds
            clip_mask = pass_chem_vals <= pass_chem_vals.quantile(1.0)
            cprint(year, 'cyan')

            if len(pass_chem_vals) > 0:
                #means_dict[catchment].append(np.mean(pass_chem_vals))
                means_dict[catchment].append(np.percentile(pass_chem_vals, 50))
                stds_dict[catchment].append(np.std(pass_chem_vals[clip_mask]))
                std_upper_dict[catchment].append(np.percentile(pass_chem_vals, (68.27+50)/2)-np.percentile(pass_chem_vals, 50))
                std_lower_dict[catchment].append(np.percentile(pass_chem_vals, 50)-np.percentile(pass_chem_vals, (100-68.27)/2))
            else:
                means_dict[catchment].append(np.nan)
                stds_dict[catchment].append(np.nan)
                std_upper_dict[catchment].append(np.nan)
                std_lower_dict[catchment].append(np.nan)

            #means_dict[catchment].append(chem_vals[year_mask].mean())
            #stds_dict[catchment].append(chem_vals[year_mask].std())

    #cprint(means_dict, 'red')
    #cprint(stds_dict, 'yellow')
    #cprint(std_upper_dict, 'blue')
    #cprint(std_lower_dict, 'green')

    # isolate the desired chemical
    # initially, build a catchment dict which retains the valid by-15-minute recordings of the runoffs for each catchment
    # then, compress these into once-yearly numbers, with the mean and standard deviations.



    return {'means' : means_dict, 'stds' : stds_dict, 'std_upper' : std_upper_dict, 'std_lower' : std_lower_dict}