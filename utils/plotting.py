import pandas as pd
from termcolor import cprint
import matplotlib.pyplot as plt

from utils.bank import LaTeX_dict, units_dict

def plot(by_catchment_dict,
              by_catchment_fertilisation_dict,
              deposition_value,
              runoff_value,
              catchment,
              start_month,
              end_month,
              runoff_per='l'):
    start_year = int(start_month[3:])
    end_year = int(end_month[3:])
    years = list(range(start_year, end_year + 1))

    start_Datetime = pd.to_datetime(f'01/{start_month} 00:00:00')
    end_Datetime = pd.to_datetime(f'01/{end_month} 23:59:59')

    cprint(start_Datetime, 'green')
    cprint(end_Datetime, 'yellow')

    runoff_Datetimes = by_catchment_dict[str(catchment)][f'Datetime']
    runoff_vals = by_catchment_dict[str(catchment)][runoff_value]

    event_Datetimes = by_catchment_fertilisation_dict[str(catchment)][f'Datetime']
    event_vals = by_catchment_fertilisation_dict[str(catchment)][f'{deposition_value}_weight']

    zero_deposit = event_vals == 0
    event_Datetimes = event_Datetimes[~zero_deposit]
    event_vals = event_vals[~zero_deposit]
    
    # if runoff is 'l', no need to change anything. if 's' then multiply the current runoff by the total flow values
    if runoff_per == 's':
        total_flow = by_catchment_dict[str(catchment)]['Flow'].sum()
        runoff_vals = runoff_vals * total_flow
        replace_str = '/s'
    else:
        replace_str = '/l'

    # if len years = 1, round to spec months
    if len(years) == 1:
        runoff_Datetimes = runoff_Datetimes[(runoff_Datetimes >= start_Datetime) & (runoff_Datetimes <= end_Datetime)]
        runoff_vals = runoff_vals[runoff_Datetimes.index]
        event_Datetimes = event_Datetimes[(event_Datetimes >= start_Datetime) & (event_Datetimes <= end_Datetime)]
        event_vals = event_vals[event_Datetimes.index]
    else:
        cprint(f'>1 year provided, ignoring spec months', 'yellow')
        
        

    fig = plt.figure(figsize=[12, 10])
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax_events = ax.twinx()

    ax.plot(runoff_Datetimes, runoff_vals, color='blue')
    ax_events.scatter(event_Datetimes, event_vals, color='orange', marker='D')

    ax.set_xlabel('Date-Time', loc='center')
    ax.set_ylabel(f'{LaTeX_dict[runoff_value]} Runoff [{units_dict[runoff_value].replace("/l",replace_str)}]', loc='center')
    ax_events.set_ylabel(f'{LaTeX_dict[deposition_value]} Deposited [{units_dict[deposition_value]}]', loc='center')
    ax.set_title(f'Catchment {catchment} - {years}')

    fig.savefig(f'{deposition_value}_v_{runoff_value}_catchment{catchment}_{start_month}-{end_month}.pdf')
    plt.close(fig)


    return