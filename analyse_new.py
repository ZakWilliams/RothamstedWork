# open precipitiation files
# assure data quality 
# apply a model learner blah

import utils.process as process
import utils.plotting as plotting
import mplhep
mplhep.style.use('LHCb2')
import time
start_time = time.time()
import pandas as pd
from termcolor import cprint

#years = [2017, 2018]
start_month = '02/2017'
end_month = '07/2018'

catchments = [1, 2, 3, 10, 15]
# build the catchment split dict
by_catchment_dict = process.build_catchment_dict(catchments=catchments, start_month=start_month, end_month=end_month)

by_catchment_fertilisation_dict = process.build_catchment_fertilisation_dict(catchments=catchments, start_month=start_month, end_month=end_month)

# print/plot assesments of data quality for each metric, in each catchment, in each year
process.assess_data_quality(by_catchment_dict)

# make a plot of deposition of chemical X, and runoff of chemical Y
# deposition chemicals - 'N', 'P2O5', 'K2O', 'SO3', 'MANURE', 'LIME
# runoff chemicals - 'NitriteANDNitrate', 'Ammonia', 'Ammonium', 'Conductivity', 'Dissolved Oxygen', 'pH', 'Turbidity (FNU)', 'Fluorescent Dissolved Organic Matter (ug/l QSU)',
plotting.plot(by_catchment_dict,
              by_catchment_fertilisation_dict,
              deposition_value='N',
              runoff_value='NitriteANDNitrate',
              catchment=3,
              start_month=start_month,
              end_month=end_month,
              runoff_per='l')
cprint(f'Finished execution in {time.time()-start_time:.2f}s!', 'green')