# choose a given set of catchments
# open and clean the relevant data
# plot the evolution of the means and standard deviations over time, including when catchments transitioned from arable to livestock

# choose chemicalto compare. Also include option to compare all nitrogens together
from termcolor import cprint
import utils_means.utils as utils
import utils_means.plotting as plotting

start_year = 2011
end_year = 2024

catchments = [5, 7]

runoff_measure = 'Turbidity'
months_range = [10, 2]
perc_limit = [1, 99]

# Nitrogen, NitriteANDNitrate, Dissolved Oxygen, pH, Ammonium, Ammonia, Turbidity

# this should produce a dict containing the mean, std of a given quantity for each asked after catchment, for each year
results, quality_assessment = utils.process_data(start_year, end_year, catchments, runoff_measure, months_range, perc_limit)

cprint(f'Plotting...', 'cyan')
plotting.plot_results(start_year, end_year, results, runoff_measure, months_range, quality_assessment)