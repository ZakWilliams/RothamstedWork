# choose a given set of catchments
# open and clean the relevant data
# plot the evolution of the means and standard deviations over time, including when catchments transitioned from arable to livestock

# choose chemicalto compare. Also include option to compare all nitrogens together
from termcolor import cprint
import utils_means.utils as utils
import utils_means.plotting as plotting

start_year = 2011
end_year = 2024

catchments = [5, 1]

runoff_measure = 'Nitrogen'
months_range = [10, 2]

# Nitrogen, NitriteANDNitrate, Dissolved Oxygen, pH, Ammonium, Ammonia, Turbidity

# this should produce a dict containing the mean, std of a given quantity for each asked after catchment, for each year
results = utils.process_data(start_year, end_year, catchments, runoff_measure, months_range)

cprint(results, 'red')

plotting.plot_results(start_year, end_year, results, runoff_measure, months_range)