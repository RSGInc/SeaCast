import os

##############################
# Input paths and model years
##############################
model_year = '2018'
base_year = '2018'
landuse_inputs = 'seacast_2018'
network_inputs = 'seacast_2018'
soundcast_inputs_dir = 'E:/projects/clients/SeaTac/GitHub/SeaCastScenarioInputs'

##############################
# Initial Setup
##############################
run_accessibility_calcs = True
run_setup_emme_project_folders = True
run_setup_emme_bank_folders = True
run_copy_scenario_inputs = True
run_import_networks = True

##############################
# Model Procedures
##############################
run_skims_and_paths_free_flow = True
run_skims_and_paths = True
run_truck_model = True
run_airport_model = True
run_supplemental_trips = True
run_daysim = True
run_daysim_popsampler = True
run_summaries = True

##############################
# Modes and Path Types
##############################
include_av = False
include_tnc = True
tnc_av = False    # TNCs (if available) are AVs
include_tnc_to_transit = False # AV to transit path type allowed
include_knr_to_transit = False # Kiss and Ride to Transit
include_delivery = False
include_telecommute = False

##############################
# Pricing
##############################
add_distance_pricing = False
distance_rate_dict = {'md': 8.5, 'ev': 8.5, 'am': 13.5, 'ni': 8.5, 'pm': 13.5}

##############################
# Household Sampling Controls
##############################
households_persons_file = r'inputs\scenario\landuse\hh_and_persons.h5'
# Popsampler - super/sub-sampling in population synthesis
sampling_option = 2 #1-3: five options available - each option is a column in pop_sample_district below
pop_sample_district = {'City of SeaTac':[1,4,2],
					'Rest of King County':[1,1,0.75], 
					'Rest':[1,1,0.75], 
					} #population sampling by districts - 3 options to choose from (each option is a column) - base case and two preferred sampling plans
zone_district_file = r'inputs\model\lookup\hh_sampling_region_taz.csv' #input to generate taz_sample_rate_file below
taz_sample_rate_file = r'inputs\model\lookup\taz_sample_rate.txt' #intermediate output, input to popsampler script


##############################
# Other Controls
##############################
run_integrated = False
should_build_shadow_price = False
delete_banks = False