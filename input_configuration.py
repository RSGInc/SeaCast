# This file contains model input parameters imported by SoundCast scripts.   

# CONFIGURATION TO RUN SOUNDCAST
# Note there are many other configuration files for specific model steps in their respective directories, such as Daysim, or skimming.

# Scenario and input paths
base_year = '2014'  #. This should always be 2010 unless the base year changes
scenario_name = '2014'
daysim_code = 'R:/SoundCast/daysim_2016' 
master_project = 'LoadTripTables'
main_inputs_folder =  'R:/SoundCast/Inputs/'
base_inputs = main_inputs_folder + scenario_name


# Script and subprocess controls
 
##### Only update parking for future-year analysis!
run_update_parking = False
run_convert_hhinc_2000_2010 = False
run_accessibility_calcs = True
run_copy_daysim_code = True
run_setup_emme_project_folders = True
run_setup_emme_bank_folders = True
run_copy_large_inputs = True
run_import_networks = True
# if run copy seed skims is tru (intentional typo for find and replace), you don't need to run skims and paths seed trips
# the model run will start with daysim
run_copy_seed_skims = True
create_no_toll_network = False
run_skims_and_paths_seed_trips = True
##### Shadow prices now copied and are always used. Only Run this if building shadow prices from scratch!
should_build_shadow_price = False
run_skims_and_paths = True
run_truck_model = True
run_supplemental_trips = True
run_daysim = True
run_accessibility_summary = True
run_network_summary = True
run_soundcast_summary = True
run_create_daily_bank = True
run_ben_cost = False
run_bike_model = True

# Model iterations, population sampling, log files, etc.
pop_sample = [5, 2, 2]
# Assignment Iterations:
max_iterations_list = [10, 100, 100]
min_pop_sample_convergence_test = 10
# start building shadow prices - only run work locations
shadow_work = [2,1]
shadow_con = 30 #%RMSE for shadow pricing to consider being converged
parcel_decay_file = 'inputs/buffered_parcels.txt' #File with parcel data to be compared to
# run daysim and assignment in feedback until convergence

main_log_file = 'soundcast_log.txt'
network_summary_files = ['6to7_transit', '7to8_transit', '8to9_transit', '9to10_transit',
                         'counts_output', 'network_summary']
#This is what you get if the model runs cleanly, but it's random:
good_thing = ["cookie", "run", "puppy", "seal sighting",  "beer", "snack", "nap","venti cinnamon dolce latte"]

# These files are often missing from a run.  We want to check they are present and warn if not.
# Please add to this list as you find files that are missing.
commonly_missing_files = ['buffered_parcels.txt', 'tazdata.in']

# Specific reports to run
run_daysim_report = True
run_day_pattern_report = True
run_mode_choice_report = True
run_dest_choice_report = True
run_long_term_report = True
run_time_choice_report = True
run_district_summary_report = False
run_tableau_db = False
run_landuse_summary = True

# Calibration Summary Configuration
h5_results_file = 'outputs/daysim_outputs.h5'
h5_results_name = 'DaysimOutputs'
h5_comparison_file = 'scripts/summarize/inputs/calibration/survey.h5'
h5_comparison_name = 'Survey'
guidefile = 'scripts/summarize/inputs/calibration/CatVarDict.xlsx'
districtfile = 'scripts/summarize/inputs/calibration/TAZ_TAD_County.csv'
report_output_location = 'outputs'
bc_outputs_file = 'outputs/BenefitCost.xlsx'