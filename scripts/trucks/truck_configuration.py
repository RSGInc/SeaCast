#################################### TRUCK MODEL ####################################
truck_model_project = 'Projects/TruckModel/TruckModel.emp'
districts_file = 'truck_districts.ens'
truck_base_net_name = 'am_roadway.in'

#TOD to create Bi-Dir skims (AM/EV Peak)
truck_generalized_cost_tod = {'7to8' : 'am', '17to18' : 'pm'}
#GC & Distance skims that get read in from Soundcast

# 4k time of day
tod_list = ['am','md', 'pm', 'ev', 'ni']
# External Magic Numbers
LOW_STATION = 3733
HIGH_STATION = 3750
EXTERNAL_DISTRICT = 'ga20'

truck_adjustment_factor = {'ltpro': 0.544,
							'mtpro': 0.545,
							'htpro': 0.530,
							'ltatt': 0.749,
							'mtatt': 0.75,
							'htatt': 1.0}
