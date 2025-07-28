import pandas as pd
import os, sys
import re
import json

sys.path.append(os.path.join(os.getcwd(), "inputs"))
sys.path.append(os.path.join(os.getcwd(), "scripts"))
sys.path.append(os.getcwd())
from scripts.emme_project import *

def adjust_ttf(state):
    # my_project = EmmeProject(state.network_settings.network_summary_project, state)
    my_project = state.main_project
    
    for key, value in state.network_settings.sound_cast_net_dict.items():
        my_project.change_active_database(key)
        
        if my_project.tod in state.network_settings.transit_tod_list:
            network = my_project.current_scenario.get_network()
            for transit_segment in network.transit_segments():
                if int(transit_segment.line.id) in state.network_settings.emp_shuttle_lines:
                    if transit_segment.transit_time_func==5:
                        transit_segment.transit_time_func=4

            my_project.current_scenario.publish_network(network, resolve_attributes=True)        
        my_project.bank.dispose()