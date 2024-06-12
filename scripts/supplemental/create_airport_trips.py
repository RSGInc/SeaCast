import pandas as pd
import numpy as np
import os,sys
import h5py
from sqlalchemy import create_engine
sys.path.append(os.path.join(os.getcwd(),"scripts"))
sys.path.append(os.getcwd())
from emme_configuration import *

def split_tod_internal(total_trips_by_mode, tod_factors_df):
    """Split trips into time of a day: apply time of the day factors to internal trips"""

    matrix_dict = {}
    #for tod, dict in test.iteritems():
    for tod in tod_factors_df.time_of_day.unique():
        #open work externals:
        tod_dict = {}
        tod_df = tod_factors_df[tod_factors_df['time_of_day'] == tod]

        ixxi_work_store = h5py.File('outputs/supplemental/external_work_' + tod + '.h5', 'r')

        # Time of day distributions to use foreach mode
        tod_dict = {'sov': 'sov',
                    'hov2': 'hov',
                    'hov3': 'hov',
                    'walk': 'sov',
                    'bike': 'sov',
                    'trnst': 'trnst',
                    'litrat': 'litrat',
                    'commuter_rail': 'commuter_rail',
                    'ferry': 'ferry',
                    'passenger_ferry': 'passenger_ferry'}

        for mode, tod_type in tod_dict.items():
            if mode in ['sov','hov2','hov3']:
                tod_dict[mode] = np.array(total_trips_by_mode[mode])*tod_df[tod_df['mode'] == tod_type]['value'].values[0] + np.array(ixxi_work_store[mode])
            else:
                tod_dict[mode] = np.array(total_trips_by_mode[mode]) * tod_df[tod_df['mode'] == 'transit']['value'].values[0] 

        ixxi_work_store.close()
        matrix_dict[tod] = tod_dict

    return matrix_dict

# Output trips
def output_trips(path, matrix_dict):
    for tod in matrix_dict.keys():
        print("Exporting supplemental trips for time period: " + str(tod))
        my_store = h5py.File(path + str(tod) + '.h5', "w")
        for mode, value in matrix_dict[tod].items():
            if mode in ['sov','hov2','hov3']:
                mode = mode + '_inc2'
            my_store.create_dataset(str(mode), data=value, compression='gzip')
        my_store.close()

def main():

    output_dir = r'outputs/supplemental/'

    conn = create_engine('sqlite:///inputs/db/soundcast_inputs.db')
    tod_factors_df = pd.read_sql('SELECT * FROM time_of_day_factors', con=conn)

    # Add airport trips to external trip table for auto modes:
    ixxi_non_work_store = h5py.File('outputs/supplemental/external_non_work.h5', 'r')
    external_modes = ['sov', 'hov2', 'hov3']
    ext_trip_table_dict = {}
    # get non-work external trips
    for mode in external_modes:
        ext_trip_table_dict[mode] = np.nan_to_num(np.array(ixxi_non_work_store[mode]))
    
    ixxi_non_work_store.close()

    non_auto_modes = ['walk', 'bike', 'trnst', 'litrat', 'commuter_rail', 'ferry', 'passenger_ferry']
    matrix_shape = ext_trip_table_dict[external_modes[0]].shape
    for mode in non_auto_modes:
        ext_trip_table_dict[mode] = np.zeros(matrix_shape, np.float32)

    # Add external non-work trips to airport trips
    # Export as income class 
    total_trips_by_mode = ext_trip_table_dict.copy()

    # Apply time of day factors
    supplemental_matrix_dict = split_tod_internal(total_trips_by_mode, tod_factors_df)

    # Output final trip tables, by time of the day and trip mode. 
    output_trips(output_dir, supplemental_matrix_dict)


if __name__ == "__main__":
    main()