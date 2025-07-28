
import numpy as np
import time
import os,sys
import h5py
import pandas as pd
sys.path.append(os.path.join(os.getcwd(),"scripts"))
sys.path.append(os.path.join(os.getcwd(),"inputs"))
sys.path.append(os.getcwd())
import random
from settings import run_args
from scripts.settings import state
state = state.generate_state(run_args.args.configs_dir)

hdf5_file_path = 'outputs/daysim/daysim_outputs.h5'

def airpot_trips_to_emp_parking_lot(hdf_filename):

    start_time = time.time()
    #Create the HDF5 Container if needed and open it in read/write mode using "r+"
    my_store=h5py.File(hdf_filename, "r+")

    #Stores in the HDF5 Container to read or write to
    daysim_set = my_store['Trip']

    #Store arrays from Daysim/Trips Group into numpy arrays, indexed by TOD.
    #This means that only trip info for the current Time Period will be included in each array.
    tourid = np.asarray(daysim_set["tour_id"])
    tourid = tourid.astype('int')

    otaz = np.asarray(daysim_set["otaz"])
    otaz = otaz.astype('int')
    # otaz = otaz[tod_index]
    
    dtaz = np.asarray(daysim_set["dtaz"])
    dtaz = dtaz.astype('int')
    # dtaz = dtaz[tod_index]

    mode = np.asarray(daysim_set["mode"])
    mode = mode.astype("int")
    # mode = mode[tod_index]

    dpurp = np.asarray(daysim_set["dpurp"])
    dpurp = dpurp.astype("int")

    # Create a trip dataframe
    trip_df = pd.DataFrame({'tourid':tourid, 'otaz':otaz, 'dtaz':dtaz, 'tmode':mode, 'dpurp':dpurp})
    # Check if airport parking lot is already in daysim output
    trip_df[trip_df.dtaz==state.emme_settings.AIRPORT_EMP_PARKING_LOT].head()

    auto_mode = [3,4,5]
    work_purpose = [1]
    # Gather the airport work trips
    airport_work_trips_index = trip_df[(trip_df.dtaz==1) & (trip_df.tmode.isin(auto_mode)) & (trip_df.dpurp.isin(work_purpose))].index
    airport_tourids = trip_df.loc[airport_work_trips_index].tourid
    # Identify valid return trips
    airport_back_trips_index = trip_df[(trip_df.otaz==1) & (trip_df.tmode.isin(auto_mode)) & (trip_df.tourid.isin(airport_tourids))].index
    return_tourids = trip_df.loc[airport_back_trips_index].tourid
    # Sample trips that will use the employee parking lot
    num_sample = round(len(return_tourids)*state.emme_settings.PARKING_ZONE_SAMPLE_RATE)
    random.seed(state.emme_settings.PARKING_SAMPLE_SEED)
    update_tourids = random.sample(return_tourids.to_list(), num_sample) 
    # Update the origin and destination TAZ to employeed parking lot for identified trips
    trip_df.loc[(trip_df.tourid.isin(update_tourids)) & (trip_df.dtaz==1) & (trip_df.tmode.isin(auto_mode)) & (trip_df.dpurp.isin(work_purpose)),'dtaz'] = int(state.emme_settings.AIRPORT_EMP_PARKING_LOT)
    trip_df.loc[(trip_df.tourid.isin(update_tourids)) & (trip_df.otaz==1) & (trip_df.tmode.isin(auto_mode)),'otaz'] = int(state.emme_settings.AIRPORT_EMP_PARKING_LOT)

    # Update the daysim output to save the updated origin and destination taz
    del my_store["Trip"]["otaz"]
    my_store["Trip"].create_dataset('otaz', data=trip_df.otaz.astype('uint16'),compression='gzip')
    del my_store["Trip"]["dtaz"]
    my_store["Trip"].create_dataset('dtaz', data=trip_df.dtaz.astype('uint16'),compression='gzip')
    my_store.close()

    # Generate a file that Airport model uses to identify work trips
    work_trips_df = pd.DataFrame({'Name':['General parking at terminal', 'Airport employee off-site parking'],
                                  'TAZ':[state.emme_settings.SEATAC, state.emme_settings.AIRPORT_EMP_PARKING_LOT],
                                  'Employee Stalls': [len(return_tourids)-num_sample, num_sample],
                                  'Share to Terminal': [1, 1],
                                  'Public Transit Share to Terminal': [0, 1]})
    
    work_trips_df.to_csv(os.path.join('scripts', 'airport', 'configs', 'airport_employee_park.csv'), index=False)

    end_time = time.time()

    #print('It took', round((end_time-start_time)/60,2), ' minutes to update airport work trips to employee parking lot.')
    #text = 'It took ' + str(round((end_time-start_time)/60,2)) + ' minutes to update airport work trips to employee parking lot.'
    #logger.debug(text)

def main():
    #Report model starting
    #current_time = str(time.strftime("%H:%M:%S"))
    #logger.debug('----Began DaySim Update Airport Trips to Parking Lot script at ' + current_time)
    airpot_trips_to_emp_parking_lot(hdf5_file_path)

if __name__ == "__main__":
    main()
