# Validation for observed data

import os, sys, shutil
sys.path.append(os.path.join(os.getcwd(),"inputs"))
sys.path.append(os.path.join(os.getcwd(),"scripts"))
sys.path.append(os.getcwd())
import array as _array
import json
import numpy as np
import time
import os,sys
import h5py
import shutil
import multiprocessing as mp
import subprocess
from multiprocessing import Pool
import logging
import datetime
import argparse
import traceback
import pandas as pd
from shutil import copy2 as shcopy
from sqlalchemy import create_engine
from input_configuration import base_year
from emme_configuration import sound_cast_net_dict, MIN_EXTERNAL, MAX_EXTERNAL 

# output directory
measure_output_dir = 'outputs/performance_measures'

# Create a clean output directory
if os.path.exists(measure_output_dir):
    shutil.rmtree(measure_output_dir)
os.makedirs(measure_output_dir)

### FIXME: move to a config file
agency_lookup = {
    1: 'King County Metro',
    2: 'Pierce Transit',
    3: 'Community Transit',
    4: 'Kitsap Transit',
    5: 'Washington Ferries',
    6: 'Sound Transit',
    7: 'Everett Transit'
}
# List of route IDs to separate for analysis
special_route_list = [6998,6999,1997,1998,6995,6996,1973,1975,
                        4200,4201,4202,1671,1672,1673,1674,1675,1676,1040,1007,6550,
                        5001,5002,5003,5004,5005,5006,5007]

facility_type_lookup = {
    1:'Freeway',   # Interstate
    2:'Freeway',   # Ohter Freeway
    3:'Freeway', # Expressway
    4:'Ramp',
    5:'Arterial',    # Principal arterial
    6:'Arterial',    # Minor Arterial
    7:'Collector',    # Major Collector
    8:'Collector',    # Minor Collector
    9:'Collector',   # Local
    10:'Busway',
    11:'Non-Motor',
    12:'Light Rail',
    13:'Commuter Rail',
    15:'Ferry',
    16:'Passenger Only Ferry',
    17:'Connector',    # centroid connector
    18:'Connector',    # facility connector
    19:'HOV',    # HOV Only Freeway
    20:'HOV'    # HOV Flag
    }
	
county_lookup = {
    33: 'King',
    35: 'Kitsap',
    53: 'Pierce',
    61: 'Snohomish'
    }

tod_lookup = {  0:'20to5',
                1:'20to5',
                2:'20to5',
                3:'20to5',
                4:'20to5',
                5:'5to6',
                6:'6to7',
                7:'7to8',
                8:'8to9',
                9:'9to10',
                10:'10to14',
                11:'10to14',
                12:'10to14',
                13:'10to14',
                14:'14to15',
                15:'15to16',
                16:'16to17',
                17:'17to18',
                18:'18to20',
                19:'18to20',
                20:'20to5',
                21:'20to5',
                22:'20to5',
                23:'20to5'}


def reindex(series1, series2):
    """
    This reindexes the first series by the second series.  This is an extremely
    common operation that does not appear to  be in Pandas at this time.
    If anyone knows of an easier way to do this in Pandas, please inform the
    UrbanSim developers.
    The canonical example would be a parcel series which has an index which is
    parcel_ids and a value which you want to fetch, let's say it's land_area.
    Another dataset, let's say of buildings has a series which indicate the
    parcel_ids that the buildings are located on, but which does not have
    land_area.  If you pass parcels.land_area as the first series and
    buildings.parcel_id as the second series, this function returns a series
    which is indexed by buildings and has land_area as values and can be
    added to the buildings dataset.
    In short, this is a join on to a different table using a foreign key
    stored in the current table, but with only one attribute rather than
    for a full dataset.
    This is very similar to the pandas "loc" function or "reindex" function,
    but neither of those functions return the series indexed on the current
    table.  In both of those cases, the series would be indexed on the foreign
    table and would require a second step to change the index.
    Parameters
    ----------
    series1, series2 : pandas.Series
    Returns
    -------
    reindexed : pandas.Series
    """
    # turns out the merge is much faster than the .loc below
    df = pd.merge(series2.to_frame(name='left'),
                  series1.to_frame(name='right'),
                  left_on="left",
                  right_index=True,
                  how="left")
    return df.right

def calc_resident_measures(trip_df, zone_label = 'SeaTacResident'):
    performance_measures = {}
    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df[zone_label] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork']
    performance_measures['auto_trip_length_work_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df[zone_label] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork']
    performance_measures['auto_trip_time_work_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df[zone_label] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork']
    performance_measures['transit_trip_length_work_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df[zone_label] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork']
    performance_measures['transit_trip_time_work_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents that are transit dependent
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df[zone_label] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] * trip_df['TransitDependent']
    performance_measures['auto_trip_length_work_transit_dependent_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df[zone_label] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['TransitDependent']
    performance_measures['auto_trip_time_work_transit_dependent_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df[zone_label] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['TransitDependent']
    performance_measures['transit_trip_length_work_transit_dependent_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df[zone_label] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['TransitDependent']
    performance_measures['transit_trip_time_work_transit_dependent_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents that are low income
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df[zone_label] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] * trip_df['LowIncome']
    performance_measures['auto_trip_length_work_lowinc_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df[zone_label] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['LowIncome']
    performance_measures['auto_trip_time_work_lowinc_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df[zone_label] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['LowIncome']
    performance_measures['transit_trip_length_work_lowinc_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df[zone_label] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['LowIncome']
    performance_measures['transit_trip_time_work_lowinc_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents that are minority
    # # Auto Trip
    # trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df[zone_label] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] * trip_df['Minority']
    # performance_measures['auto_trip_length_work_minority_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    # trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df[zone_label] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['Minority']
    # performance_measures['auto_trip_time_work_minority_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # # TransitTrip
    # trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df[zone_label] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['Minority']
    # performance_measures['transit_trip_length_work_minority_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    # trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df[zone_label] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['Minority']
    # performance_measures['transit_trip_time_work_minority_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac employees
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['SeaTacEmployee'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] * trip_df[zone_label]
    performance_measures['auto_trip_length_work_employees'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['SeaTacEmployee'] * trip_df['travtime'] * trip_df['trexpfac'] * trip_df['PurposeWork'] * trip_df[zone_label]
    performance_measures['auto_trip_time_work_employees'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['SeaTacEmployee'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork']  * trip_df[zone_label]
    performance_measures['transit_trip_length_work_employees'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['SeaTacEmployee'] * trip_df['travtime'] * trip_df['trexpfac'] * trip_df['PurposeWork']  * trip_df[zone_label]
    performance_measures['transit_trip_time_work_employees'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    

    # Average transit and auto trip lengths (miles) and travel times (minutes) to shop for all
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeShop'] * trip_df[zone_label]
    performance_measures['auto_trip_length_shop_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeShop'] * trip_df[zone_label]
    performance_measures['auto_trip_time_shop_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeShop'] * trip_df[zone_label]
    performance_measures['transit_trip_length_shop_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeShop'] * trip_df[zone_label]
    performance_measures['transit_trip_time_shop_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to social/recreational for all
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeSocRec'] * trip_df[zone_label]
    performance_measures['auto_trip_length_socrec_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeSocRec'] * trip_df[zone_label]
    performance_measures['auto_trip_time_socrec_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeSocRec'] * trip_df[zone_label]
    performance_measures['transit_trip_length_socrec_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeSocRec'] * trip_df[zone_label]
    performance_measures['transit_trip_time_socrec_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    return performance_measures


def main():

    performance_xwalk = pd.read_csv(os.path.join('inputs', 'model', 'lookup', 'performance_crosswalk.csv'))
    gc_taz = performance_xwalk[performance_xwalk.GC==1].TAZ.values
    nv_taz = performance_xwalk[performance_xwalk.NV==1].TAZ.values
    nc_taz = performance_xwalk[performance_xwalk.NC==1].TAZ.values
    cs_taz = performance_xwalk[performance_xwalk.CS==1].TAZ.values
    
    popsim_file_path = 'inputs/model/roster/hh_and_persons_sampled.h5'
    popsim_store = h5py.File(popsim_file_path, "r+")

    # Create a person dataframe
    popsim_per_set = popsim_store['Person']
    popsim_per_dict = {}
    for set_keys in list(popsim_per_set.keys()):
        if set_keys in ['hhno', 'pno', 'prace', 'psexpfac']:
            popsim_per_dict[set_keys] = np.asarray(popsim_per_set[set_keys])

    popsim_per_df = pd.DataFrame(popsim_per_dict)

    hdf5_file_path = 'outputs/daysim/daysim_outputs.h5'
    start_time = time.time()
    #Create the HDF5 Container if needed and open it in read/write mode using "r+"
    my_store=h5py.File(hdf5_file_path, "r+")

    #Store arrays from Daysim/Trips Group into numpy arrays, indexed by TOD.
    #This means that only trip info for the current Time Period will be included in each array.
    # Create a trip dataframe
    trip_set = my_store['Trip']
    trip_dict = {}
    for set_keys in list(trip_set.keys()):
        trip_dict[set_keys] = np.asarray(trip_set[set_keys])
    
    trip_df = pd.DataFrame(trip_dict)

    # Create a person dataframe
    tour_set = my_store['Tour']
    tour_dict = {}
    for set_keys in list(tour_set.keys()):
        tour_dict[set_keys] = np.asarray(tour_set[set_keys])
    
    tour_df = pd.DataFrame(tour_dict)

    # Create a household dataframe
    hh_set = my_store['Household']
    hh_dict = {}
    for set_keys in list(hh_set.keys()):
        hh_dict[set_keys] = np.asarray(hh_set[set_keys])
    
    hh_df = pd.DataFrame(hh_dict)

    # Create a person dataframe
    per_set = my_store['Person']
    per_dict = {}
    for set_keys in list(per_set.keys()):
        per_dict[set_keys] = np.asarray(per_set[set_keys])
    
    per_df = pd.DataFrame(per_dict)
    # per_df = per_df.merge(popsim_per_df[['hhno', 'pno', 'prace']],how='left',on=['hhno', 'pno'])

    trip_df['parent'] = reindex(tour_df.set_index('id')['parent'],trip_df.tour_id)

    # Identify SeaTac Resident by household taz
    hh_df['SeaTacResident'] = np.where((hh_df['hhtaz']<211) & (hh_df['hhtaz']>0),1,0)
    per_df['SeaTacResident'] = reindex(hh_df.set_index('hhno')['SeaTacResident'], per_df.hhno)
    tour_df['SeaTacResident'] = reindex(hh_df.set_index('hhno')['SeaTacResident'], tour_df.hhno)
    trip_df['SeaTacResident'] = reindex(hh_df.set_index('hhno')['SeaTacResident'], trip_df.hhno)

    # Identify Growth Center Resident by household taz
    hh_df['GrowthCenterResident'] = np.where((hh_df['hhtaz'].isin(gc_taz)) & (hh_df['hhtaz']>0),1,0)
    per_df['GrowthCenterResident'] = reindex(hh_df.set_index('hhno')['GrowthCenterResident'], per_df.hhno)
    tour_df['GrowthCenterResident'] = reindex(hh_df.set_index('hhno')['GrowthCenterResident'], tour_df.hhno)
    trip_df['GrowthCenterResident'] = reindex(hh_df.set_index('hhno')['GrowthCenterResident'], trip_df.hhno)

    # Identify Neighborhood Village Resident by household taz
    hh_df['NeighborhoodVillageResident'] = np.where((hh_df['hhtaz'].isin(nv_taz)) & (hh_df['hhtaz']>0),1,0)
    per_df['NeighborhoodVillageResident'] = reindex(hh_df.set_index('hhno')['NeighborhoodVillageResident'], per_df.hhno)
    tour_df['NeighborhoodVillageResident'] = reindex(hh_df.set_index('hhno')['GrowthCenterResident'], tour_df.hhno)
    trip_df['NeighborhoodVillageResident'] = reindex(hh_df.set_index('hhno')['NeighborhoodVillageResident'], trip_df.hhno)

    # Identify Neighborhood Center Resident by household taz
    hh_df['NeighborhoodCenterResident'] = np.where((hh_df['hhtaz'].isin(nc_taz)) & (hh_df['hhtaz']>0),1,0)
    per_df['NeighborhoodCenterResident'] = reindex(hh_df.set_index('hhno')['NeighborhoodCenterResident'], per_df.hhno)
    tour_df['NeighborhoodCenterResident'] = reindex(hh_df.set_index('hhno')['NeighborhoodCenterResident'], tour_df.hhno)
    trip_df['NeighborhoodCenterResident'] = reindex(hh_df.set_index('hhno')['NeighborhoodCenterResident'], trip_df.hhno)

    # Identify Corner Store Resident by household taz
    hh_df['CornerStoreResident'] = np.where((hh_df['hhtaz'].isin(cs_taz)) & (hh_df['hhtaz']>0),1,0)
    per_df['CornerStoreResident'] = reindex(hh_df.set_index('hhno')['CornerStoreResident'], per_df.hhno)
    tour_df['CornerStoreResident'] = reindex(hh_df.set_index('hhno')['CornerStoreResident'], tour_df.hhno)
    trip_df['CornerStoreResident'] = reindex(hh_df.set_index('hhno')['CornerStoreResident'], trip_df.hhno)

    # Identify SeaTac Employee by work location
    # per_df['SeaTacEmployee'] = np.where((per_df['pwtaz'] < 211) & (per_df['pwtaz']>0), 1, 0)
    per_df['SeaTacEmployee'] = np.where((per_df['SeaTacResident'] > 0) & (per_df['pwtyp']>0), 1, 0)
    tour_df['SeaTacEmployee'] = reindex(per_df.set_index('id')['SeaTacEmployee'], tour_df.person_id)
    trip_df['SeaTacEmployee'] = reindex(tour_df.set_index('id')['SeaTacEmployee'], trip_df.tour_id)

    hh_df['TransitDependent'] = np.where((hh_df['hhftw'] > hh_df['hhvehs']) | (hh_df['hhvehs'] == 0), 1, 0)
    per_df['TransitDependent'] = reindex(hh_df.set_index('hhno')['TransitDependent'], per_df.hhno)
    tour_df['TransitDependent'] = reindex(hh_df.set_index('hhno')['TransitDependent'], tour_df.hhno)
    trip_df['TransitDependent'] = reindex(hh_df.set_index('hhno')['TransitDependent'], trip_df.hhno)

    # Minority Race
    # PSRC definition
    # Value 	Description
    #   1 	White alone non-Hispanic
    #   2 	Black or African American alone non-Hispanic
    #   3 	Asian alone non-Hispanic
    #   4 	Some Other Race alone non-Hispanic
    #   5 	Two or More Races non-Hispanic
    #   6 	White Hispanic
    #   7 	Non-white Hispanic
    # per_df['Minority'] = np.where(per_df['prace']>1,1,0)
    # tour_df = tour_df.merge(per_df[['hhno','pno','Minority']], how='left', on=['hhno', 'pno'])
    # trip_df = trip_df.merge(per_df[['hhno','pno','Minority']], how='left', on=['hhno', 'pno'])

    # Label destination zones as SeaTac zones
    tour_df['SeaTacDestination'] = np.where(tour_df['tdtaz']<211,1,0)
    trip_df['SeaTacDestination'] = np.where(trip_df['dtaz']<211,1,0)

    # Define constants
    auto_mode = [3,4,5]
    transit_mode = [6]
    occ_factor = {3:1, 4:1/2, 5:1/3.5}

    work_purpose = [1]
    shop_purpose = [5]
    socrec_purpose = [7,8]

    daysim_lowinc_vot = 14.32

    # Label tour destination purpose
    tour_df['TourPurposeWork'] = 0
    tour_df.loc[tour_df.pdpurp.isin(work_purpose),'TourPurposeWork'] = 1
    trip_df['TourPurposeWork'] = reindex(tour_df.set_index('id')['TourPurposeWork'], trip_df.tour_id)

    # Label destination purpose
    trip_df['PurposeWork'] = 0
    trip_df['PurposeShop'] = 0
    trip_df['PurposeSocRec'] = 0

    trip_df.loc[trip_df.dpurp.isin(work_purpose),'PurposeWork'] = 1
    trip_df.loc[trip_df.dpurp.isin(shop_purpose),'PurposeShop'] = 1
    trip_df.loc[trip_df.dpurp.isin(socrec_purpose),'PurposeSocRec'] = 1

    # Label Auto mode
    trip_df['AutoMode'] = 0
    trip_df['TransitMode'] = 0

    trip_df.loc[trip_df['mode'].isin(auto_mode),'AutoMode'] = 1
    trip_df.loc[trip_df['mode'].isin(transit_mode),'TransitMode'] = 1

    # Add occupancy factor
    trip_df['OccFactor'] = trip_df['mode']
    trip_df['OccFactor'] = trip_df['OccFactor'].map(occ_factor).fillna(0)

    # Add low income travel
    trip_df['LowIncome'] = np.where(trip_df['vot'] < daysim_lowinc_vot, 1, 0)

    # performance_measures = {}

    # # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents
    # # Auto Trip
    # trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['SeaTacResident'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork']
    # performance_measures['auto_trip_length_work_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    # trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['SeaTacResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork']
    # performance_measures['auto_trip_time_work_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # # TransitTrip
    # trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['SeaTacResident'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork']
    # performance_measures['transit_trip_length_work_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    # trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['SeaTacResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork']
    # performance_measures['transit_trip_time_work_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac employees
    # # Auto Trip
    # trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['SeaTacDestination'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork']
    # performance_measures['auto_trip_length_work_employees'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    # trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['SeaTacDestination'] * trip_df['travtime'] * trip_df['trexpfac'] * trip_df['PurposeWork']
    # performance_measures['auto_trip_time_work_employees'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # # TransitTrip
    # trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['SeaTacDestination'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] 
    # performance_measures['transit_trip_length_work_employees'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    # trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['SeaTacDestination'] * trip_df['travtime'] * trip_df['trexpfac'] * trip_df['PurposeWork'] 
    # performance_measures['transit_trip_time_work_employees'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents that are transit dependent
    # # Auto Trip
    # trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['SeaTacResident'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] * trip_df['TransitDependent']
    # performance_measures['auto_trip_length_work_transit_dependent_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    # trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['SeaTacResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['TransitDependent']
    # performance_measures['auto_trip_time_work_transit_dependent_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # # TransitTrip
    # trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['SeaTacResident'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['TransitDependent']
    # performance_measures['transit_trip_length_work_transit_dependent_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    # trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['SeaTacResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['TransitDependent']
    # performance_measures['transit_trip_time_work_transit_dependent_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents that are low income
    # # Auto Trip
    # trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['SeaTacResident'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] * trip_df['LowIncome']
    # performance_measures['auto_trip_length_work_lowinc_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    # trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['SeaTacResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['LowIncome']
    # performance_measures['auto_trip_time_work_lowinc_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # # TransitTrip
    # trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['SeaTacResident'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['LowIncome']
    # performance_measures['transit_trip_length_work_lowinc_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    # trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['SeaTacResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['LowIncome']
    # performance_measures['transit_trip_time_work_lowinc_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents that are minority
    # # Auto Trip
    # trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['SeaTacResident'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] * trip_df['Minority']
    # performance_measures['auto_trip_length_work_minority_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    # trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['SeaTacResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['Minority']
    # performance_measures['auto_trip_time_work_minority_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # # TransitTrip
    # trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['SeaTacResident'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['Minority']
    # performance_measures['transit_trip_length_work_minority_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    # trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['SeaTacResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['Minority']
    # performance_measures['transit_trip_time_work_minority_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()


    # Use the function to calculate performance metrics:
    all_measures = calc_resident_measures(trip_df, 'SeaTacResident')
    gc_measures = calc_resident_measures(trip_df, 'GrowthCenterResident')
    nv_measures = calc_resident_measures(trip_df, 'NeighborhoodVillageResident')
    nc_measures = calc_resident_measures(trip_df, 'NeighborhoodCenterResident')
    cs_measures = calc_resident_measures(trip_df, 'CornerStoreResident')

    # # Average transit and auto trip lengths (miles) and travel times (minutes) to shop for all
    # # Auto Trip
    # trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeShop']
    # all_measures['auto_trip_length_shop_all'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    # trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeShop']
    # all_measures['auto_trip_time_shop_all'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # # TransitTrip
    # trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeShop']
    # all_measures['transit_trip_length_shop_all'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    # trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeShop']
    # all_measures['transit_trip_time_shop_all'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # # Average transit and auto trip lengths (miles) and travel times (minutes) to social/recreational for all
    # # Auto Trip
    # trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeSocRec']
    # all_measures['auto_trip_length_socrec_all'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    # trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeSocRec']
    # all_measures['auto_trip_time_socrec_all'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # # TransitTrip
    # trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeSocRec']
    # all_measures['transit_trip_length_socrec_all'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    # trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeSocRec']
    # all_measures['transit_trip_time_socrec_all'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Cacluate VMT
    # Per resident
    trip_df['VMT'] = trip_df['AutoMode'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['SeaTacResident']
    all_measures['vmt_resident'] = trip_df['VMT'].sum()/(per_df['SeaTacResident'] * per_df['psexpfac']).sum()
    
    # work tour per employee
    trip_df['VMT'] = trip_df['AutoMode'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['SeaTacEmployee'] * trip_df['TourPurposeWork']
    all_measures['vmt_worktour_employee'] = trip_df['VMT'].sum()/(per_df['SeaTacEmployee'] * per_df['psexpfac']).sum()
    
    # work tour per resident
    trip_df['VMT'] = trip_df['AutoMode'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['SeaTacResident'] * trip_df['TourPurposeWork']
    all_measures['vmt_worktour_resident'] = trip_df['VMT'].sum()/(per_df['SeaTacResident'] * per_df['psexpfac']).sum()

    # # VHT calculation
    # vht_df = pd.read_csv(os.path.join('outputs', 'network', 'vht_subarea_facility.csv'))
    # all_measures['vht_all'] = vht_df[['arterial', 'connector', 'highway']].sum().sum()
    # all_measures['vht_seatac'] = vht_df.loc[vht_df['@subarea_flag']==1][['arterial', 'connector', 'highway']].sum().sum()

    # Calculate concurrency delays index
    network_df = pd.read_csv(os.path.join('outputs','network','network_results.csv'))
    valid_links = network_df.loc[network_df['@tveh'] > 0]['ij'].unique()
    network_filtered_df = network_df.loc[network_df.ij.isin(valid_links)]

    # Low income VHT calculation
    # all_measures['vht_lowinc_seatac'] = (network_filtered_df[['@sov_inc1', '@hov2_inc1', '@hov3_inc1', '@tnc_inc1']].sum(axis=1) * network_filtered_df['@subarea_flag'] * network_filtered_df['auto_time']/60).sum()

    # Delay calculation
    delay_df = network_filtered_df.loc[network_filtered_df['tod'] == '20to5'][['ij', 'auto_time']]
    delay_df.rename(columns={'auto_time':'freeflow_time'}, inplace=True)

    # Merge delay field back onto network link df
    network_filtered_df = pd.merge(network_filtered_df, delay_df, on='ij', how='left')
    network_filtered_df['delay'] = ((network_filtered_df['auto_time']-network_filtered_df['freeflow_time'])*network_filtered_df['@tveh'])/60    # sum of (volume)*(travtime diff from freeflow)
    network_filtered_df['delay_lowinc'] = ((network_filtered_df['auto_time']-network_filtered_df['freeflow_time'])*network_filtered_df[['@sov_inc1', '@hov2_inc1', '@hov3_inc1', '@tnc_inc1']].sum(axis=1))/60    # sum of (volume)*(travtime diff from freeflow)

    all_measures['vhd_all'] = network_filtered_df['delay'].sum()
    all_measures['vhd_seatac'] = network_filtered_df.loc[network_filtered_df['@subarea_flag']==1,'delay'].sum()
    all_measures['vhd_lowinc_seatac'] = network_filtered_df.loc[network_filtered_df['@subarea_flag']==1,'delay_lowinc'].sum()

    # Read corridor links
    corridor_links_df = pd.read_csv(os.path.join('inputs', 'model', 'lookup', 'PM_edges_CC.csv'))
    network_filtered_df['corridorid'] = -1
    network_filtered_df['corridorid'] = reindex(corridor_links_df[~corridor_links_df.ID.duplicated()].set_index('ID')['Corr-Dir'], network_filtered_df['ij'])

    # Concurrency calculation
    # concurrency_df = network_filtered_df.loc[network_filtered_df['@concurrency']==1].copy()
    corridor_delay_df = network_filtered_df.loc[(~network_filtered_df.corridorid.isna()) & (network_filtered_df.tod=='16to17')]

    for corridor_index, corridor_df in corridor_delay_df.groupby('corridorid'):
        metric_name = 'delayindex_corridor_num' + str(corridor_index)
        all_measures[metric_name] = corridor_df['auto_time'].sum()/corridor_df['freeflow_time'].sum()

    # Calcualte hourly delay
    # concurrency_df['delayindex'] = concurrency_df['auto_time']/concurrency_df['freeflow_time']    # delay index = congested/freeflow time
    # all_measures['delayindex_concurrency_corridors'] = concurrency_df['auto_time'].sum()/concurrency_df['freeflow_time'].sum()    # delay index = congested/freeflow time

    # # Calculate truckroute delays index
    # truckroute_df = network_filtered_df.loc[network_filtered_df['@truck_route']==1]

    # # Calcualte hourly delay
    # truckroute_df['delayindex'] = truckroute_df['auto_time']/truckroute_df['freeflow_time']    # delay index = congested/freeflow time
    # all_measures['delayindex_freight_corridors'] = truckroute_df['auto_time'].sum()/truckroute_df['freeflow_time'].sum()    # delay index = congested/freeflow time    

    measure_df = pd.DataFrame({'Measures':all_measures.keys(),
                               'Values':all_measures.values()})
    
    measure_df['Type'] = ''
    measure_df['Type'] = np.where(measure_df.Measures.str.contains('resident'),'Resident',measure_df.Type)
    measure_df['Type'] = np.where(measure_df.Measures.str.contains('employee'),'Employee',measure_df.Type)
    measure_df['Region'] = np.where((measure_df.Measures.str.contains('all')) & (measure_df.Type==''),'All', 'SeaTAC')
    measure_df['Mode'] = ''
    measure_df['Mode'] = np.where(measure_df.Measures.str.contains('auto'),'Auto', measure_df['Mode'])
    measure_df['Mode'] = np.where((measure_df.Measures.str.contains('transit')) & (measure_df.Mode==''),'Transit', measure_df['Mode'])
    measure_df['MeasureType'] = ''
    measure_df['MeasureType'] = np.where(measure_df.Measures.str.contains('length'),'Length', measure_df['MeasureType'])
    measure_df['MeasureType'] = np.where((measure_df.Measures.str.contains('time')) & (measure_df.MeasureType==''),'Time', measure_df['MeasureType'])

    measure_df.to_csv(os.path.join('outputs', 'performance_measures', 'metrics.csv'), index=False)   

    gc_measure_df = pd.DataFrame({'Measures':gc_measures.keys(),
                               'Values':gc_measures.values()})
    
    gc_measure_df['Type'] = ''
    gc_measure_df['Type'] = np.where(gc_measure_df.Measures.str.contains('resident'),'Resident',gc_measure_df.Type)
    gc_measure_df['Type'] = np.where(gc_measure_df.Measures.str.contains('employee'),'Employee',gc_measure_df.Type)
    gc_measure_df['Region'] = np.where((gc_measure_df.Measures.str.contains('all')) & (gc_measure_df.Type==''),'All', 'SeaTAC')
    gc_measure_df['Mode'] = ''
    gc_measure_df['Mode'] = np.where(gc_measure_df.Measures.str.contains('auto'),'Auto', gc_measure_df['Mode'])
    gc_measure_df['Mode'] = np.where((gc_measure_df.Measures.str.contains('transit')) & (gc_measure_df.Mode==''),'Transit', gc_measure_df['Mode'])

    gc_measure_df.to_csv(os.path.join('outputs', 'performance_measures', 'growth_center_metrics.csv'), index=False)      

    nv_measure_df = pd.DataFrame({'Measures':nv_measures.keys(),
                               'Values':nv_measures.values()})
    
    nv_measure_df['Type'] = ''
    nv_measure_df['Type'] = np.where(nv_measure_df.Measures.str.contains('resident'),'Resident',nv_measure_df.Type)
    nv_measure_df['Type'] = np.where(nv_measure_df.Measures.str.contains('employee'),'Employee',nv_measure_df.Type)
    nv_measure_df['Region'] = np.where((nv_measure_df.Measures.str.contains('all')) & (nv_measure_df.Type==''),'All', 'SeaTAC')
    nv_measure_df['Mode'] = ''
    nv_measure_df['Mode'] = np.where(nv_measure_df.Measures.str.contains('auto'),'Auto', nv_measure_df['Mode'])
    nv_measure_df['Mode'] = np.where((nv_measure_df.Measures.str.contains('transit')) & (nv_measure_df.Mode==''),'Transit', nv_measure_df['Mode'])

    nv_measure_df.to_csv(os.path.join('outputs', 'performance_measures', 'nbhd_village_metrics.csv'), index=False)     

    nc_measure_df = pd.DataFrame({'Measures':nc_measures.keys(),
                               'Values':nc_measures.values()})
    
    nc_measure_df['Type'] = ''
    nc_measure_df['Type'] = np.where(nc_measure_df.Measures.str.contains('resident'),'Resident',nc_measure_df.Type)
    nc_measure_df['Type'] = np.where(nc_measure_df.Measures.str.contains('employee'),'Employee',nc_measure_df.Type)
    nc_measure_df['Region'] = np.where((nc_measure_df.Measures.str.contains('all')) & (nc_measure_df.Type==''),'All', 'SeaTAC')
    nc_measure_df['Mode'] = ''
    nc_measure_df['Mode'] = np.where(nc_measure_df.Measures.str.contains('auto'),'Auto', nc_measure_df['Mode'])
    nc_measure_df['Mode'] = np.where((nc_measure_df.Measures.str.contains('transit')) & (nc_measure_df.Mode==''),'Transit', nc_measure_df['Mode'])

    nc_measure_df.to_csv(os.path.join('outputs', 'performance_measures', 'nbhd_center_metrics.csv'), index=False)    

    cs_measure_df = pd.DataFrame({'Measures':cs_measures.keys(),
                               'Values':cs_measures.values()})
    
    cs_measure_df['Type'] = ''
    cs_measure_df['Type'] = np.where(cs_measure_df.Measures.str.contains('resident'),'Resident',cs_measure_df.Type)
    cs_measure_df['Type'] = np.where(cs_measure_df.Measures.str.contains('employee'),'Employee',cs_measure_df.Type)
    cs_measure_df['Region'] = np.where((cs_measure_df.Measures.str.contains('all')) & (cs_measure_df.Type==''),'All', 'SeaTAC')
    cs_measure_df['Mode'] = ''
    cs_measure_df['Mode'] = np.where(cs_measure_df.Measures.str.contains('auto'),'Auto', cs_measure_df['Mode'])
    cs_measure_df['Mode'] = np.where((cs_measure_df.Measures.str.contains('transit')) & (cs_measure_df.Mode==''),'Transit', cs_measure_df['Mode'])

    cs_measure_df.to_csv(os.path.join('outputs', 'performance_measures', 'corner_store_metrics.csv'), index=False)     

    

    # Mode share calculations
    daysim_modes = {1:'walk' , 2:'bike', 3:'sov', 4:'hov2', 5:'hov3+', 6:'transit', 8:'school bus', 9:'tnc'}
    daysim_purposes = {0:'home', 1:'work', 2:'school', 3:'escort', 4:'personal business', 5:'shop', 6:'meal', 7:'social/recreational', 8:'social/recreational', 9:'personal business', 10:'change mode inserted purpose'}
    # daysim_purposes = {0:'none/home', 1:'work', 2:'school', 3:'escort', 4:'pers.bus', 5:'shop', 6:'meal', 7:'social', 8:'recreational', 9:'medical', 10:'change mode inserted purpose'}

    

    trip_df['modelabels'] = trip_df['mode'].map(daysim_modes)
    trip_df['purposelabels'] = trip_df['dpurp'].map(daysim_purposes)
    mode_shares = trip_df.loc[(trip_df['SeaTacResident']==1) & (trip_df['dpurp']<10)].assign(persontrips=lambda _df: _df.trexpfac).groupby(['purposelabels', 'modelabels'], as_index=False)['persontrips'].sum()
    # mode_shares = trip_df.loc[(trip_df['SeaTacResident']==1) & (trip_df['dpurp']<10)].assign(vehtrips=lambda _df: _df.trexpfac*np.where(_df.OccFactor>0,_df.OccFactor,1)).groupby(['purposelabels', 'modelabels'], as_index=False)['vehtrips'].sum()
    mode_shares['propshares'] = mode_shares.groupby('purposelabels', as_index=False)['persontrips'].transform(lambda _x: _x/_x.sum())
    mode_shares = mode_shares.pivot_table(values='propshares', columns='modelabels', index='purposelabels').reset_index()
    mode_shares.columns = mode_shares.columns.values
    mode_shares.to_csv(os.path.join('outputs', 'performance_measures', 'mode_shares_resident.csv'), index=False)   

    # Sustainaible mode share calculations    
    sus_mode_shares = trip_df.loc[(trip_df.modelabels.isin(['walk', 'bike', 'transit'])) & (trip_df['SeaTacResident']==1) & (trip_df['dpurp']<10)].assign(persontrips=lambda _df: _df.trexpfac).groupby(['modelabels'], as_index=False)['persontrips'].sum()
    sus_mode_shares['propshares'] = sus_mode_shares['persontrips'].transform(lambda _x: _x/_x.sum())
    # sus_mode_shares = sus_mode_shares.pivot_table(values='propshares', columns='modelabels').reset_index()
    # sus_mode_shares.columns = mode_shares.columns.values
    sus_mode_shares.to_csv(os.path.join('outputs', 'performance_measures', 'sus_mode_shares_resident.csv'), index=False)   

    # Transit service hours
    transit_df = pd.read_csv(os.path.join('outputs','transit','transit_line_results.csv'))
    transit_summaries = transit_df.groupby('mode', as_index=False)['time', 'length'].sum()
    transit_modes_dict = {'b': 'Bus', 'c': 'Commuter Rail', 'r': 'Light Rail', 'f': 'Ferry', 'p': 'Passenger Ferry'}
    transit_summaries['TransitMode'] = transit_summaries['mode'].map(transit_modes_dict)
    transit_summaries['time'] = transit_summaries['time']/60
    transit_summaries[['TransitMode', 'time', 'length']].to_csv(os.path.join('outputs', 'performance_measures', 'transit_summaries.csv'), index=False)   

if __name__ == '__main__':
    main()