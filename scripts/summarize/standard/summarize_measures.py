# Validation for observed data

import os, sys, shutil
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(CURRENT_DIR))
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
performance_xwalk = 'inputs/model/lookup/performance_crosswalk.csv'

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
                20:'18to20',
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


def main():

    # Get performance measure zones
    performance_xwalk_df = pd.read_csv(performance_xwalk)
    corner_store_zones = performance_xwalk_df[~performance_xwalk_df.CS.isna()].TAZ.tolist()
    nbhd_village_zones = performance_xwalk_df[~performance_xwalk_df.NV.isna()].TAZ.tolist()
    nbhd_corridor_zones = performance_xwalk_df[~performance_xwalk_df.NC.isna()].TAZ.tolist()

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
    per_df = per_df.merge(popsim_per_df[['hhno', 'pno', 'prace']],how='left',on=['hhno', 'pno'])

    hh_df['SeaTacResident'] = np.where(hh_df['hhtaz']<211,1,0)
    per_df['SeaTacResident'] = reindex(hh_df.set_index('hhno')['SeaTacResident'], per_df.hhno)
    tour_df['SeaTacResident'] = reindex(hh_df.set_index('hhno')['SeaTacResident'], tour_df.hhno)
    trip_df['SeaTacResident'] = reindex(hh_df.set_index('hhno')['SeaTacResident'], trip_df.hhno)

    hh_df['CornerStoreResident'] = np.where(hh_df['hhtaz'].isin(corner_store_zones),1,0)
    per_df['CornerStoreResident'] = reindex(hh_df.set_index('hhno')['CornerStoreResident'], per_df.hhno)
    tour_df['CornerStoreResident'] = reindex(hh_df.set_index('hhno')['CornerStoreResident'], tour_df.hhno)
    trip_df['CornerStoreResident'] = reindex(hh_df.set_index('hhno')['CornerStoreResident'], trip_df.hhno)

    hh_df['NbhdVillageResident'] = np.where(hh_df['hhtaz'].isin(nbhd_village_zones),1,0)
    per_df['NbhdVillageResident'] = reindex(hh_df.set_index('hhno')['NbhdVillageResident'], per_df.hhno)
    tour_df['NbhdVillageResident'] = reindex(hh_df.set_index('hhno')['NbhdVillageResident'], tour_df.hhno)
    trip_df['NbhdVillageResident'] = reindex(hh_df.set_index('hhno')['NbhdVillageResident'], trip_df.hhno)

    hh_df['NbhdCorridorResident'] = np.where(hh_df['hhtaz'].isin(nbhd_corridor_zones),1,0)
    per_df['NbhdCorridorResident'] = reindex(hh_df.set_index('hhno')['NbhdCorridorResident'], per_df.hhno)
    tour_df['NbhdCorridorResident'] = reindex(hh_df.set_index('hhno')['NbhdCorridorResident'], tour_df.hhno)
    trip_df['NbhdCorridorResident'] = reindex(hh_df.set_index('hhno')['NbhdCorridorResident'], trip_df.hhno)

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
    per_df['Minority'] = np.where(per_df['prace']>1,1,0)
    tour_df = tour_df.merge(per_df[['hhno','pno','Minority']], how='left', on=['hhno', 'pno'])
    trip_df = trip_df.merge(per_df[['hhno','pno','Minority']], how='left', on=['hhno', 'pno'])

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

    performance_measures = {}

    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac employees
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['SeaTacDestination'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork']
    performance_measures['auto_trip_length_work_employees'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['SeaTacDestination'] * trip_df['travtime'] * trip_df['trexpfac'] * trip_df['PurposeWork']
    performance_measures['auto_trip_time_work_employees'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['SeaTacDestination'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] 
    performance_measures['transit_trip_length_work_employees'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['SeaTacDestination'] * trip_df['travtime'] * trip_df['trexpfac'] * trip_df['PurposeWork'] 
    performance_measures['transit_trip_time_work_employees'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['SeaTacResident'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork']
    performance_measures['auto_trip_length_work_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['SeaTacResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork']
    performance_measures['auto_trip_time_work_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['SeaTacResident'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork']
    performance_measures['transit_trip_length_work_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['SeaTacResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork']
    performance_measures['transit_trip_time_work_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents that are transit dependent
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['SeaTacResident'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] * trip_df['TransitDependent']
    performance_measures['auto_trip_length_work_transit_dependent_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['SeaTacResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['TransitDependent']
    performance_measures['auto_trip_time_work_transit_dependent_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['SeaTacResident'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['TransitDependent']
    performance_measures['transit_trip_length_work_transit_dependent_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['SeaTacResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['TransitDependent']
    performance_measures['transit_trip_time_work_transit_dependent_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents that are low income
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['SeaTacResident'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] * trip_df['LowIncome']
    performance_measures['auto_trip_length_work_lowinc_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['SeaTacResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['LowIncome']
    performance_measures['auto_trip_time_work_lowinc_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['SeaTacResident'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['LowIncome']
    performance_measures['transit_trip_length_work_lowinc_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['SeaTacResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['LowIncome']
    performance_measures['transit_trip_time_work_lowinc_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents that are minority
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['SeaTacResident'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] * trip_df['Minority']
    performance_measures['auto_trip_length_work_minority_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['SeaTacResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['Minority']
    performance_measures['auto_trip_time_work_minority_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['SeaTacResident'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['Minority']
    performance_measures['transit_trip_length_work_minority_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['SeaTacResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['Minority']
    performance_measures['transit_trip_time_work_minority_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents in Corner Store
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['CornerStoreResident'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork']
    performance_measures['auto_trip_length_work_cs_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['CornerStoreResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork']
    performance_measures['auto_trip_time_work_cs_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['CornerStoreResident'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork']
    performance_measures['transit_trip_length_work_cs_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['CornerStoreResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork']
    performance_measures['transit_trip_time_work_cs_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents that are transit dependent
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['CornerStoreResident'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] * trip_df['TransitDependent']
    performance_measures['auto_trip_length_work_transit_dependent_cs_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['CornerStoreResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['TransitDependent']
    performance_measures['auto_trip_time_work_transit_dependent_cs_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['CornerStoreResident'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['TransitDependent']
    performance_measures['transit_trip_length_work_transit_dependent_cs_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['CornerStoreResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['TransitDependent']
    performance_measures['transit_trip_time_work_transit_dependent_cs_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents that are low income
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['CornerStoreResident'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] * trip_df['LowIncome']
    performance_measures['auto_trip_length_work_lowinc_cs_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['CornerStoreResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['LowIncome']
    performance_measures['auto_trip_time_work_lowinc_cs_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['CornerStoreResident'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['LowIncome']
    performance_measures['transit_trip_length_work_lowinc_cs_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['CornerStoreResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['LowIncome']
    performance_measures['transit_trip_time_work_lowinc_cs_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents that are minority
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['CornerStoreResident'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] * trip_df['Minority']
    performance_measures['auto_trip_length_work_minority_cs_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['CornerStoreResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['Minority']
    performance_measures['auto_trip_time_work_minority_cs_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['CornerStoreResident'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['Minority']
    performance_measures['transit_trip_length_work_minority_cs_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['CornerStoreResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['Minority']
    performance_measures['transit_trip_time_work_minority_cs_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents in Corner Store
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['NbhdVillageResident'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork']
    performance_measures['auto_trip_length_work_nv_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['NbhdVillageResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork']
    performance_measures['auto_trip_time_work_nv_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['NbhdVillageResident'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork']
    performance_measures['transit_trip_length_work_nv_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['NbhdVillageResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork']
    performance_measures['transit_trip_time_work_nv_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents that are transit dependent
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['NbhdVillageResident'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] * trip_df['TransitDependent']
    performance_measures['auto_trip_length_work_transit_dependent_nv_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['NbhdVillageResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['TransitDependent']
    performance_measures['auto_trip_time_work_transit_dependent_nv_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['NbhdVillageResident'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['TransitDependent']
    performance_measures['transit_trip_length_work_transit_dependent_nv_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['NbhdVillageResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['TransitDependent']
    performance_measures['transit_trip_time_work_transit_dependent_nv_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents that are low income
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['NbhdVillageResident'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] * trip_df['LowIncome']
    performance_measures['auto_trip_length_work_lowinc_nv_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['NbhdVillageResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['LowIncome']
    performance_measures['auto_trip_time_work_lowinc_nv_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['NbhdVillageResident'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['LowIncome']
    performance_measures['transit_trip_length_work_lowinc_nv_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['NbhdVillageResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['LowIncome']
    performance_measures['transit_trip_time_work_lowinc_nv_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents that are minority
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['NbhdVillageResident'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] * trip_df['Minority']
    performance_measures['auto_trip_length_work_minority_nv_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['NbhdVillageResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['Minority']
    performance_measures['auto_trip_time_work_minority_nv_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['NbhdVillageResident'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['Minority']
    performance_measures['transit_trip_length_work_minority_nv_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['NbhdVillageResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['Minority']
    performance_measures['transit_trip_time_work_minority_nv_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents in Corner Store
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['NbhdCorridorResident'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork']
    performance_measures['auto_trip_length_work_nc_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['NbhdCorridorResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork']
    performance_measures['auto_trip_time_work_nc_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['NbhdCorridorResident'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork']
    performance_measures['transit_trip_length_work_nc_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['NbhdCorridorResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork']
    performance_measures['transit_trip_time_work_nc_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents that are transit dependent
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['NbhdCorridorResident'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] * trip_df['TransitDependent']
    performance_measures['auto_trip_length_work_transit_dependent_nc_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['NbhdCorridorResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['TransitDependent']
    performance_measures['auto_trip_time_work_transit_dependent_nc_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['NbhdCorridorResident'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['TransitDependent']
    performance_measures['transit_trip_length_work_transit_dependent_nc_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['NbhdCorridorResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['TransitDependent']
    performance_measures['transit_trip_time_work_transit_dependent_nc_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents that are low income
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['NbhdCorridorResident'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] * trip_df['LowIncome']
    performance_measures['auto_trip_length_work_lowinc_nc_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['NbhdCorridorResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['LowIncome']
    performance_measures['auto_trip_time_work_lowinc_nc_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['NbhdCorridorResident'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['LowIncome']
    performance_measures['transit_trip_length_work_lowinc_nc_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['NbhdCorridorResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['LowIncome']
    performance_measures['transit_trip_time_work_lowinc_nc_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to work for SeaTac residents that are minority
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['NbhdCorridorResident'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeWork'] * trip_df['Minority']
    performance_measures['auto_trip_length_work_minority_nc_resident'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['NbhdCorridorResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['Minority']
    performance_measures['auto_trip_time_work_minority_nc_resident'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['NbhdCorridorResident'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['Minority']
    performance_measures['transit_trip_length_work_minority_nc_resident'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['NbhdCorridorResident'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeWork'] * trip_df['Minority']
    performance_measures['transit_trip_time_work_minority_nc_resident'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to shop for all
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeShop']
    performance_measures['auto_trip_length_shop_all'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeShop']
    performance_measures['auto_trip_time_shop_all'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeShop']
    performance_measures['transit_trip_length_shop_all'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeShop']
    performance_measures['transit_trip_time_shop_all'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Average transit and auto trip lengths (miles) and travel times (minutes) to social/recreational for all
    # Auto Trip
    trip_df['AutoTripLength'] = trip_df['AutoMode'] * trip_df['OccFactor'] * trip_df['travdist'] * trip_df['trexpfac'] * trip_df['PurposeSocRec']
    performance_measures['auto_trip_length_socrec_all'] = trip_df.loc[trip_df['AutoTripLength']>0,'AutoTripLength'].sum()/trip_df.loc[trip_df['AutoTripLength']>0,'trexpfac'].sum()
    trip_df['AutoTripTime'] = trip_df['AutoMode'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeSocRec']
    performance_measures['auto_trip_time_socrec_all'] = trip_df.loc[trip_df['AutoTripTime']>0,'AutoTripTime'].sum()/trip_df.loc[trip_df['AutoTripTime']>0,'trexpfac'].sum()
    # TransitTrip
    trip_df['TransitTripLength'] = trip_df['TransitMode'] * trip_df['travdist'] * trip_df['trexpfac']  * trip_df['PurposeSocRec']
    performance_measures['transit_trip_length_socrec_all'] = trip_df.loc[trip_df['TransitTripLength']>0,'TransitTripLength'].sum()/trip_df.loc[trip_df['TransitTripLength']>0,'trexpfac'].sum()
    trip_df['TransitTripTime'] = trip_df['TransitMode'] * trip_df['travtime'] * trip_df['trexpfac']  * trip_df['PurposeSocRec']
    performance_measures['transit_trip_time_socrec_all'] = trip_df.loc[trip_df['TransitTripTime']>0,'TransitTripTime'].sum()/trip_df.loc[trip_df['TransitTripTime']>0,'trexpfac'].sum()

    # Mode share calculations
    daysim_modes = {1:'walk' , 2:'bike', 3:'sov', 4:'hov2', 5:'hov3+', 6:'transit', 8:'school bus', 9:'other'}
    daysim_purposes = {0:'none/home', 1:'work', 2:'school', 3:'escort', 4:'pers.bus', 5:'shop', 6:'meal', 7:'social', 8:'recreational', 9:'medical', 10:'change mode inserted purpose'}
    trip_df['modelabels'] = trip_df['mode'].map(daysim_modes)
    trip_df['purposelabels'] = trip_df['dpurp'].map(daysim_purposes)
    mode_shares = trip_df.groupby(['purposelabels', 'modelabels'], as_index=False)['trexpfac'].sum()
    mode_shares['propshares'] = mode_shares.groupby('purposelabels', as_index=False)['trexpfac'].transform(lambda _x: _x/_x.sum())
    mode_shares = mode_shares.pivot_table(values='propshares', columns='modelabels', index='purposelabels').reset_index()
    mode_shares.columns = mode_shares.columns.values

    # Delay calculation
    vht_df = pd.read_csv(os.path.join('outputs', 'network', 'vht_subarea_facility.csv'))
    performance_measures['vht_all'] = vht_df[['arterial', 'connector', 'highway']].sum().sum()
    performance_measures['vht_seatac'] = vht_df.loc[vht_df['@subarea_flag']==1][['arterial', 'connector', 'highway']].sum().sum()

    measure_df = pd.DataFrame({'Measures':performance_measures.keys(),
                               'Values':performance_measures.values()})

    conn = create_engine('sqlite:///inputs/db/soundcast_inputs.db')

    ########################################
    # Transit Boardings by Line
    ########################################

    # Load observed data for given base year
    df_obs = pd.read_sql("SELECT * FROM observed_transit_boardings WHERE year=" + str(base_year), con=conn)
    df_obs['route_id'] = df_obs['route_id'].astype('int')
    df_line_obs = df_obs.copy()

    # Load model results and calculate modeled daily boarding by line
    df_transit_line = pd.read_csv(r'outputs\transit\transit_line_results.csv')
    df_model = df_transit_line.copy()
    df_model_daily = df_model.groupby('route_code').agg({   'description': 'first',
                                                            'boardings': 'sum'}).reset_index()

    # Merge modeled with observed boarding data
    df = df_model_daily.merge(df_obs, left_on='route_code', right_on='route_id', how='left')
    df.rename(columns={'boardings': 'modeled_5to20', 'observed_20to5': 'observed_5to20'}, inplace=True)
    df['diff'] = df['modeled_5to20']-df['observed_5to20']
    df['perc_diff'] = df['diff']/df['observed_5to20']
    df[['modeled_5to20','observed_5to20']] = df[['modeled_5to20','observed_5to20']].fillna(-1)

    # Write to file
    df.to_csv(os.path.join(measure_output_dir,'daily_boardings_by_line.csv'), index=False)

    # Write SeaTac transit routes
    seatac_routes = pd.read_csv(r'inputs\model\lookup\seatac_transit_routes.csv')
    df[df.route_id.isin(seatac_routes.seatac_route_code.values)].to_csv(os.path.join(measure_output_dir, 'daily_boardings_by_lines_seatac.csv'), index=False)

    # Write SeaTac transit routes
    seatac_airport_routes = pd.read_csv(r'inputs\model\lookup\seatac_airport_transit_routes.csv')
    df[df.route_id.isin(seatac_airport_routes.seatac_airport_route_code.values)].to_csv(os.path.join(measure_output_dir, 'daily_boardings_by_lines_airport_seatac.csv'), index=False)

    # Boardings by agency
    df_agency = df.groupby(['agency']).sum().reset_index()
    df_agency['diff'] = df_agency['modeled_5to20']-df_agency['observed_5to20']
    df_agency['perc_diff'] = df_agency['diff']/df_agency['observed_5to20']
    df_agency.to_csv(os.path.join(measure_output_dir,'daily_boardings_by_agency.csv'), 
                        index=False, columns=['agency','observed_5to20','modeled_5to20','diff','perc_diff'])

    # Boardings for special lines
    df_special = df[df['route_code'].isin(special_route_list)]
    df_special.to_csv(os.path.join(measure_output_dir,'daily_boardings_key_routes.csv'), 
                        index=False, columns=['description','route_code','agency','observed_5to20','modeled_5to20','diff','perc_diff'])

    ########################################
    # Transit Boardings by Stop
    ########################################
	
    # Light Rail
    df_obs = pd.read_sql("SELECT * FROM light_rail_station_boardings WHERE year=" + str(base_year), con=conn)

    # Scale boardings for model period 5to20, based on boardings along entire line
    light_rail_list = [6996]
    daily_factor = df_line_obs[df_line_obs['route_id'].isin(light_rail_list)]['daily_factor'].values[0]
    df_obs['observed_5to20'] = df_obs['boardings']/daily_factor

    df = pd.read_csv(r'outputs\transit\boardings_by_stop.csv')
    df = df[df['i_node'].isin(df_obs['emme_node'])]
    df = df.merge(df_obs, left_on='i_node', right_on='emme_node')
    df.rename(columns={'total_boardings':'modeled_5to20'},inplace=True)
    df['observed_5to20'] = df['observed_5to20'].astype('float')
    df.index = df['station_name']
    df_total = df.copy()[['observed_5to20','modeled_5to20']]
    df_total.loc['Total',['observed_5to20','modeled_5to20']] = df[['observed_5to20','modeled_5to20']].sum().values
    df_total.to_csv(r'outputs\validation\light_rail_boardings.csv', index=True)

    # Light Rail Transfers
    df_transfer = df.copy() 
    df_transfer['observed_transfer_rate'] = df_transfer['observed_transfer_rate'].fillna(-99).astype('float')
    df_transfer['modeled_transfer_rate'] = df_transfer['transfers']/df_transfer['modeled_5to20']
    df_transfer['diff'] = df_transfer['modeled_transfer_rate']-df_transfer['observed_transfer_rate']
    df_transfer['percent_diff'] = df_transfer['diff']/df_transfer['observed_transfer_rate']
    df_transfer = df_transfer[['modeled_transfer_rate','observed_transfer_rate','diff','percent_diff']]
    df_transfer.to_csv(r'outputs\validation\light_rail_transfers.csv', index=True)


    ########################################
    # Traffic Volumes
    ########################################

    # Count data
    
    # Model results
    df_network = pd.read_csv(r'outputs\network\network_results.csv')
    model_vol_df = df_network.copy()
    model_vol_df['@facilitytype'] = model_vol_df['@facilitytype'].map(facility_type_lookup)

    # Get daily and model volumes
    #daily_counts = counts.groupby('flag').sum()[['vehicles']].reset_index()
    daily_counts = pd.read_sql("SELECT * FROM daily_counts WHERE year=" + str(base_year), con=conn)
    df_daily = model_vol_df.groupby(['@countid']).agg({'@tveh':'sum', '@facilitytype': 'first', '@subarea_flag':'first'}).reset_index()

    # Merge observed with model
    df_daily = df_daily.merge(daily_counts, left_on='@countid', right_on='flag')

    # Merge with attributes
    df_daily.rename(columns={'@tveh': 'modeled','vehicles': 'observed','@subarea_flag':'subarea_flag'}, inplace=True)
    df_daily['diff'] = df_daily['modeled']-df_daily['observed']
    df_daily['perc_diff'] = df_daily['diff']/df_daily['observed']
    df_daily[['modeled','observed']] = df_daily[['modeled','observed']]
    df_daily['county'] = df_daily['countyid'].map(county_lookup)
    df_daily.to_csv(os.path.join(measure_output_dir,'daily_volume.csv'), 
                        index=False, columns=['@countid','@countid','county', 'subarea_flag','@facilitytype','modeled','observed','diff','perc_diff'])

    # Counts by county and facility type
    df_county_facility_counts = df_daily.groupby(['county','@facilitytype','subarea_flag'])[['observed','modeled']].sum().reset_index()
    df_county_facility_counts.to_csv(os.path.join(measure_output_dir,'daily_volume_county_facility.csv'))

    # Model results by volume
    df_daily_volume = model_vol_df.groupby(['@countid', 'ij']).agg({'@tveh':'sum', '@facilitytype': 'first', '@subarea_flag':'first'}).reset_index()
    df_daily_volume = df_daily_volume[df_daily_volume['@countid']>0].groupby(['@countid']).agg({'@tveh':'sum', '@facilitytype': 'first', '@subarea_flag':['first','count']}).reset_index()
    df_daily_volume.columns = df_daily_volume.columns.map('_'.join)
    df_daily_volume = df_daily_volume.merge(daily_counts, left_on='@countid_', right_on='flag')
    volume_bins = [0, 10000, 25000, 50000, 100000, 999999999]
    df_daily_volume['volbin'] = pd.cut(df_daily_volume['@tveh_sum'],volume_bins, right=False, labels=False)
    # df_daily_volume = df_daily_volume.groupby(['volbin','@subarea_flag_first']).agg({'@tveh_sum':'sum', 'vehicles':'sum', '@subarea_flag_count':'sum'}).reset_index()
    df_daily_volume = df_daily_volume[['@countid_', 'volbin', '@subarea_flag_first', '@tveh_sum', 'vehicles', '@subarea_flag_count']]
    df_daily_volume.columns = ['countid', 'volbin', 'subarea_flag', 'modeled', 'observed', 'nlinks']
    
    df_daily_volume.to_csv(os.path.join(measure_output_dir,'daily_volume_by_flow.csv'), index=False)

    # hourly counts
    # Create Time of Day (TOD) column based on start hour, group by TOD
    hr_counts = pd.read_sql("SELECT * FROM hourly_counts WHERE year=" + str(base_year), con=conn)
    hr_counts['tod'] = hr_counts['start_hour'].map(tod_lookup)
    counts_tod = hr_counts.groupby(['tod','flag']).sum()[['vehicles']].reset_index()

    # Account for bi-directional links or links that include HOV volumes
    hr_model = model_vol_df.groupby(['@countid','tod']).agg({'@tveh':'sum','@facilitytype':'first',
                                                  '@countyid':'first','auto_time':'first',
                                                  'type':'first', '@subarea_flag':'first'}).reset_index()

    # Join by time of day and flag ID
    df = pd.merge(hr_model, counts_tod, left_on=['@countid','tod'], right_on=['flag','tod'])
    df.rename(columns={'@tveh': 'modeled', 'vehicles': 'observed','@subarea_flag':'subarea_flag'}, inplace=True)
    df['county'] = df['@countyid'].map(county_lookup)
    df.to_csv(os.path.join(measure_output_dir,'hourly_volume.csv'), index=False)

    # Truck counts
    truck_counts = pd.read_sql("SELECT * FROM daily_truck_counts", con=conn)
    df_truck_daily = model_vol_df.groupby(['@countid']).agg({'@mveh':'sum', '@hveh':'sum', '@facilitytype': 'first', '@subarea_flag':'first'}).reset_index()
    # Merge observed with model
    df_truck_daily = df_truck_daily.merge(truck_counts, left_on='@countid', right_on='flag')

    # Merge with attributes
    df_truck_daily.rename(columns={'@mveh': 'modeled_medt','@hveh': 'modeled_hvyt','@subarea_flag':'subarea_flag'}, inplace=True)
    df_truck_daily['diff_medt'] = df_truck_daily['modeled_medt']-df_truck_daily['observed_medt']
    df_truck_daily['perc_diff_medt'] = df_truck_daily['diff_medt']/df_truck_daily['observed_medt']
    df_truck_daily[['modeled_medt','observed_medt']] = df_truck_daily[['modeled_medt','observed_medt']].apply(lambda x: round(x,2))
    df_truck_daily['diff_hvyt'] = df_truck_daily['modeled_hvyt']-df_truck_daily['observed_hvyt']
    df_truck_daily['perc_diff_hvyt'] = df_truck_daily['diff_hvyt']/df_truck_daily['observed_hvyt']
    df_truck_daily[['modeled_hvyt','observed_hvyt']] = df_truck_daily[['modeled_hvyt','observed_hvyt']].apply(lambda x: round(x,2))
    df_truck_daily['modeled'] = df_truck_daily['modeled_medt']+df_truck_daily['modeled_hvyt']
    df_truck_daily['observed'] = df_truck_daily['observed_medt']+df_truck_daily['observed_hvyt']
    df_truck_daily['diff'] = df_truck_daily['modeled']-df_truck_daily['observed']
    df_truck_daily['perc_diff'] = df_truck_daily['diff']/df_truck_daily['observed']
    df_truck_daily[['modeled','observed']] = df_truck_daily[['modeled','observed']].apply(lambda x: round(x,2))
    df_truck_daily.to_csv(os.path.join(measure_output_dir,'daily_truck_volume.csv'), 
                        index=False)

    # Roll up results to assignment periods
    df['time_period'] = df['tod'].map(sound_cast_net_dict)

    ########################################
    # Ferry Boardings by Bike
    ########################################    
    df_transit_seg = pd.read_csv(r'outputs\transit\transit_segment_results.csv')
    df_transit_seg = df_transit_seg[df_transit_seg['tod'] == '7to8']
    df_transit_seg = df_transit_seg.drop_duplicates(['i_node','line_id'])
    df_transit_line = df_transit_line[df_transit_line['tod'] == '7to8']

    _df = df_transit_line.merge(df_transit_seg, on='line_id', how='left')
    _df = _df.drop_duplicates('line_id')

    df_ij = _df.merge(df_network, left_on='i_node', right_on='i_node', how='left')
    # select only ferries
    df_ij = df_ij[df_ij['@facilitytype'].isin([15,16])]
    # both link and transit line modes should only include passenger (p) or general ferries (f)
    for colname in ['modes','mode']:
        df_ij['_filter'] = df_ij[colname].apply(lambda x: 1 if 'f' in x or 'p' in x else 0)
        df_ij = df_ij[df_ij['_filter'] == 1]
        
    df_total = df_ij.groupby('route_code').sum()[['@bvol']].reset_index()
    df_total = df_total.merge(df_ij[['description','route_code']], on='route_code').drop_duplicates('route_code')
    df_total.to_csv(r'outputs\validation\bike_ferry_boardings.csv', index=False)

    ########################################
    # Vehicle Screenlines 
    ########################################

    # Screenline is defined in "type" field for network links, all values other than 90 represent a screenline

    # Daily volume screenlines
    #df = model_vol_df.merge(model_vol_df[['i_node','j_node','type']], on=['i_node','j_node'], how='left').drop_duplicates()
    #df = model_vol_df.copy()
    #df = df.groupby('type').sum()[['@tveh']].reset_index()

    # Observed screenline data
    df_obs = pd.read_sql("SELECT * FROM observed_screenline_volumes WHERE year=" + str(base_year), con=conn)
    df_obs['observed'] = df_obs['observed'].astype('float')

    df_model = pd.read_csv(r'outputs\network\network_results.csv')
    df_model = model_vol_df.copy()
    df_model['screenline_id'] = df_model['type'].astype('str')
    # Auburn screenline is the combination of 14 and 15, change label for 14 and 15 to a combined label
    df_model.loc[df_model['screenline_id'].isin(['14','15']),'screenline_id'] = '14/15'
    _df = df_model.groupby(['screenline_id', '@subarea_flag']).sum()[['@tveh']].reset_index()

    _df = _df.merge(df_obs, on='screenline_id')
    _df.rename(columns={'@tveh':'modeled', '@subarea_flag':'subarea_flag'},inplace=True)
    _df = _df[['name','observed','modeled','county', 'subarea_flag']]
    _df['diff'] = _df['modeled']-_df['observed']
    _df = _df.sort_values('observed',ascending=False)
    _df.to_csv(r'outputs\validation\screenlines.csv', index=False)
    
    ########################################
    # External Volumes
    ########################################

    # External stations
    external_stations = range(MIN_EXTERNAL,MAX_EXTERNAL+1)
    df_model = df_model[df_model['@countid'].isin(external_stations)]
    _df = df_model.groupby('@countid').sum()[['@tveh']].reset_index()

    # Join to observed
    # By Mode
    df_obs = pd.read_sql("SELECT * FROM observed_external_volumes WHERE year=" + str(base_year), con=conn)
    newdf = _df.merge(df_obs,left_on='@countid' ,right_on='external_station')
    newdf.rename(columns={'@tveh':'modeled','AWDT':'observed'},inplace=True)
    newdf['observed'] = newdf['observed'].astype('float')
    newdf['diff'] = newdf['modeled'] - newdf['observed']
    newdf = newdf[['external_station','location','county','observed','modeled','diff']].sort_values('observed',ascending=False)
    newdf.to_csv(r'outputs\validation\external_volumes.csv',index=False)
	
	
	########################################
	# Corridor Speeds
	########################################
	
    df_model = model_vol_df.copy()
    df_model['@corridorid'] = df_model['@corridorid'].astype('int')

    df_obs = pd.read_sql_table('observed_corridor_speed', conn) 

    # Average  6 and 7 pm observed data 
    df_obs['6pm_spd_7pm_spd_avg'] = (df_obs['6pm_spd'] + df_obs['7pm_spd'])/2.0

    df_obs[['Flag1','Flag2','Flag3','Flag4','Flag5','Flag6']] = df_obs[['Flag1','Flag2','Flag3','Flag4','Flag5','Flag6']].fillna(-1).astype('int')

    tod_cols = [u'ff_spd', u'5am_spd', u'6am_spd',
    	   u'7am_spd', u'8am_spd', u'9am_spd', u'3pm_spd', u'4pm_spd', u'5pm_spd',
    	   u'6pm_spd_7pm_spd_avg']

    _df_obs = pd.melt(df_obs, id_vars='Corridor_Number', value_vars=tod_cols, var_name='tod', value_name='observed_speed')
    _df_obs = _df_obs[_df_obs['tod'] != 'ff_spd']

    # Set TOD
    tod_dict = {
        # hour of observed data represents start hour
    	'5am_spd': '5to6',
    	'6am_spd': '6to7',
    	'7am_spd': '7to8',
    	'8am_spd': '8to9',
    	'9am_spd': '9to10',
    	'3pm_spd': '15to16',
    	'4pm_spd': '16to17',
    	'5pm_spd': '17to18',
    	'6pm_spd_7pm_spd_avg': '18to20',
    }
    _df_obs['tod'] = _df_obs['tod'].map(tod_dict)

    _df = _df_obs.merge(df_obs, on=['Corridor_Number'])
    _df.drop(tod_cols, axis=1, inplace=True)

    # Get the corridor number from the flag file
    flag_lookup_df = pd.melt(df_obs[['Corridor_Number','Flag1', 'Flag2','Flag3','Flag4','Flag5','Flag6']], 
    		id_vars='Corridor_Number', value_vars=['Flag1', 'Flag2','Flag3','Flag4','Flag5','Flag6'], 
    		var_name='flag_number', value_name='flag_value')

    df_speed = df_model.merge(flag_lookup_df,left_on='@corridorid',right_on='flag_value')

    # Note that we need to separate out the Managed HOV lanes
    df_speed = df_speed[df_speed['@is_managed'] == 0]

    df_speed = df_speed.groupby(['Corridor_Number','tod','@subarea_flag']).sum()[['auto_time','length']].reset_index()
    df_speed['model_speed'] = (df_speed['length']/df_speed['auto_time'])*60
    df_speed = df_speed[(df_speed['model_speed'] < 80) & ((df_speed['model_speed'] > 0))]
    df_speed = df_speed.rename(columns={'@subarea_flag':'subarea_flag'})

    # Join to the observed data
    df_speed = df_speed.merge(_df,on=['Corridor_Number','tod'])

    df_speed.plot(kind='scatter', y='model_speed', x='observed_speed')
    df_speed.to_csv(r'outputs\validation\corridor_speeds.csv', index=False)

    ########################################
    # ACS Comparisons
    ########################################

    # Auto Ownership
    df_obs = pd.read_sql("SELECT * FROM observed_auto_ownership_acs_block_group", con=conn)
    df_obs.index = df_obs['GEOID10']
    df_obs.drop(['id','GEOID10'], inplace=True, axis=1)
    df_obs.rename(columns={'cars_none_control': 0, 'cars_one_control': 1, 'cars_two_or_more_control': 2}, inplace=True)
    df_obs_sum = df_obs.sum()
    df_obs_sum = pd.DataFrame(df_obs_sum, columns=['census'])
    df_obs = df_obs.unstack().reset_index()
    df_obs.rename(columns={'level_0': 'hhvehs', 0: 'census'}, inplace=True)

    df_model = pd.read_csv(r'outputs\agg\census\auto_ownership_block_group.csv')
    # Record categories to max of 2+
    df_model.loc[df_model['hhvehs'] >= 2, 'hhvehs'] = 2
    df_model = df_model.groupby(['hhvehs','hh_block_group']).sum()[['hhexpfac']].reset_index()

    df_model_sum = df_model.pivot_table(index='hh_block_group', columns='hhvehs', aggfunc='sum', values='hhexpfac')
    df_model_sum = df_model_sum.fillna(0)
    df_model_sum = df_model_sum.sum()
    df_model_sum = pd.DataFrame(df_model_sum.reset_index(drop=True), columns=['model'])
    df_sum = df_obs_sum.merge(df_model_sum,left_index=True,right_index=True)
    df = df_model.merge(df_obs, left_on=['hh_block_group','hhvehs'], right_on=['GEOID10','hhvehs'], how='left')
    df.rename(columns={'hhexpfac': 'modeled'}, inplace=True)
    df.to_csv(r'outputs\validation\auto_ownership_block_group.csv', index=False)

    # compare vs survey
    df_survey = pd.read_csv(r'outputs\agg\census\survey\auto_ownership_block_group.csv')
    
    df_survey.loc[df_survey['hhvehs'] >= 2, 'hhvehs'] = 2
    df_survey = df_survey.groupby(['hhvehs','hh_block_group']).sum()[['hhexpfac']].reset_index()

    df_survey_sum = df_survey.pivot_table(index='hh_block_group', columns='hhvehs', aggfunc='sum', values='hhexpfac')
    df_survey_sum = df_survey_sum.fillna(0)
    df_survey_sum = df_survey_sum.sum()
    df_survey_sum = pd.DataFrame(df_survey_sum.reset_index(drop=True), columns=['survey'])
    df_sum.merge(df_survey_sum, left_index=True, right_index=True).to_csv(r'outputs\validation\auto_ownership_census_totals.csv', index=False)

    # Commute Mode Share by Workplace Geography
    # Model Data
    df_model = pd.read_csv(r'outputs\agg\census\tour_place.csv')
    df_model = df_model[df_model['pdpurp'] == 'Work']
    df_model = df_model.groupby(['t_o_place','t_d_place','tmodetp']).sum()[['toexpfac']].reset_index()
    # rename columns
    df_model.loc[df_model['tmodetp'] == 'SOV','mode'] = 'auto'
    df_model.loc[df_model['tmodetp'] == 'HOV2','mode'] = 'auto'
    df_model.loc[df_model['tmodetp'] == 'HOV3+','mode'] = 'auto'
    df_model.loc[df_model['tmodetp'] == 'Transit','mode'] = 'transit'
    df_model.loc[df_model['tmodetp'] == 'Walk','mode'] = 'walk_and_bike'
    df_model.loc[df_model['tmodetp'] == 'Bike','mode'] = 'walk_and_bike'
    df_model = df_model.groupby(['mode','t_d_place']).sum()[['toexpfac']].reset_index()

    # Observed Data
    df = pd.read_sql("SELECT * FROM acs_commute_mode_by_workplace_geog WHERE year=" + str(base_year), con=conn)
    df = df[df['geography'] == 'place']
    df = df[df['mode'] != 'worked_at_home']
    df['geog_name'] = df['geog_name'].apply(lambda row: row.split(' city')[0])

    # FIXME: 
    # no HOV modes - is SOV including all auto trips?
    df.loc[df['mode']=='sov','mode'] = 'auto'

    # Merge modeled and observed
    df = df.merge(df_model, left_on=['geog_name','mode'], right_on=['t_d_place','mode'])
    df.rename(columns={'trips': 'observed', 'toexpfac': 'modeled', 'geog_name': 'work_place'}, inplace=True)
    df = df[['work_place','mode','modeled','observed']]
    df['percent_diff'] = (df['modeled']-df['observed'])/df['observed']
    df['diff'] = df['modeled']-df['observed']

    df.to_csv(r'outputs\validation\acs_commute_share_by_workplace_geog.csv',index=False)

    # Commute Mode Share by Home Tract
    df_model = pd.read_csv(r'outputs\agg\census\tour_dtract.csv')

    df_model[['to_tract','td_tract']] = df_model[['to_tract','td_tract']].astype('str')
    df_model['to_tract'] = df_model['to_tract'].apply(lambda row: row.split('.')[0])
    df_model['td_tract'] = df_model['td_tract'].apply(lambda row: row.split('.')[0])

    df_model = df_model[df_model['pdpurp'] == 'Work']
    df_model = df_model.groupby(['to_tract','tmodetp']).sum()[['toexpfac']].reset_index()

    # # Group all HOV together
    df_model['mode'] = df_model['tmodetp']
    df_model.loc[df_model['tmodetp'] == 'HOV2', 'mode'] = 'HOV'
    df_model.loc[df_model['tmodetp'] == 'HOV3+', 'mode'] = 'HOV'
    df_model = df_model.groupby(['to_tract','mode']).sum().reset_index()

    df_model['to_tract']=df_model['to_tract'].astype('int64')
    df_model['modeled'] = df_model['toexpfac']

    # Load the census data
    df_acs = pd.read_sql("SELECT * FROM acs_commute_mode_home_tract WHERE year=" + str(base_year), con=conn)
    
    # Select only tract records
    df_acs = df_acs[df_acs['place_type'] == 'tr']

    # Only include modes for people that travel to work (exclude telecommuters and others)
    mode_map = {'Drove Alone': 'SOV',
                'Carpooled': 'HOV',
                'Walked': 'Walk',
                'Other': 'Other',
               'Transit':'Transit'}

    df_acs['mode'] = df_acs['variable_description'].map(mode_map)
    df_acs = df_acs[-df_acs['mode'].isnull()]

    # Drop the Other mode for now
    df_acs = df_acs[df_acs['mode'] != 'Other']

    # Merge the model and observed data
    df = df_acs[['mode','geoid','place_name','estimate','margin_of_error']].merge(df_model,left_on=['geoid','mode'], 
                                                                             right_on=['to_tract','mode'])
    df.rename(columns={'estimate': 'observed', 'trexpfac': 'modeled'}, inplace=True)
    df[['observed','modeled']] = df[['observed','modeled']].astype('float')

    # Add geography columns based on tract
    parcel_geog = pd.read_sql("SELECT * FROM parcel_"+str(base_year)+"_geography", con=conn) 

    tract_geog = parcel_geog.groupby('Census2010Tract').first()[['CountyName','rg_proposed','CityName','GrowthCenterName','TAZ','District','subarea_flag']].reset_index()
    df = df.merge(tract_geog, left_on='geoid', right_on='Census2010Tract', how='left')
    df.to_csv(r'outputs\validation\acs_commute_share_by_home_tract.csv', index=False)
	
	# Copy select results to dash directory    # Copy existing CSV files for topsheet
    dash_table_list = ['daily_volume_county_facility','external_volumes','screenlines','daily_volume','daily_boardings_by_agency', 
        'daily_boardings_key_routes','light_rail_boardings']
    for fname in dash_table_list:
        shutil.copy(os.path.join(r'outputs/validation',fname+'.csv'), r'outputs/agg/dash')

if __name__ == '__main__':
    main()