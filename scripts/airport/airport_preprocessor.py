import os
import h5py
import pandas as pd
import openmatrix as omx

import pandas as pd
import numpy as np
import os
import sys
import yaml
from collections import OrderedDict
import argparse
import subprocess


def convert_skim_hdf5_to_omx(settings):

    TAZ_index = pd.read_csv(settings['TAZ_index_file'], sep='\t')

    for skim_name, tod in settings['files_to_time_period'].items():
        hdf5_file_path = os.path.join(settings['hdf5_skim_folder'], f"{skim_name}.h5")
        omx_file_path = os.path.join(output_dir, f'skims_{skim_name}_{tod}.omx')
        print("converting", hdf5_file_path, "to", omx_file_path)

        omx_file = omx.open_file(omx_file_path,'w')
        
        hdf5_file = h5py.File(hdf5_file_path, 'r')
        for m_name, data in hdf5_file['Skims'].items():
            m = np.asanyarray(data[()]).astype(np.float32)
            if m_name == 'indices':
                continue
            # Skims are in 100's, e.g. distance of 2 miles would be 200. Time of 1 min would be 100.
            # Dividing by 100 to get miles and minutes in input skims.
            # Boardings are also in 100s.  Costs are converted from cents to dollars.
            m = m / 100
            full_m_name = f'{m_name}__{tod}'
            print(full_m_name)
            omx_file[full_m_name] = m

            # ActivitySim destination models look for a DIST skim, adding to MD omx file
            if (tod == 'MD') & (m_name == 'sov_inc2d'):
                omx_file['DIST'] = m
        
        assert len(TAZ_index.Zone_id) == m.shape[0], f"Shape of matrix ({m.shape[0]}) doesn't match number of TAZs ({len(TAZ_index.Zone_id)})"
        omx_file.create_mapping('taz', TAZ_index.Zone_id.values)
        
        omx_file.close()

def create_tours(settings):
    """ Create tours from airport model settings and probability distributions"""
    print('Creating tours.')
    employee_park = pd.read_csv(os.path.join(config_dir, settings['employee_park_fname']))
    arrival_sched = pd.read_csv(os.path.join(config_dir, settings['arrival_sched_probs_fname']))
    departure_sched = pd.read_csv(os.path.join(config_dir, settings['departure_sched_probs_fname']))
    purp_probs = pd.read_csv(os.path.join(config_dir, settings['purpose_probs_input_fname']))
    party_size_probs = pd.read_csv(os.path.join(config_dir, settings['party_size_probs_input_fname']))
    nights_probs_df = pd.read_csv(os.path.join(config_dir, settings['nights_probs_input_fname']))
    income_probs_df = pd.read_csv(os.path.join(config_dir, settings['income_probs_input_fname']))
    ext_station_probs_df = pd.read_csv(os.path.join(config_dir, settings['ext_station_probs_input_fname']))
    tour_settings = settings['tours']
    
    enplanements = tour_settings['num_enplanements']
    annualization = tour_settings['annualization_factor']
    connecting = tour_settings['connecting']
    avg_party_size = tour_settings['avg_party_size']
    airport_TAZ = tour_settings['airport_TAZ']
    airport_employee_TAZ = tour_settings['airport_employee_TAZ']
    
    num_tours = int((enplanements - connecting)/annualization/avg_party_size *2) 
    departing_tours = int(num_tours /2)
    arriving_tours = num_tours - departing_tours
    employee_tours = int(sum(employee_park['Employee Stalls']*employee_park['Share to Terminal']))
    arr_tours = pd.DataFrame(
        index=range(arriving_tours), columns=[
            'direction', 'purpose','party_size','nights', 'income'])
    arr_tours.index.name = 'id'
    arr_tours['direction'] = 'inbound'
    dep_tours = pd.DataFrame(
        index=range(departing_tours), columns=[
            'direction', 'purpose','party_size','nights', 'income'])
    dep_tours.index.name = 'id'
    dep_tours['direction'] = 'outbound'
    emp_tours = pd.DataFrame(
        index=range(employee_tours), columns=[
            'direction', 'purpose','party_size','nights', 'income'])
    emp_tours.index.name = 'id'
    emp_tours.loc[0:int(len(emp_tours)/2),'direction'] = 'inbound'
    emp_tours.loc[len(emp_tours)/2:len(emp_tours),'direction'] = 'outbound'
    
    # assign purpose
    purp_probs_sum = sum(purp_probs.Percent)
    purp_probs = {k: v / purp_probs_sum for k, v in zip(purp_probs['Purpose'],purp_probs['Percent'])}
    id_to_purp = {0:'purp0_perc',
                  1:'purp1_perc',
                  2:'purp2_perc',
                  3:'purp3_perc',
                  4:'purp4_perc',
                  5:'purp5_perc'}
    purp_cum_probs = np.array(list(purp_probs.values())).cumsum()
    
    for tour_table in [dep_tours, arr_tours]:
        purp_scaled_probs = np.subtract(
           purp_cum_probs, np.random.rand(len(tour_table), 1))
        purp_type_ids = np.argmax((purp_scaled_probs + 1.0).astype('i4'), axis=1)
        tour_table['purpose_id'] = purp_type_ids
        tour_table['purpose'] = tour_table['purpose_id'].map(id_to_purp)
    
    time_probs_list = [departure_sched, arrival_sched]
    time_col = ['start','end']
    for i,df in enumerate([dep_tours, arr_tours]):
        for purp_type, group in df.groupby('purpose'):
            num_purp_tours = len(group)
           
            #assign size
            size_probs = OrderedDict(party_size_probs[purp_type])
            # scale probs to so they sum to 1
            size_sum = sum(size_probs.values())
            size_probs = {k: v / size_sum for k,v in size_probs.items()}
            size_cum_probs = np.array(list(size_probs.values())).cumsum()
            size_scaled_probs = np.subtract(
                size_cum_probs, np.random.rand(num_purp_tours, 1))
            size = np.argmax((size_scaled_probs + 1.0).astype('i4'), axis=1)
            group['party_size'] = size
            df.loc[group.index, 'party_size'] = size

            #assign nights
            nights_probs = OrderedDict(nights_probs_df[purp_type])
            # scale probs to so they sum to 1
            nights_sum = sum(nights_probs.values())
            nights_probs = {k: v / nights_sum for k,v in nights_probs.items()}
            nights_cum_probs = np.array(list(nights_probs.values())).cumsum()
            nights_scaled_probs = np.subtract(
                nights_cum_probs, np.random.rand(num_purp_tours, 1))
            nights = np.argmax((nights_scaled_probs + 1.0).astype('i4'), axis=1)
            group['nights'] = nights
            df.loc[group.index, 'nights'] = nights
           
            #assign income
            income_probs = OrderedDict(income_probs_df[purp_type])
            # scale probs to so they sum to 1
            income_sum = sum(income_probs.values())
            income_probs = {k: v / income_sum for k,v in income_probs.items()}
            income_cum_probs = np.array(list(income_probs.values())).cumsum()
            income_scaled_probs = np.subtract(
                income_cum_probs, np.random.rand(num_purp_tours, 1))
            income = np.argmax((income_scaled_probs + 1.0).astype('i4'), axis=1)
            group['income'] = income
            df.loc[group.index, 'income'] = income


    #enumerate employee tours
    emp_tours['purpose'] = 'purp5_perc'
    emp_tours['purpose_id'] = 5
    emp_tours['party_size'] = 1
    # emp_tours['nights'] = -99
    emp_tours['nights'] = 0
    emp_tours['income'] = 'emp_inc'
    #choose employee park destination
    park_probs_sum = sum(employee_park['Employee Stalls']*employee_park['Share to Terminal'])
    # employee_park = employee_park[employee_park['Share to Terminal'] > 0]
    if park_probs_sum > 0:
        park_probs = {k: v / park_probs_sum for k, v in zip(employee_park['TAZ'],employee_park['Employee Stalls']*employee_park['Share to Terminal'])}
    else:
        park_probs = {k: v for k,v in zip(employee_park['TAZ'],employee_park['Employee Stalls']*employee_park['Share to Terminal'])}
    park_cum_probs = np.array(list(park_probs.values())).cumsum()
    id_to_park = {k:v for k,v in zip(employee_park.index,employee_park['TAZ'])}

    for tour_table in [emp_tours]:
        park_scaled_probs = np.subtract(
           park_cum_probs, np.random.rand(len(tour_table), 1))
        park_type_ids = np.argmax((park_scaled_probs + 1.0).astype('i4'), axis=1)
        tour_table['parkinglot'] = park_type_ids
        tour_table['parkinglot'] = tour_table['parkinglot'].map(id_to_park)
    if len(emp_tours) > 0:
        emp_tours['destination'] = emp_tours['parkinglot']
        emp_tours['origin'] = airport_employee_TAZ
    #choose employee mode
    # employee_park = employee_park[employee_park['Public Transit Share to Terminal']>0]
        employee_mode = employee_park.copy()
        employee_mode['PT_terminal'] = employee_mode['Public Transit Share to Terminal']
        employee_mode['Mode'] = 'EMP_TRANSIT'
        employee_mode_2 = employee_park.copy()
        employee_mode_2['PT_terminal'] = 1- employee_mode_2['Public Transit Share to Terminal']
        employee_mode_2['Mode'] = 'EMP_WALK'
        employee_mode = pd.concat([employee_mode,employee_mode_2])
        employee_mode = employee_mode.pivot(index = 'Mode', columns = 'TAZ', values = 'PT_terminal' ).reset_index().fillna(0)
        employee_mode['Name'] = pd.Series([0,1])
        final_employee = pd.DataFrame()
        for TAZ in employee_park.TAZ.unique():
            mode_probs = {k: v  for k, v in zip(employee_mode['Mode'],employee_mode[TAZ])}
            mode_cum_probs = np.array(list(mode_probs.values())).cumsum()
            id_to_mode = {k:v for k,v in zip(employee_mode['Name'],employee_mode['Mode'])}
        
            for tour_table in [emp_tours[emp_tours.parkinglot == TAZ]]:
                mode_scaled_probs = np.subtract(
                   mode_cum_probs, np.random.rand(len(tour_table), 1))
                mode_type_ids = np.argmax((mode_scaled_probs + 1.0).astype('i4'), axis=1)
                tour_table['emp_trip_mode'] = mode_type_ids
                tour_table['emp_trip_mode'] = tour_table['emp_trip_mode'].map(id_to_mode)
                final_employee = pd.concat([final_employee,tour_table])
        final_employee = final_employee.drop('parkinglot',axis = 1)
    else:
        final_employee = emp_tours.drop('parkinglot',axis = 1).copy()
        final_employee['emp_trip_mode'] = None
        
    # FIXME need external station info
    # pick external tour destination
    # ext_probs_sum = sum(ext_station_probs_df['{}.Pct'.format(settings['airport_code'])])
    ext_probs_dep = ext_station_probs_df.dep.to_dict()
    ext_probs_arr = ext_station_probs_df.arr.to_dict()

    ext_cum_probs_dep = np.array(list(ext_probs_dep.values())).cumsum()
    ext_cum_probs_arr = np.array(list(ext_probs_arr.values())).cumsum()
    id_to_ext_dep = OrderedDict(ext_station_probs_df['TAZ'])
    id_to_ext_arr = OrderedDict(ext_station_probs_df['TAZ'])
    
    ext_cum_probs = [ext_cum_probs_dep, ext_cum_probs_arr]
    id_to_ext = [id_to_ext_dep,id_to_ext_arr]
    dep_tours_ext = dep_tours[dep_tours.purpose == 'purp4_perc'].copy()
    arr_tours_ext = arr_tours[arr_tours.purpose == 'purp4_perc'].copy()
    dep_tours = dep_tours[dep_tours.purpose != 'purp4_perc'].copy()
    arr_tours = arr_tours[arr_tours.purpose != 'purp4_perc'].copy()

    ext_scaled_probs_dep = np.subtract(ext_cum_probs_dep, np.random.rand(len(dep_tours_ext),1))
    ext_scaled_probs_arr = np.subtract(ext_cum_probs_arr,np.random.rand(len(arr_tours_ext),1))
    ext_type_ids_dep = np.argmax((ext_scaled_probs_dep + 1.0).astype('i4'),axis=1)
    ext_type_ids_arr = np.argmax((ext_scaled_probs_arr + 1.0).astype('i4'),axis=1)

    dep_tours_ext['origin'] = airport_TAZ
    dep_tours_ext['destination'] = ext_type_ids_dep
    dep_tours_ext.destination = dep_tours_ext.origin.map(id_to_ext_dep)
    arr_tours_ext['destination'] = ext_type_ids_arr
    arr_tours_ext.destination = arr_tours_ext.destination.map(id_to_ext_arr)
    arr_tours_ext['origin'] = airport_TAZ

    dep_tours['destination'] = airport_TAZ
    arr_tours['origin'] = airport_TAZ

    # FIXME need external station info
    tours = pd.concat([dep_tours,arr_tours,dep_tours_ext,arr_tours_ext,final_employee],ignore_index = True).fillna(0)
        
    # tours = pd.concat([dep_tours,arr_tours,final_employee],ignore_index = True).fillna(0)
    tours['tour_id'] = np.arange(1, len(tours) +1)
    tours = tours.set_index('tour_id')
    tours['tour_category'] = 'non_mandatory'
    tours.loc[tours.origin!= airport_employee_TAZ,'origin'] = airport_TAZ
    for i,purp in enumerate(['bus','per']):
        for income in range(8):
            tours.loc[(tours.purpose_id ==i) & (tours.income ==income), 'tour_type'] = 'res_{}{}'.format(purp,income+1)
            if i == 0:
                tours.loc[(tours.purpose_id.isin([0,2])) & (tours.income == income), 'mode_segment'] = '{}{}'.format(purp,income+1)
            else:
                tours.loc[(tours.purpose_id.isin([1,3])) & (tours.income == income), 'mode_segment'] = '{}{}'.format(purp,income+1)


    tours.loc[(tours.purpose_id ==2) , 'tour_type'] = 'vis_bus'
    tours.loc[(tours.purpose_id ==3) , 'tour_type'] = 'vis_per'
    tours.loc[(tours.purpose_id ==4) , 'tour_type'] = 'external'
    tours.loc[(tours.purpose_id ==5) , 'tour_type'] = 'emp'
    tours.loc[(tours.purpose_id.isin([5])), 'mode_segment'] = 'emp'
    for income in range(8):
        tours.loc[(tours.purpose_id == 4) & (tours.income ==income), 'mode_segment'] = 'ext{}'.format(income+1)



    return tours



def create_sched_probs(settings):
    """ Create tours from airport model settings and probability distributions"""
    print('Creating tour scheduling probabilities.')
    arrival_sched = pd.read_csv(os.path.join(config_dir, settings['arrival_sched_probs_fname']))
    departure_sched = pd.read_csv(os.path.join(config_dir, settings['departure_sched_probs_fname']))
    asim_sched = [pd.DataFrame(columns = arrival_sched.columns[1:]),pd.DataFrame(columns = departure_sched.columns[1:])]
    for m, distribution in enumerate([departure_sched, arrival_sched]):
        distribution = distribution.rename(columns = {'period': 'Period'}).set_index('Period')

        if distribution.index.max() <= 40:
            # the following code is to convert from 40 CTRAMP bins to 48   half hour bins
            for i in range(1,49):
                if i <= 4:
                    asim_sched[m].loc[i] = distribution.loc[1]/4
                elif i < 43:
                    asim_sched[m].loc[i] = distribution.loc[i-3]
                else:
                    asim_sched[m].loc[i] = distribution.iloc[-1]/6
        else:
            asim_sched[m] = distribution

        asim_sched[m] = pd.DataFrame(asim_sched[m].T).reset_index().rename(columns = {'index': 'purpose'})
    
        # asim_sched[m].columns = ['period','purpose','prob']
        asim_sched[m]['outbound'] = m==0
        asim_sched[m] = asim_sched[m][['purpose','outbound'] + [i for i in range(1,49)]]
        if m ==0:
            asim_sched[m].columns = [['purpose','outbound'] + ["1_{}".format(i) for i in range(1,49)]]
        else:
            asim_sched[m].columns = [['purpose','outbound'] + ["{}_48".format(i) for i in range(1,49)]]

    tour_schl_probs = asim_sched[0].append(asim_sched[1]).fillna(0)
    tour_schl_probs.columns = tour_schl_probs.columns.get_level_values(0)
    # normalize probabilities to 1
    tour_schl_probs.set_index(['purpose', 'outbound'], inplace=True)
    tour_schl_probs = tour_schl_probs.div(tour_schl_probs.sum(axis=1), axis=0)
    tour_schl_probs.reset_index(inplace=True)
        
    return tour_schl_probs


def create_households(tours):
    print("Creating households.")
    num_tours = len(tours)

    # one household per tour
    households = pd.DataFrame({'household_id': np.arange(1,num_tours +1)})
    # not used, but ActivitySim required non-zero values
    households['home_zone_id'] = 1
    return households


def create_persons(settings, num_households):

    print("Creating persons")
    # one person per household
    persons = pd.DataFrame({'person_id': np.arange(num_households)+1})
    persons['household_id'] = np.random.choice(
        num_households , num_households, replace=False)
    persons['household_id'] = persons['household_id'] +1
    return persons



def assign_hh_p_to_tours(tours, persons):

    num_tours = len(tours)

    # assign persons and households to tours
    tours['household_id'] = np.random.choice(
        num_tours, num_tours, replace=False)
    tours['household_id'] = tours['household_id'] +1
    tours['person_id'] = persons.set_index('household_id').reindex(
        tours['household_id'])['person_id'].values

    return tours

def create_landuse(settings):

    print("Creating land use")
    # one person per household
    # input_lu = pd.read_csv(os.path.join(data_dir, settings['maz_input_fname']))
    input_lu = pd.read_csv(os.path.join(data_dir, settings['parcel_input_fname']), sep=' ')

    output_lu = input_lu.groupby('TAZ_P').sum().drop(columns=['XCOORD_P', 'YCOORD_P', 'PARCELID', 'PPRICDYP', 'PPRICHRP'])
    output_lu.index.name = 'TAZ'

    # Appending columns to landuse table for number of households in each TAZ for each income category
    print("Reading syn pop from: ", settings['hdf5_syn_pop_file'])
    with h5py.File(settings['hdf5_syn_pop_file'], "r") as f:
        hh = f['Household']
        hh_df = pd.DataFrame()
        for hh_key, data in hh.items():
            hh_df[hh_key] = data[:]
    
    hh_df['airport_income_bin'] = pd.cut(
        hh_df['hhincome'],
        bins = [-99999999,25000,50000,75000,100000,125000,150000,200000,9999999999],
        labels = ['a1','a2','a3','a4','a5','a6','a7','a8']
    )
    hh_df.rename(columns={'hhtaz': 'TAZ'}, inplace=True)
    hh_df = hh_df.groupby(['TAZ','airport_income_bin'], as_index=False)[['hhno']].count()
    income_per_zone = pd.pivot(hh_df, index='TAZ', columns='airport_income_bin', values='hhno')
    output_lu = output_lu.merge(income_per_zone, how='left', left_index=True, right_index=True).fillna(0)

    # Adding zones to landuse file that are missing 
    # this is needed so that output omx files from activitysim are the same shape as skims
    TAZ_index = pd.read_csv(settings['TAZ_index_file'], sep='\t')

    missing_internal_zones = [
        taz for taz in TAZ_index[TAZ_index.Dest_eligible == 1].Zone_id if taz not in output_lu.index
    ]
    print(f" TAZ {sorted(missing_internal_zones)} are missing")
    missing_non_dest_zones = [
        taz for taz in TAZ_index[TAZ_index.Dest_eligible == 0].Zone_id if taz not in output_lu.index
    ]
    print(f"Also need to add {len(missing_non_dest_zones)} non-destination zones")

    zones_to_add_to_landuse = missing_internal_zones + missing_non_dest_zones

    existing_zones = [taz for taz in zones_to_add_to_landuse if taz in output_lu.index]
    assert len(existing_zones) == 0, f" TAZ {existing_zones} already exist"
    
    rows_to_insert = pd.DataFrame(0, index=zones_to_add_to_landuse, columns=output_lu.columns)
    output_lu = pd.concat([output_lu, rows_to_insert])
    output_lu.index.name = 'TAZ'
    output_lu = output_lu.sort_index()
    
    # Check if taz skips integer 
    max_tazid = max(missing_internal_zones)
    all_tazids = set(range(1, max_tazid + 1))
    missing_zones = all_tazids - set(output_lu.index)
    if missing_zones:
        print(f" Destination eligible TAZ {sorted(missing_zones)} are missing")

    assert len(output_lu) == len(TAZ_index), "Ouput landuse and input TAZ index is different!"
            
    return output_lu


if __name__ == '__main__':

    # runtime args
    parser = argparse.ArgumentParser(prog='preprocessor')
    parser.add_argument(
         '-c', '--configs',
         help = 'Config Directory')
    parser.add_argument(
         '-d', '--data',
         help = 'Input data Directory')
    parser.add_argument(
         '-o', '--output',
         help = 'Output Directory')
    parser.add_argument(
        '-s', '--skims',
        action='store_true', help='Run conversion of skims from h5 to omx.')
    
    args = parser.parse_args()
    config_dir = args.configs
    data_dir = args.data
    output_dir = args.output

    print('RUNNING PREPROCESSOR!')
    with open(os.path.join(config_dir,'preprocessing.yaml')) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    # create input data
    tours = create_tours(settings)
    lu = create_landuse(settings)
    households = create_households(tours)  # 1 per tour
    persons = create_persons(settings, num_households=len(households))
    tours = assign_hh_p_to_tours(tours, persons)
    sched_probs = create_sched_probs(settings)

    # # store input files to disk
    tours.to_csv(os.path.join(
        output_dir, settings['tours_output_fname']))
    households.to_csv(os.path.join(
        output_dir, settings['households_output_fname']), index=False)
    persons.to_csv(os.path.join(
        output_dir, settings['persons_output_fname']), index=False)
    lu.to_csv(os.path.join(
        output_dir,settings['taz_output_fname']))
    sched_probs.to_csv(os.path.join(config_dir, settings['tour_scheduling_probs_output_fname']), index = False)

    # convert skims
    if args.skims:
        convert_skim_hdf5_to_omx(settings)