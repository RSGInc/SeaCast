# from curses import echo
import os, sys, shutil
import numpy as np
import pandas as pd
import sqlalchemy as db

wd = r"input_files/db"
db_input_file = r'soundcast_inputs.db'
st_psrc_taz = r'data/psrctaz_to_sttaz.csv'
st_psrc_prcl = r'data/psrcprcl_to_sttaz.csv'
st_zones_prcl = r'data/seatac_zones_parcels.csv'
out_dir = r"output_files"
validation_file = r'data/filtered_validation_dailies.csv'
validation_truck_file = r'data/filtered_validation_dailies.csv'
validation_hourly_file = r'data/filtered_validation_hourlies.csv'
screenline_total_file = r'data/screenline_totals.csv'

db_output_file = r'soundcast_inputs_st.db'
conn = db.create_engine('sqlite:///'+os.path.join(wd,db_input_file))
if os.path.exists(os.path.join(out_dir, db_output_file)):
    os.remove(os.path.join(out_dir, db_output_file))
output_conn = db.create_engine('sqlite:///'+os.path.join(out_dir,db_output_file), echo=True)


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

def expandTazShares(table):    
    print("expand PSRC to PC zone crosswalk to full OD percents table")
    #replicate into expanded OD table
    od_table = pd.DataFrame()
    od_table["o"] = np.repeat(table["taz"].tolist(),len(table))
    od_table["st_o"] = np.repeat(table["taz_st"].tolist(),len(table))
    od_table["percent_o"] = np.repeat(table["pctshare"].tolist(),len(table))
    od_table["d"] = np.tile(table["taz"].tolist(),len(table))
    od_table["st_d"] = np.tile(table["taz_st"].tolist(),len(table))
    od_table["percent_d"] = np.tile(table["pctshare"].tolist(),len(table))    
    #calculat the share by OD group
    od_table["od"] = od_table["o"] * 10000 + od_table["d"]
    od_table["percent"] = od_table["percent_o"] * od_table["percent_d"]
    #od_table[(od_table.o==1) & (od_table.d==1)] #for debugging
    #od_table[(od_table.o==1) & (od_table.d==4)] #for debugging
    return(od_table)

def roundcumsum(values_):
    new_values = np.diff(np.insert(np.round(np.cumsum(values_)),0,0))
    return(new_values.astype(int))

# Read crosswalk
xwalk = pd.read_csv(st_psrc_taz).rename(columns={'PSRCTAZ':'taz', 'STTAZ':'taz_st', 'PropArea':'pctshare'})
xwalkprcl = pd.read_csv(st_psrc_prcl).rename(columns={'ParcelID':'parcelid', 'STTAZ':'taz_st'})
xwalkprcl = xwalkprcl.set_index('parcelid')
stzonesprcl = pd.read_csv(st_zones_prcl)
validation_df = pd.read_csv(validation_file)
validation_df = validation_df.groupby(['CountID'], as_index=False)['AADT', 'AADT_SUT', 'AADT_MUT'].sum()
validation_truck_df = pd.read_csv(validation_truck_file)
validation_truck_df = validation_truck_df.groupby(['CountID'], as_index=False)['AADT', 'AADT_SUT', 'AADT_MUT'].sum()
validation_hourly_df = pd.read_csv(validation_hourly_file)
validation_hourly_df = validation_hourly_df.groupby(['CountID', 'Hour'], as_index=False)['Vehicles'].sum()
screenline_total_df = pd.read_csv(screenline_total_file)

# Read all the table names in db
db_tables = pd.read_sql("select tbl_name from 'main'.sqlite_master", con=conn)['tbl_name'].values

# Store output tables in the dictionary and update as we go along the script
output_st_tables = {st_key: pd.read_sql('select * from '+st_key, con=conn).to_dict('list') for st_key in db_tables}

# Daily truck counts
df_daily_truck_counts = validation_truck_df.groupby('CountID', as_index=False)[['AADT_SUT','AADT_MUT']].sum()
df_daily_truck_counts = df_daily_truck_counts[(df_daily_truck_counts.AADT_SUT>0) | (df_daily_truck_counts.AADT_MUT>0)]
df_daily_truck_counts = df_daily_truck_counts.rename(columns={'CountID':'flag', 'AADT_SUT':'observed_medt', 'AADT_MUT':'observed_hvyt'})
output_st_tables['daily_truck_counts'] = df_daily_truck_counts

# Daily counts
traffic_counts_df = validation_df.groupby('CountID', as_index=False)['AADT'].sum().rename(columns={'AADT':'vehicles'})
df_daily_counts = pd.DataFrame.from_dict(output_st_tables.get('daily_counts'))
count_columns = df_daily_counts.columns
df_daily_counts = df_daily_counts.drop_duplicates()
df_daily_counts[df_daily_counts.flag.isin(traffic_counts_df.CountID)].shape
df_daily_counts = df_daily_counts[~df_daily_counts.flag.isin(traffic_counts_df.CountID)]
# df_daily_counts = df_daily_counts[df_daily_counts.flag < traffic_counts_df.CountID.min()].copy()
orig_sums = df_daily_counts['vehicles'].sum()
# df_daily_counts_st_df = traffic_counts_df.groupby(['countid', 'countyid'], as_index=False)['AADT'].sum()
df_daily_counts_st_df = traffic_counts_df.copy()
df_daily_counts_st_df['year'] = 2018
df_daily_counts_st_df['countyid'] = 33
df_daily_counts_st_df = df_daily_counts_st_df[~df_daily_counts_st_df.CountID.isin(range(3733, 3751))]
df_daily_counts_st_df = df_daily_counts_st_df.rename(columns={'CountID':'flag'})
df_daily_counts_st_df['index'] = df_daily_counts['index'].max() + df_daily_counts_st_df.index + 1
df_daily_counts_st_df = pd.concat([df_daily_counts, df_daily_counts_st_df], ignore_index=True)
# Change countyid for countid 2796 to 33 (incorrectly classified as 53)
df_daily_counts_st_df.loc[df_daily_counts_st_df.flag==2796,'countyid'] = 33
output_st_tables['daily_counts'] = df_daily_counts_st_df.to_dict('list')

# Observed hourly counts
df_hourly_counts = pd.DataFrame.from_dict(output_st_tables.get('hourly_counts'))
count_columns = df_hourly_counts.columns
df_hourly_counts = df_hourly_counts.drop_duplicates()
orig_sums = df_hourly_counts['vehicles'].sum()
df_hourly_counts_st_df = validation_hourly_df.rename(columns={'CountID':'flag', 'Hour':'start_hour', 'Vehicles':'vehicles'})
df_hourly_counts_st_df = df_hourly_counts_st_df[~df_hourly_counts_st_df.flag.isin(range(3733, 3751))]
df_hourly_counts_st_df['countyid'] = 53
df_hourly_counts_st_df['year'] = 2018
df_hourly_counts_st_df['index'] = df_hourly_counts['index'].max() + df_hourly_counts_st_df.index + 1
df_hourly_counts_st_df = pd.concat([df_hourly_counts, df_hourly_counts_st_df], ignore_index=True)
output_st_tables['hourly_counts'] = df_hourly_counts_st_df.to_dict('list')

# Observed screenline counts
df_screenline_counts = pd.DataFrame.from_dict(output_st_tables.get('observed_screenline_volumes'))
count_columns = df_screenline_counts.columns
df_screenline_counts = df_screenline_counts.drop_duplicates()
# df_screenline_counts = df_screenline_counts[df_screenline_counts.county!='Pierce'].copy()
orig_sums = df_screenline_counts['observed'].sum()
df_screenline_counts_st_df = screenline_total_df.rename(columns={'Description':'name', 'new_type':'screenline_id'})

df_screenline_counts_st_df['AADT'] = df_screenline_counts_st_df['AADT'].astype('int')
# df_screenline_counts_st_df['screenline_id'] = df_screenline_counts_st_df.screenlineid
df_screenline_counts_st_df['index'] = df_screenline_counts_st_df.index + 1 + df_screenline_counts['index'].max()
df_screenline_counts_st_df = df_screenline_counts_st_df[['index', 'screenline_id', 'name', 'AADT']].rename(columns={'AADT':'observed'})
df_screenline_counts_st_df['year'] = 2018
df_screenline_counts_st_df['county'] = 'SeaTAC'
df_screenline_counts_st_df = pd.concat([df_screenline_counts, df_screenline_counts_st_df], ignore_index=True)
output_st_tables['observed_screenline_volumes'] = df_screenline_counts_st_df.to_dict('list')

# External nonwork trips
external_taz_start = 3700
# external_nonwork_df = pd.read_sql("SELECT * FROM external_nonwork", con=conn)
external_nonwork_df = pd.DataFrame.from_dict(output_st_tables.get('external_nonwork'))
columns_to_adjust = [col_names for col_names in external_nonwork_df.columns if col_names != 'taz']
orig_sums = external_nonwork_df[columns_to_adjust].sum()
external_nonwork_st_df = xwalk.merge(external_nonwork_df, how='left', on='taz')
external_nonwork_st_df[columns_to_adjust] = external_nonwork_st_df[columns_to_adjust].apply(lambda x: x*external_nonwork_st_df.pctshare)
external_nonwork_st_df = external_nonwork_st_df.drop(columns=['taz']).rename(columns={'taz_st':'taz'}).drop_duplicates()
external_nonwork_st_df = external_nonwork_st_df.groupby('taz', as_index=False)[columns_to_adjust].sum()
external_nonwork_st_df = pd.concat([external_nonwork_st_df, external_nonwork_df[external_nonwork_df.taz>external_taz_start]], axis=0, ignore_index=True)
for col_name, dtype_ in external_nonwork_df.dtypes.items():
    external_nonwork_st_df[col_name] = external_nonwork_st_df[col_name].astype(dtype_)
new_sums = external_nonwork_st_df[columns_to_adjust].sum()
columns_not_same = [col_names for col_names in columns_to_adjust if np.abs(orig_sums[col_names]-new_sums[col_names]) > 1e-4]
if len(columns_not_same) > 0:
    print("Didn't work: external_nonwork")
else:
    output_st_tables['external_nonwork'] = external_nonwork_st_df[external_nonwork_df.columns].to_dict('list')

# External trip distribution
external_taz_start = 3700
# external_trip_dist = pd.read_sql('SELECT * FROM external_trip_distribution', con=conn)
external_trip_dist = pd.DataFrame.from_dict(output_st_tables.get('external_trip_distribution'))
external_trip_dist = external_trip_dist.drop_duplicates()
groupby_col = ['GEOID', 'Large_Area', 'PSRC_TAZ', 'BKR_TAZ', 'External_Station', 'Station_Name']
ixxi_cols = ['Total_IE', 'Total_EI', 'SOV_Veh_IE', 'SOV_Veh_EI','HOV2_Veh_IE','HOV2_Veh_EI','HOV3_Veh_IE','HOV3_Veh_EI']
orig_sums = external_trip_dist[ixxi_cols].sum()
external_trip_dist = external_trip_dist.groupby(groupby_col, as_index=False, dropna=False)[ixxi_cols].sum()
external_trip_dist_st_df = xwalk.merge(external_trip_dist.rename(columns={'PSRC_TAZ':'taz'}), how='left', on='taz')
external_trip_dist_st_df = external_trip_dist_st_df[~external_trip_dist_st_df.External_Station.isna()]
external_trip_dist_st_df[ixxi_cols] = external_trip_dist_st_df[ixxi_cols].apply(lambda x: x*external_trip_dist_st_df.pctshare)
external_trip_dist_st_df = external_trip_dist_st_df.drop(columns=['taz']).rename(columns={'taz_st':'taz'}).drop_duplicates()
external_trip_dist_st_df = external_trip_dist_st_df.rename(columns={'taz':'PSRC_TAZ'})
# groupby_col = ['GEOID', 'Large_Area', 'taz', 'BKR_TAZ', 'External_Station', 'Station_Name']
external_trip_dist_st_df = external_trip_dist_st_df.groupby(groupby_col, as_index=False, dropna=False)[ixxi_cols].sum()
external_trip_dist_st_df = pd.concat([external_trip_dist_st_df, external_trip_dist[~external_trip_dist.PSRC_TAZ.isin(xwalk.taz.unique())]], axis=0, ignore_index=True)
# Do JBLM trips
jblm_trip_dist_df = xwalk.merge(external_trip_dist_st_df.rename(columns={'External_Station':'taz'}), how='left', on='taz')
jblm_trip_dist_df = jblm_trip_dist_df[~jblm_trip_dist_df.PSRC_TAZ.isna()]
jblm_trip_dist_df[ixxi_cols] = jblm_trip_dist_df[ixxi_cols].apply(lambda x: x*jblm_trip_dist_df.pctshare)
jblm_trip_dist_df = jblm_trip_dist_df.drop(columns=['taz']).rename(columns={'taz_st':'taz'}).drop_duplicates()
jblm_trip_dist_df = jblm_trip_dist_df.rename(columns={'taz':'External_Station'})
jblm_trip_dist_df = jblm_trip_dist_df.groupby(groupby_col, as_index=False, dropna=False)[ixxi_cols].sum()
external_trip_dist_st_df = pd.concat([external_trip_dist_st_df[~external_trip_dist_st_df.External_Station.isin(xwalk.taz.values)], jblm_trip_dist_df], axis=0, ignore_index=True)
new_sums = external_trip_dist_st_df[ixxi_cols].sum()
columns_not_same = [col_names for col_names in ixxi_cols if np.abs(orig_sums[col_names]-new_sums[col_names]) > 1e-4]
if len(columns_not_same) > 0:
    print("Didn't work: external_trip_distribution")
else:
    output_st_tables['external_trip_distribution'] = external_trip_dist_st_df[external_trip_dist.columns].to_dict('list')

# Group Quarters
# total_gq_df = pd.read_sql_query("SELECT * FROM group_quarters", con=conn)
total_gq_df = pd.DataFrame.from_dict(output_st_tables.get('group_quarters'))
total_gq_df = total_gq_df.drop_duplicates()
groupby_col = ['geoid10', 'taz', 'year']
gq_cols = ['dorm_share','military_share','other_share', 'group_quarters']
total_gq_df = total_gq_df.groupby(groupby_col+gq_cols[:-1], as_index=False, dropna=False)[gq_cols[-1]].sum()
orig_sums = total_gq_df[gq_cols[-1]].sum()
total_gq_df_st_df = xwalk.merge(total_gq_df, how='left', on='taz')
total_gq_df_st_df[gq_cols[-1]] = total_gq_df_st_df[gq_cols[-1]]*total_gq_df_st_df.pctshare
total_gq_df_st_df = total_gq_df_st_df.drop(columns=['taz']).rename(columns={'taz_st':'taz'}).drop_duplicates()
total_gq_df_st_df = total_gq_df_st_df.groupby(groupby_col+gq_cols[:-1], as_index=False, dropna=False)[gq_cols[-1]].sum()
total_gq_df_st_df = pd.concat([total_gq_df_st_df, total_gq_df[~total_gq_df.taz.isin(xwalk.taz.unique())]], axis=0, ignore_index=True)
for col_name, dtype_ in total_gq_df.dtypes.items():
    total_gq_df_st_df[col_name] = total_gq_df_st_df[col_name].astype(dtype_)
new_sums = total_gq_df_st_df[gq_cols[-1]].sum()
columns_not_same = (np.abs(orig_sums-new_sums) > 1e-4)
if columns_not_same:
    print("Didn't work: group_quarters")
else:
    output_st_tables['group_quarters'] = total_gq_df_st_df[total_gq_df.columns].to_dict('list')


# Heavy Trucks
external_taz_start = 3700
heavy_trucks = pd.DataFrame.from_dict(output_st_tables.get('heavy_trucks'))
heavy_trucks = heavy_trucks.drop_duplicates()
groupby_col = ['record', 'atri_zone', 'taz', 'year']
data_col = ['htkpro', 'htkatt']
heavy_trucks = heavy_trucks.groupby(groupby_col, as_index=False, dropna=False)[data_col].sum()
orig_sums = heavy_trucks[data_col].sum()
heavy_trucks_st_df = xwalk.merge(heavy_trucks, how='left', on='taz').dropna()
heavy_trucks_st_df[data_col] = heavy_trucks_st_df[data_col].apply(lambda x: x*heavy_trucks_st_df.pctshare)
heavy_trucks_st_df = heavy_trucks_st_df.drop(columns=['taz']).rename(columns={'taz_st':'taz'}).drop_duplicates().dropna()
heavy_trucks_st_df = heavy_trucks_st_df.groupby(groupby_col, as_index=False, dropna=False)[data_col].sum()
heavy_trucks_st_df = pd.concat([heavy_trucks_st_df, heavy_trucks[~heavy_trucks.taz.isin(xwalk.taz.unique())]], axis=0, ignore_index=True)
for col_name, dtype_ in heavy_trucks.dtypes.items():
    heavy_trucks_st_df[col_name] = heavy_trucks_st_df[col_name].astype(dtype_)
new_sums = heavy_trucks_st_df[data_col].sum()
columns_not_same = [col_names for col_names in data_col if np.abs(orig_sums[col_names]-new_sums[col_names]) > 1e-4]
if len(columns_not_same) > 0:
    print("Didn't work: heavy_trucks")
else:
    output_st_tables['heavy_trucks'] = heavy_trucks_st_df[heavy_trucks.columns].to_dict('list')

# JBLM Trips
external_taz_start = 3700
jblm_trips = pd.DataFrame.from_dict(output_st_tables.get('jblm_trips'))
jblm_trips = jblm_trips.drop_duplicates()
groupby_col = ['record', 'atri_zone', 'taz', 'year']
data_col = ['trips']
orig_sums = jblm_trips[data_col].sum()
odShares = expandTazShares(xwalk)
jblm_trips_st_df = jblm_trips.merge(odShares, how='left', left_on=['origin_zone', 'destination_zone'],right_on=['o','d']).dropna()
jblm_trips_st_df['trips'] = jblm_trips_st_df['trips'] * jblm_trips_st_df['percent']
ext_o = jblm_trips.loc[jblm_trips.origin_zone==3733]
ext_d = jblm_trips.loc[jblm_trips.destination_zone==3733]
ext_o_trips_st_df = ext_o.merge(xwalk, how='left', left_on=['destination_zone'],right_on=['taz']).dropna()
ext_o_trips_st_df['trips'] = ext_o_trips_st_df['trips'] * ext_o_trips_st_df['pctshare']
ext_o_trips_st_df = ext_o_trips_st_df.rename(columns={'taz':'d', 'taz_st':'st_d', 'pctshare':'percent_d'})
ext_o_trips_st_df['o'] = ext_o_trips_st_df['origin_zone']
ext_o_trips_st_df['st_o'] = ext_o_trips_st_df['origin_zone']
ext_o_trips_st_df['percent_o'] = 1
ext_o_trips_st_df['percent'] = ext_o_trips_st_df['percent_d']
ext_o_trips_st_df['od'] = ext_o_trips_st_df['o'] * 10000 + ext_o_trips_st_df['d']
ext_o_trips_st_df = ext_o_trips_st_df[jblm_trips_st_df.columns]

ext_d_trips_st_df = ext_d.merge(xwalk, how='left', left_on=['origin_zone'],right_on=['taz']).dropna()
ext_d_trips_st_df['trips'] = ext_d_trips_st_df['trips'] * ext_d_trips_st_df['pctshare']
ext_d_trips_st_df = ext_d_trips_st_df.rename(columns={'taz':'o', 'taz_st':'st_o', 'pctshare':'percent_o'})
ext_d_trips_st_df['d'] = ext_d_trips_st_df['destination_zone']
ext_d_trips_st_df['st_d'] = ext_d_trips_st_df['destination_zone']
ext_d_trips_st_df['percent_d'] = 1
ext_d_trips_st_df['percent'] = ext_d_trips_st_df['percent_o']
ext_d_trips_st_df['od'] = ext_d_trips_st_df['o'] * 10000 + ext_d_trips_st_df['d']
ext_d_trips_st_df = ext_d_trips_st_df[jblm_trips_st_df.columns]

jblm_trips_st_df = pd.concat([jblm_trips_st_df.loc[~((jblm_trips_st_df.origin_zone==3733)|(jblm_trips_st_df.destination_zone==3733))], ext_o_trips_st_df, ext_d_trips_st_df], axis=0)
groupby_col = ['year', 'matrix_id', 'geoid10', 'st_o', 'st_d', 'trip_direction', 'jblm_gate']
jblm_trips_st_df = jblm_trips_st_df.groupby(groupby_col, as_index=False, dropna=False)[data_col].sum()
jblm_trips_st_df = jblm_trips_st_df.rename(columns={'st_o':'origin_zone', 'st_d':'destination_zone'})
jblm_trips_st_df['record'] = jblm_trips_st_df.index+1
jblm_trips_st_df = jblm_trips_st_df[jblm_trips.columns]
new_sums = jblm_trips_st_df[data_col].sum()
columns_not_same = (np.abs(orig_sums-new_sums) > 1e-4).values
if columns_not_same:
    print("Didn't work: jblm_trips")
else:
    output_st_tables['jblm_trips'] = jblm_trips_st_df[jblm_trips.columns].to_dict('list')


# Parking zones
df_parking_zones = pd.DataFrame.from_dict(output_st_tables.get('parking_zones'))
df_parking_zones = df_parking_zones.drop_duplicates()
df_parking_zones_st_df = df_parking_zones.rename(columns={'TAZ':'taz'}).merge(xwalk, how='left', on='taz').dropna()
df_parking_zones_st_df = df_parking_zones_st_df.drop(columns=['taz']).rename(columns={'taz_st':'TAZ'}).drop_duplicates().dropna()
df_parking_zones_st_df = df_parking_zones_st_df.sort_values('pctshare').drop(columns=['pctshare'])
df_parking_zones_st_df = df_parking_zones_st_df.groupby('TAZ').first().reset_index()
for col_name, dtype_ in df_parking_zones.dtypes.items():
    df_parking_zones_st_df[col_name] = df_parking_zones_st_df[col_name].astype(dtype_)
output_st_tables['parking_zones'] = df_parking_zones_st_df[df_parking_zones.columns].to_dict('list')

# Parking costs add 2044 data
df_parking_costs = pd.DataFrame.from_dict(output_st_tables.get('parking_costs'))
df_parking_costs = df_parking_costs.drop_duplicates()
df_parking_costs_2044 = df_parking_costs.loc[df_parking_costs.year==2040]
df_parking_costs_2044['year'] = 2044
df_parking_costs_st_df = pd.concat([df_parking_costs, df_parking_costs_2044], ignore_index=True)
for col_name, dtype_ in df_parking_costs.dtypes.items():
    df_parking_costs_st_df[col_name] = df_parking_costs_st_df[col_name].astype(dtype_)
output_st_tables['parking_costs'] = df_parking_costs_st_df[df_parking_costs.columns].to_dict('list')


# Start and Running Emission Rates add 2044 data
df_start_emission_rates_by_veh_type = pd.DataFrame.from_dict(output_st_tables.get('start_emission_rates_by_veh_type'))
df_start_emission_rates_by_veh_type = df_start_emission_rates_by_veh_type.drop_duplicates()
df_start_emission_rates_by_veh_type_2044 = df_start_emission_rates_by_veh_type.loc[df_start_emission_rates_by_veh_type.year==2040]
df_start_emission_rates_by_veh_type_2044['year'] = 2044
df_start_emission_rates_by_veh_type_st_df = pd.concat([df_start_emission_rates_by_veh_type, df_start_emission_rates_by_veh_type_2044], ignore_index=True)
for col_name, dtype_ in df_start_emission_rates_by_veh_type.dtypes.items():
    df_start_emission_rates_by_veh_type_st_df[col_name] = df_start_emission_rates_by_veh_type_st_df[col_name].astype(dtype_)
output_st_tables['start_emission_rates_by_veh_type'] = df_start_emission_rates_by_veh_type_st_df[df_start_emission_rates_by_veh_type.columns].to_dict('list') 


# Start and Running Emission Rates add 2044 data
df_running_emission_rates_by_veh_type = pd.DataFrame.from_dict(output_st_tables.get('running_emission_rates_by_veh_type'))
df_running_emission_rates_by_veh_type = df_running_emission_rates_by_veh_type.drop_duplicates()
df_running_emission_rates_by_veh_type_2044 = df_running_emission_rates_by_veh_type.loc[df_running_emission_rates_by_veh_type.year==2040]
df_running_emission_rates_by_veh_type_2044['year'] = 2044
df_running_emission_rates_by_veh_type_st_df = pd.concat([df_running_emission_rates_by_veh_type, df_running_emission_rates_by_veh_type_2044], ignore_index=True)
for col_name, dtype_ in df_running_emission_rates_by_veh_type.dtypes.items():
    df_running_emission_rates_by_veh_type_st_df[col_name] = df_running_emission_rates_by_veh_type_st_df[col_name].astype(dtype_)
output_st_tables['running_emission_rates_by_veh_type'] = df_running_emission_rates_by_veh_type_st_df[df_running_emission_rates_by_veh_type.columns].to_dict('list') 

# PSRC zones
# df_psrc = pd.read_sql("SELECT * FROM psrc_zones", con=conn)
df_psrc = pd.DataFrame.from_dict(output_st_tables.get('psrc_zones'))
df_psrc = df_psrc.drop_duplicates()
df_psrc_st_df = df_psrc.merge(xwalk, how='left', on='taz').dropna()
df_psrc_st_df = df_psrc_st_df.drop(columns=['taz', 'pctshare']).rename(columns={'taz_st':'taz'}).drop_duplicates().dropna()
df_psrc_st_df = df_psrc_st_df[df_psrc.columns]
df_psrc_st_df = pd.concat([df_psrc_st_df, df_psrc.loc[df_psrc.external==1,]], axis=0, ignore_index=True)
df_psrc_st_df = df_psrc_st_df.drop(columns='record').drop_duplicates().reset_index(drop=True)
df_psrc_st_df['record'] = df_psrc_st_df.index+1
for col_name, dtype_ in df_psrc.dtypes.items():
    df_psrc_st_df[col_name] = df_psrc_st_df[col_name].astype(dtype_)
output_st_tables['psrc_zones'] = df_psrc_st_df[df_psrc.columns].to_dict('list')

# SeaTac Airport
# df_seatac = pd.read_sql("SELECT * FROM seatac", con=conn)
df_seatac = pd.DataFrame.from_dict(output_st_tables.get('seatac'))
df_seatac = df_seatac.drop_duplicates()
orig_sums = df_seatac['enplanements'].sum()
df_seatac_st_df = df_seatac.merge(xwalk, how='left', on='taz').dropna()
df_seatac_st_df['enplanements'] = df_seatac_st_df['enplanements'] * df_seatac_st_df['pctshare']
df_seatac_st_df = df_seatac_st_df.drop(columns=['taz', 'pctshare']).rename(columns={'taz_st':'taz'}).drop_duplicates().dropna()
df_seatac_st_df = df_seatac_st_df[df_seatac.columns]
# df_seatac_st_df['enplanements'] = df_seatac_st_df.groupby('taz')['enplanements'].transform(lambda x: roundcumsum(x.values))
df_seatac_st_df['enplanements'] = roundcumsum(df_seatac_st_df.enplanements.values)
for col_name, dtype_ in df_seatac.dtypes.items():
    df_seatac_st_df[col_name] = df_seatac_st_df[col_name].astype(dtype_)
new_sums = df_seatac_st_df['enplanements'].sum()
columns_not_same = np.abs(orig_sums-new_sums) > 1e-4
if columns_not_same:
    print("Didn't work: seatac")
else:
    output_st_tables['seatac'] = df_seatac_st_df[df_seatac.columns].to_dict('list')

# SeaTac Airport
# df_special = pd.read_sql("SELECT * FROM special_generators", con=conn)
df_special = pd.DataFrame.from_dict(output_st_tables.get('special_generators'))
df_special = df_special.drop_duplicates()
orig_sums = df_special.trips.sum()
df_special_st_df = df_special.merge(xwalk, how='left', on='taz').dropna()
df_special_st_df['trips'] = df_special_st_df['trips'] * df_special_st_df['pctshare']
df_special_st_df = df_special_st_df.drop(columns=['taz', 'pctshare']).rename(columns={'taz_st':'taz'}).drop_duplicates().dropna()
df_special_st_df = df_special_st_df[df_special.columns]
for col_name, dtype_ in df_special.dtypes.items():
    df_special_st_df[col_name] = df_special_st_df[col_name].astype(dtype_)
new_sums = df_special_st_df.trips.sum()
columns_not_same = np.abs(orig_sums-new_sums) > 1e-4
if columns_not_same:
    print("Didn't work: special_generators")
else:
    output_st_tables['special_generators'] = df_special_st_df[df_special.columns].to_dict('list')

# TAZ Geography
# county_df = pd.read_sql('SELECT * FROM taz_geography', con=conn)
county_df = pd.DataFrame.from_dict(output_st_tables.get('taz_geography'))
county_df = county_df.drop_duplicates()
county_df_st_df = county_df.merge(xwalk, how='left', on='taz').dropna()
county_df_st_df = county_df_st_df.drop(columns=['taz', 'pctshare']).rename(columns={'taz_st':'taz'}).drop_duplicates().dropna()
county_df_st_df = county_df_st_df[county_df.columns]
for col_name, dtype_ in county_df.dtypes.items():
    county_df_st_df[col_name] = county_df_st_df[col_name].astype(dtype_)

county_df_st_df['subarea_flag'] = 0
county_df_st_df.loc[county_df_st_df.taz.isin(stzonesprcl.STTAZ.values),'subarea_flag']=1
output_st_tables['taz_geography'] = county_df_st_df[county_df.columns].to_dict('list')

# Parcel 2018 Geography
# parcel_df = pd.read_sql('SELECT * FROM taz_geography', con=conn)
parcel_df = pd.DataFrame.from_dict(output_st_tables.get('parcel_2018_geography'))
parcel_df = parcel_df.drop_duplicates()
parcel_df_st_df = parcel_df.copy()
parcel_df_st_df['subarea_flag'] = 0
parcel_df_st_df.loc[parcel_df_st_df.ParcelID.isin(stzonesprcl.ParcelID.values),'subarea_flag']=1
output_st_tables['parcel_2018_geography'] = parcel_df_st_df.to_dict('list')

for key,value in output_st_tables.items():
    print('Writing table '+ key +' to the database')
    pd.DataFrame.from_dict(value).to_sql(name=key, con=output_conn, index=False)