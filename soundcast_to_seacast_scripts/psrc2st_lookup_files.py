#Convert district correspondence in the TAZ Index file
#Nagendra Dhakar, nagendra.dhakar@rsginc.com, 09/23/16

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os, shutil
import pandas as pd
import h5py
import numpy as np

#input settings
wd = r"input_files/model/lookup"
fileNames = ['county_taz.csv', 'district_lookup.csv', 'equity_geog.csv', 'mic_taz.csv', 'parking_gz.csv', 'rgc_taz.csv',\
    'TAZ_Reg_Geog.csv', 'TAZ_TAD_County.csv']
out_dir = r"output_files"
max_seatac_zone = 210

# Crosswalk file
st_psrc_taz = r'data/psrctaz_to_sttaz.csv'
    
def runDistrictsPSRCtoBKRZones():

    # read crosswalk between tazs
    xwalk = pd.read_csv(st_psrc_taz).rename(columns={'PSRCTAZ':'taz', 'STTAZ':'taz_st', 'PropArea':'pctshare'})
    external_zone_start = 3700

    file_name = fileNames[0]
    print("Converting "+file_name)
    outfile = file_name.split('.')[0] + '_st.csv'
    temp_orig = pd.read_csv(os.path.join(wd, file_name))
    column_order = temp_orig.columns
    temp = temp_orig.merge(xwalk, on='taz', how='left')
    temp = temp.drop(columns=['taz']).rename(columns={'taz_st':'taz'}).drop_duplicates()
    temp = temp.groupby('taz').first().reset_index().drop(columns=['pctshare'])
    temp = pd.concat([temp, temp_orig[temp_orig.taz > external_zone_start]], axis=0, ignore_index=True)
    temp[column_order].to_csv(os.path.join(out_dir,outfile), line_terminator="\n", index=False)

    # Create crosswalk for household sampling
    file_name_in = fileNames[0]
    file_name_out = 'hh_sampling_region_taz.csv'
    print("Converting "+file_name_in+' to '+file_name_out)
    outfile = file_name_out.split('.')[0] + '_st.csv'
    temp_orig = pd.read_csv(os.path.join(wd, file_name_in))
    column_order = temp_orig.columns
    temp = temp_orig.merge(xwalk, on='taz', how='left')
    temp = temp.drop(columns=['taz']).rename(columns={'taz_st':'taz'}).drop_duplicates()
    temp = temp.groupby('taz').first().reset_index().drop(columns=['pctshare'])
    temp = pd.concat([temp, temp_orig[temp_orig.taz > external_zone_start]], axis=0, ignore_index=True)
    temp.loc[temp.taz <= max_seatac_zone, 'geog_name'] = 'City of SeaTac'
    temp[column_order].to_csv(os.path.join(out_dir,outfile), line_terminator="\n", index=False)

    
    file_name = fileNames[1]
    print("Converting "+file_name)
    outfile = file_name.split('.')[0] + '_st.csv'
    temp_orig = pd.read_csv(os.path.join(wd, file_name))
    column_order = temp_orig.columns
    temp = temp_orig.merge(xwalk, on='taz', how='left')
    temp['TAZ'] = temp['taz_st']
    temp = temp.drop(columns=['taz']).rename(columns={'taz_st':'taz'}).drop_duplicates()
    temp = temp.groupby('taz').first().reset_index().drop(columns=['pctshare']) 
    temp = pd.concat([temp, temp_orig[temp_orig.taz > external_zone_start]], axis=0, ignore_index=True)        
    temp[column_order].to_csv(os.path.join(out_dir,outfile), line_terminator="\n", index=False)

    
    file_name = fileNames[3]
    print("Converting "+file_name)
    outfile = file_name.split('.')[0] + '_st.csv'
    temp_orig = pd.read_csv(os.path.join(wd, file_name))
    column_order = temp_orig.columns
    temp = temp_orig.merge(xwalk, on='taz', how='left')
    temp = temp.drop(columns=['taz']).rename(columns={'taz_st':'taz'}).drop_duplicates()
    temp = temp.groupby(['lat', 'lon', 'center']).first().reset_index().drop(columns=['pctshare'])
    temp = pd.concat([temp, temp_orig[temp_orig.taz > external_zone_start]], axis=0, ignore_index=True)       
    temp[column_order].to_csv(os.path.join(out_dir,outfile), line_terminator="\n", index=False)
    
    file_name = fileNames[4]
    print("Converting "+file_name)
    outfile = file_name.split('.')[0] + '_st.csv'
    temp_orig = pd.read_csv(os.path.join(wd, file_name))
    column_order = temp_orig.columns
    temp = temp_orig.rename(columns={'TAZ':'taz'}).merge(xwalk, on='taz', how='left')
    temp.loc[~temp.taz_st.isna(),'taz'] = temp.loc[~temp.taz_st.isna(),'taz_st']
    temp = temp.drop(columns=['taz_st']).drop_duplicates().sort_values(['taz', 'ENS']).rename(columns={'taz':'TAZ'}).drop(columns=['pctshare'])
    # temp = pd.concat([temp, temp_orig[temp_orig.TAZ > external_zone_start]], axis=0, ignore_index=True)
    temp[column_order].to_csv(os.path.join(out_dir,outfile), line_terminator="\n", index=False)
    
    file_name = fileNames[5]
    print("Converting "+file_name)
    outfile = file_name.split('.')[0] + '_st.csv'
    temp_orig = pd.read_csv(os.path.join(wd, file_name))
    column_order = temp_orig.columns
    temp = temp_orig.merge(xwalk, on='taz', how='left')
    temp = temp.drop(columns=['taz']).rename(columns={'taz_st':'taz'}).drop_duplicates().drop(columns=['pctshare'])
    temp = pd.concat([temp, temp_orig[temp_orig.taz > external_zone_start]], axis=0, ignore_index=True) 
    # temp = temp.groupby(['lat_1', 'lon_1', 'geog_name']).first().reset_index()  
    temp[column_order].to_csv(os.path.join(out_dir,outfile), line_terminator="\n", index=False)
    
    file_name = fileNames[6]
    print("Converting "+file_name)
    outfile = file_name.split('.')[0] + '_st.csv'
    temp_orig = pd.read_csv(os.path.join(wd, file_name))
    column_order = temp_orig.columns
    temp = temp_orig.rename(columns={'taz_p':'taz'}).merge(xwalk, on='taz', how='left')
    temp = temp.drop(columns=['taz']).rename(columns={'taz_st':'taz_p'}).drop_duplicates()
    temp = temp.groupby(['taz_p']).first().reset_index().drop(columns=['pctshare'])
    temp = pd.concat([temp, temp_orig[temp_orig.taz_p > external_zone_start]], axis=0, ignore_index=True)  
    temp[column_order].to_csv(os.path.join(out_dir,outfile), line_terminator="\n", index=False)
    
    file_name = fileNames[7]
    print("Converting "+file_name)
    outfile = file_name.split('.')[0] + '_st.csv'
    temp_orig = pd.read_csv(os.path.join(wd, file_name))
    column_order = temp_orig.columns
    temp = temp_orig.rename(columns={'TAZ':'taz'}).merge(xwalk, on='taz', how='left')
    temp = temp.drop(columns=['taz']).rename(columns={'taz_st':'TAZ'}).drop_duplicates()
    temp = temp.groupby(['TAZ']).first().reset_index().drop(columns=['pctshare'])
    temp = pd.concat([temp, temp_orig[temp_orig.TAZ > external_zone_start]], axis=0, ignore_index=True)
    temp[column_order].to_csv(os.path.join(out_dir,outfile), line_terminator="\n", index=False)


if __name__== "__main__":
    runDistrictsPSRCtoBKRZones()

