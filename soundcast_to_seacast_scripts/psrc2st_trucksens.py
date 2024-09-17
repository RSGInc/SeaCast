#Convert PSRC zones to BKR zones (one to one relation - take the max area)
#Nagendra Dhakar, nagendra.dhakar@rsginc.com, 09/14/16

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.


# NOTE:
# Added special generator data manually
# copy pasted thecorresponding zone data in soundacast

import os, shutil
import pandas as pd
import h5py
import numpy as np
import csv

#inputs
wd = r"input_files\model\trucks"
truck_file = r'truck_districts.ens'
out_dir = r"output_files"

#Cross walk file
st_psrc_taz = r'data/psrctaz_to_sttaz.csv'
xwalk = pd.read_csv(st_psrc_taz).rename(columns={'PSRCTAZ':'taz', 'STTAZ':'taz_st'}).drop(columns='PropArea')

def runTruckDistrictsEns():
    print('processing: ' + truck_file )
    # read psrc zone group file
    external_zone_start = 3700
    outfile = os.path.join(out_dir,truck_file.split('.')[0] + '_st.ens')
    psrcFileName = os.path.join(wd, truck_file)
    #read header - use "#" as seperator as it is less likely to present in the file
    truck_districts_header = pd.read_table(psrcFileName, delimiter = "#", header = None, nrows = 2)
    ttdata = pd.read_table(psrcFileName, delimiter=" ", header=None, skiprows=2, names=['c', 'districts', 'Zone_id'])
    tazdata_st = xwalk.merge(ttdata,how='left', left_on='taz', right_on='Zone_id')
    tazdata_st = tazdata_st.fillna(0)
    tazdata_st['Zone_id'] = tazdata_st['taz_st']
    tazdata_st = tazdata_st[["c", "districts", "Zone_id"]].drop_duplicates()
    tazdata_st = tazdata_st.groupby('Zone_id', as_index=False).first()[["c", "districts", "Zone_id"]]
    tazdata_st = pd.concat([tazdata_st, ttdata[ttdata.Zone_id > external_zone_start]], axis=0, ignore_index=True)
    truck_districts_header.to_csv(outfile, sep = '#', header = False, index = False, \
        quoting=csv.QUOTE_NONE, quotechar='"', line_terminator='\n') #had to add space as escapechar otherwise throws an error
    with open(outfile, 'a') as file:
        tazdata_st.to_csv(file, sep = " " , header = False, index = False, line_terminator='\n')

if __name__== "__main__":
    runTruckDistrictsEns()

