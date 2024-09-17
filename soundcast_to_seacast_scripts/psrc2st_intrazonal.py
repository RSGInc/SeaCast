#Convert PSRC zones to PSRC zones (one to one relation - take the max area)
#Nagendra Dhakar, nagendra.dhakar@rsginc.com, 09/21/16

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
import csv

# from sympy import N

# inputs
wd = r"input_files\model\intrazonals"

files_list = ["origin_tt.in", "destination_tt.in", "taz_acres.in"]

#Crosswalk file
st_psrc_taz = r'data\psrctaz_to_sttaz.csv'
# Requires shapefile to get area in acres
out_dir = r"output_files"

    
def runPSRCtoPCZones():
    #read terminal times
    # read crosswalk between tazs
    xwalk = pd.read_csv(st_psrc_taz)
    external_taz_start = 3700
    #number of rows in the begining of the file before the actual data - user input
    header_rows = [5, 5, 3]
    for i in range(0, len(files_list)):
        file = files_list[i]
        print ("updating: " + file)
        outFileName = os.path.join(out_dir, file)
        if i < 2:
            #psrc file
            psrcFileName = file
            psrcFileName = os.path.join(wd, psrcFileName)
            #read header - use "#" as seperator as it is less likely to present in the file
            header = pd.read_table(psrcFileName, delimiter = "#", header = None, nrows = header_rows[i])
            if i == 0:
                ttdata = pd.read_table(psrcFileName, delimiter=" ", header=None, skiprows=header_rows[i], usecols=[1,2,3], names=['Zone_id', 'c', 'termtime'])
                extdata = ttdata[ttdata.Zone_id > external_taz_start]
            else:
                ttdata = pd.read_table(psrcFileName, delimiter=" ", header=None, skiprows=header_rows[i], usecols=[1,2,3], names=['c', 'Zone_id', 'termtime'])
                ttdata['Zone_id'] = ttdata['Zone_id'].str.replace(':','').astype(int)
                extdata = ttdata[ttdata.Zone_id > external_taz_start]
                extdata['Zone_id'] = extdata["Zone_id"].astype(str) + ":"
            tazdata_st = xwalk.merge(ttdata,how='left', left_on='PSRCTAZ', right_on='Zone_id')
            tazdata_st = tazdata_st.fillna(0)
            tazdata_st['Zone_id'] = tazdata_st['STTAZ'].astype(np.int32)
            tazdata_st = tazdata_st[["Zone_id", "c", "termtime"]]
            tazdata_st = tazdata_st.groupby('Zone_id', as_index=False)['termtime'].aggregate(lambda x: x.max())
            if i==0: #origin file
                tazdata_st["c"] = "all:"
                tazdata_st['first'] = np.nan
                tazdata_st = tazdata_st[["first", "Zone_id", "c", "termtime"]]
            else: #destination file
                tazdata_st["c"] = "all" #space before the word 'all' is intentional, the model throws an error if space is not there
                tazdata_st["Zone_id"] = tazdata_st["Zone_id"].astype(str) + ":"
                tazdata_st['first'] = np.nan
                tazdata_st = tazdata_st[["first", "c", "Zone_id", "termtime"]]
            tazdata_st = pd.concat([tazdata_st, extdata], axis=0, ignore_index=True)
        else:
            #psrc file
            psrcFileName = file
            psrcFileName = os.path.join(wd, psrcFileName)
            #read header - use "#" as seperator as it is less likely to present in the file
            header = pd.read_table(psrcFileName, delimiter = "#", header = None, nrows = header_rows[i])
            ttdata = pd.read_table(psrcFileName, delimiter=" ", header=None, skiprows=header_rows[i], usecols=[1,2,3], names=['Zone_id', 'c', 'acres'])
            tazdata_st = xwalk.merge(ttdata,how='left', left_on='PSRCTAZ', right_on='Zone_id')
            tazdata_st = tazdata_st.fillna(0)
            tazdata_st['Zone_id'] = tazdata_st['STTAZ'].astype(np.int32)
            tazdata_st['PropAcres'] = tazdata_st.acres * tazdata_st.PropArea
            tazdata_st = tazdata_st[["Zone_id", "c", "PropAcres"]]
            tazdata_st = tazdata_st.groupby('Zone_id', as_index=False)['PropAcres'].aggregate(lambda x: x.sum())
            tazdata_st['c'] = 'all:'
            tazdata_st['first'] = np.nan
            tazdata_st = tazdata_st[["first", "Zone_id", "c", "PropAcres"]].sort_values('Zone_id')
        # write - first header and then append the updated data
        outfile = outFileName.split(".")[0]
        outfile = outfile + "_st.in"
        header.to_csv(outfile, sep = '#', header = False, index = False, quoting=csv.QUOTE_NONE, quotechar='"', line_terminator='\n') #had to add space as escapechar otherwise throws an error
        with open(outfile, 'a') as file:
            tazdata_st.to_csv(file, sep = " ", header = False, index = False, line_terminator='\n')


if __name__== "__main__":
    runPSRCtoPCZones()

