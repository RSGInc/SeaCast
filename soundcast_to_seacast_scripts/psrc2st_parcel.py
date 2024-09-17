#Convert zones in the parcel file
#Nagendra Dhakar, nagendra.dhakar@rsginc.com, 12/22/16

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

from sqlalchemy import column

# inputs
wd = r"input_files/landuse/2018/v3.0_RTP"
# wd = r"input_files/landuse/2044"
parcel_file = 'parcels_urbansim.txt'
# out_dir = r"output_2044"
out_dir = r"output_files"

# correspondence file
parcel_psrc_taz_file = r"data/psrcprcl_to_sttaz.csv"

def roundcumsum(values_):
    new_values = np.diff(np.insert(np.round(np.cumsum(values_)),0,0))
    return(new_values.astype(int))

def runPSRCtoPSRCZones():
    #read parcel file
    parcel_file_path = os.path.join(wd, parcel_file)
    parcels_psrc = pd.read_csv(parcel_file_path, sep = " ")
    parcels_fields = list(parcels_psrc.columns)

    #read parcel to psrc taz correspondence
    # parcel_psrc_taz_file_path = os.path.join(script_dir, parcel_psrc_taz_file)
    parcel_psrc_taz = pd.read_csv(parcel_psrc_taz_file).rename(columns={'ParcelID':'parcelid', 'STTAZ':'TAZNUM'})

    #merge psrc taz to parcel file
    parcels_psrc = pd.merge(parcels_psrc, parcel_psrc_taz, left_on = 'parcelid', right_on = 'parcelid')
    parcels_psrc['taz_p'] = parcels_psrc['TAZNUM'].astype(np.int32)
    parcels_psrc = parcels_psrc[parcels_fields]
    parcels_psrc = parcels_psrc.sort_values(by = ['parcelid'], ascending=[True])

    # Update the parcel 1302423 to the total employment of 14647 as per email from Mark Simpson PSRC on March 29.
    total_emp = 14647
    emp_fields = [field for field in parcels_fields if 'emp' in field and 'tot' not in field]
    original_emp = parcels_psrc.loc[627564,emp_fields]
    # new_emp = original_emp*total_emp/original_emp.sum()
    # Use Mark's distributin
    new_emp = pd.Series({'empedu_p':0,
                         'empfoo_p':1452,
                         'empgov_p':2542,
                         'empind_p':8234,
                         'empmed_p':7,
                         'empofc_p':651,
                         'empoth_p':75,
                         'empret_p':465,
                         'emprsc_p':0,
                         'empsvc_p':1221,
                         'emptot_p':14647})

    if '2044' not in out_dir:
        parcels_psrc.loc[parcels_psrc.parcelid==1302423,new_emp.index] = new_emp.values

    if len(parcels_psrc) != len(parcels_psrc[~parcels_psrc.taz_p.isna()]):
        print('ERROR: some parcels do not have a psrc taz assigned')
    else:
        #write out the updated parcel file
        parcel_file_out = parcel_file.split(".")[0]+ "_st.txt"
        parcel_file_out_path = os.path.join(out_dir, parcel_file_out)
        parcels_psrc.to_csv(parcel_file_out_path, sep = ' ', index = False,  line_terminator='\n')

if __name__== "__main__":
    print('started ...')
    runPSRCtoPSRCZones()
    print('finished!')

