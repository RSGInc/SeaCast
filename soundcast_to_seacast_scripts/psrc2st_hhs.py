
#Convert PSRC matrices to PSRC matrices
#Ben Stabler, ben.stabler@rsginc.com, 08/29/16
#updated for 2014 popsyn file

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
wd = r"input_files/landuse/2044"
# wd = r"input_files/landuse/2018/v3.0_RTP"
popsynFileName = "hh_and_persons.h5"
xwalkFile = r"data/psrcprcl_to_sttaz.csv"
out_dir = r"output_2044"
# out_dir = r"output_files"

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


def readSynPopTables(fileName):
    print('read synpop file')
    popsyn = h5py.File(fileName)
    hhFields = map(lambda x: x[0], popsyn.get("Household").items())
    perFields = map(lambda x: x[0], popsyn.get("Person").items())    
    #build pandas data frames
    #hhFields.remove('incomeconverted') #not a column attribute
    hhTable = pd.DataFrame()
    for hhField in hhFields:
        hhTable[hhField] = popsyn.get("Household").get(hhField)[:]
    perTable = pd.DataFrame()
    for perField in perFields:
        perTable[perField] = popsyn.get("Person").get(perField)[:]
    return(hhTable, perTable)

def writeSynPopTables(fileName, households, persons):
    print('write synpop file')
    #delete columns first and then write
    popsyn = h5py.File(fileName, "a")
    for hhField in households.columns:
        dataset = "Household/" + hhField
        print(dataset)
        del popsyn[dataset]
        popsyn.create_dataset(dataset, data = households[hhField],compression="gzip")
    for perField in persons.columns:
        dataset = "Person/" + perField
        print(dataset)
        del popsyn[dataset]
        popsyn.create_dataset(dataset, data = persons[perField], compression="gzip")
    popsyn.close()
    
def runSynPopPSRCtoPSRCZones():

    #read popsyn file
    popsynFile = os.path.join(wd, popsynFileName)
    households, persons = readSynPopTables(popsynFile)

    # get cross walk between parcel and taz
    parcels = pd.read_csv(xwalkFile).rename(columns={'ParcelID': 'parcelid', 'STTAZ':'taz_p'})
    parcels = parcels.set_index('parcelid')

    #merge to households
    print('assign seatac county tazs')
    hh_columns = households.columns
    households['hhtaz'] = reindex(parcels.taz_p, households.hhparcel).astype(np.int32)
    remaining_households = households[households.hhtaz.isna()]
    print('Remaining households {}'.format(remaining_households.shape[0]))
    
    households = households.sort_values("hhno")
    households = households[hh_columns]

    #write result file by copying input file and writing over arrays
    popsynOutFileName = popsynFileName.split(".")[0]+ "_st.h5"
    popsynOutFileName = os.path.join(out_dir, popsynOutFileName)
    shutil.copy2(popsynFile, popsynOutFileName)
    writeSynPopTables(popsynOutFileName, households, persons)

if __name__== "__main__":
    runSynPopPSRCtoPSRCZones()
