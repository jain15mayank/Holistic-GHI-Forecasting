import os
import numpy as np
import pandas as pd

METdataCSVfilePath = os.path.join(os.getcwd(), 'data', 'METdata.csv')

gsiVectorsPath = os.path.join(os.getcwd(), 'data', 'gsi_cloud_vectors')
satMasksPath = os.path.join(os.getcwd(), 'data', 'satellite_cloud_masks')

trainYears = [2020, 2021]
testYears = [2022]

finalDFsavePath = os.path.join(os.getcwd(), 'data')

# Function to increase the temporal resolution of a dataframe with "Timestamp"
# column from 5 minutes to 2 minutes
def increaseTemporalResolution5to2(df):
    df2 = df.set_index('Timestamp')
    df2 = df2.resample('2Min').asfreq().ffill(limit=1).bfill(limit=1).reset_index()
    filtered_rows = df2[(df2['Timestamp'].dt.minute % 10).isin([4, 6])]
    filtered_rows['AdjustedMinute'] = filtered_rows["Timestamp"].dt.round("5min")
    merged_df = filtered_rows.merge(df, how='left', left_on=['AdjustedMinute'], right_on=['Timestamp'])
    merged_df = merged_df.rename(columns={'Timestamp_x': 'Timestamp'})
    merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith('_x')]+['AdjustedMinute', 'Timestamp_y'])
    merged_df.set_index('Timestamp', inplace=True)
    df2.set_index('Timestamp', inplace=True)
    for column in df2.columns:
        if column != 'Timestamp':
            df2[column] = df2[column].combine_first(merged_df[column+'_y'])
    # Reset the index of df2
    df2.reset_index(inplace=True)
    df2 = df2.dropna().reset_index(drop=True)
    return df2

def loadGSIvectors(gsiVectorsPath, year):
    loaded_data = np.load(os.path.join(gsiVectorsPath, 'reduced_data_'+str(year)+'.npz'))
    timestamps_numeric = loaded_data['timestamps']
    reduced_data = loaded_data['reduced_data']
    csm_data = loaded_data['csm_data']
    sza_data = loaded_data['sza_data']
    saa_data = loaded_data['saa_data']
    ghi_data = loaded_data['ghi_data']
    columns = ['Timestamp'] + [f'GSIvec{i}' for i in range(1, reduced_data.shape[1]+1)] + ['SZA'] + ['SAA'] + ['CSM'] + ['GHI']
    gsiDF = pd.DataFrame(data=np.hstack([timestamps_numeric.reshape(-1, 1),
                                         reduced_data,
                                         sza_data.reshape(-1, 1),
                                         saa_data.reshape(-1, 1),
                                         csm_data.reshape(-1, 1),
                                         ghi_data.reshape(-1, 1)]), columns=columns)
    gsiDF['Timestamp'] = pd.to_datetime(gsiDF['Timestamp'])
    return gsiDF

def loadSatMasks(satMasksPath, year):
    loaded_data = np.load(os.path.join(satMasksPath, 'encoded_data_'+str(year)+'.npz'))
    timestamps_numeric = loaded_data['timestamps']
    encoded_data = loaded_data['encoded_data']
    columns = ['Timestamp'] + [f'SatMask{i}' for i in range(1, encoded_data.shape[1]+1)]
    satDF = pd.DataFrame(data=np.hstack([timestamps_numeric.reshape(-1, 1), encoded_data]), columns=columns)
    satDF['Timestamp'] = pd.to_datetime(satDF['Timestamp'])
    return satDF

def loadMETdata(METdataCSVfilePath):
    # Define column names
    column_names = [
        "Timestamp",
        "WindSpeed",
        "WindDirection",
        "AirTemperature",
        "RelativeHumidity",
        "Pressure",
        "PrecipitationRate"
    ]
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(METdataCSVfilePath, names=column_names, header=None)
    # Parse the "Timestamp" column as datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y%m%d-%H%M%S')
    df = df.drop(columns=['WindSpeed', 'WindDirection', 'Pressure', 'PrecipitationRate'])
    df = df.dropna().reset_index(drop=True)
    return df

metDF = loadMETdata(METdataCSVfilePath)
print("\n###################################################")
print("###################################################")
print(metDF.head())
print(metDF.describe())
print(metDF.tail())
print(len(metDF))
print("###################################################")
print("###################################################\n")
for year in trainYears+testYears:
    satDF = loadSatMasks(satMasksPath, year)
    satDF = increaseTemporalResolution5to2(satDF)
    gsiDF = loadGSIvectors(gsiVectorsPath, year)
    satNgsiDF = satDF.merge(gsiDF, on='Timestamp', how='inner', validate='one_to_one')
    # print("\n###################################################")
    # print("###################################################")
    # print(satNgsiDF.head())
    # print(satNgsiDF.describe())
    # print(satNgsiDF.tail())
    # print(len(satNgsiDF))
    # print("###################################################")
    # print("###################################################\n")
    del satDF, gsiDF
    finDF = satNgsiDF.merge(metDF, on='Timestamp', how='inner', validate='one_to_one')
    # print("\n###################################################")
    # print("###################################################")
    # print(finDF.head())
    # print(finDF.describe())
    # print(finDF.tail())
    # print(len(finDF))
    # print("###################################################")
    # print("###################################################\n")
    del satNgsiDF
    # finDF.to_csv(os.path.join(finalDFsavePath, f'{year}.csv'), index=False)
    finDF.to_parquet(os.path.join(finalDFsavePath, f'{year}.parquet'), index=False)
    print("Saved data for year =", year)
    # loaded_finDF = pd.read_csv(os.path.join(finalDFsavePath, f'{year}.csv'), parse_dates=["Timestamp"])
    loaded_finDF = pd.read_parquet(os.path.join(finalDFsavePath, f'{year}.parquet'))
    # print("\n###################################################")
    # print("###################################################")
    # print(loaded_finDF.head())
    # print(loaded_finDF.describe())
    # print(loaded_finDF.tail())
    # print(len(loaded_finDF))
    # print("###################################################")
    # print("###################################################\n")
    pd.testing.assert_frame_equal(finDF, loaded_finDF)
    print("Loaded data for", year, "year and confirmed that its valid!")
    del finDF, loaded_finDF