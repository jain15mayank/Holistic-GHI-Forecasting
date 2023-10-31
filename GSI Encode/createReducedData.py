import os
import sys
sys.path.append('./datasets')
sys.path.append('./models')
sys.path.append('./utils')
from sirtaGSIGHI import denormalize_elements, makeDatasetGSIfeatureGHI, generate_date_range
from sirtaGSIGHImodel import CloudImpactNNtoGHI
from trainFromFeatures import generate_train_val_dates

from datetime import datetime
import random
import time
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

'''
DEFINE CONSTANTS FOR EVALUATION
'''

manualSeed = 51
batchSize = 256

modelSavePath = './models/savedWeights/GSIGHI_processedFeatures.pth'
ckptSavePath = './models/savedWeights/ckpts/GSIGHI_processedFeatures_ckpt.pth'
featureDir = './datasets/segNclsData/2020to2022/'

resultsDir = './results/'

# Required to get normalization params
trainStartDate = '20200101'
trainEndDate = '20210630'

createReducedDataForYear = 2022

startDate = str(createReducedDataForYear)+'0101'
endDate = str(createReducedDataForYear)+'1231'

save_data_path = resultsDir+'reducedData/reduced_data_'+str(createReducedDataForYear)+'.npz'

'''
MAIN CODE STARTS HERE
'''

print("Random Seed:\t", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Find Normalization Parameters used in Training Data
trainDataset = makeDatasetGSIfeatureGHI(featureDir, trainStartDate, trainEndDate)
normalizationParams = trainDataset.getNormalizationParams()
del trainDataset

# Make Dataset
yearDataset = makeDatasetGSIfeatureGHI(featureDir, startDate, endDate, normalization_params = normalizationParams)
# Create DataLoader
print("Batch Size:\t", batchSize)
yearDataLoader = DataLoader(yearDataset, batch_size=batchSize, shuffle=False)

### Setup gpu device
device = torch.device('cuda', 0)
### Create model
model = CloudImpactNNtoGHI(gridSize=4, spFlag = False)
### Put the model to the device
model.to(device)
### Print summary of the model
# summary(model, [(batchSize,4*4), (batchSize,4*4), (batchSize,1), (batchSize,1), (batchSize,1)])
### Load the saved model
model.load_state_dict(torch.load(modelSavePath))
### Set model to evaluation mode
model.eval()


date_list = generate_date_range(startDate, endDate)
# Create a generator expression to yield DataFrames
dfs_generator = (
    pd.DataFrame({
        'Timestamp': data['dt'].tolist(),
        'CloudFraction': data['cf'].tolist(),
        'CloudClass': data['sky_class'].tolist(),
        'SZA': data['sza'].tolist(),
        'SAA': data['saa'].tolist(),
        'CSM': data['csm'].tolist(),
        'GHI': data['ghi'].tolist()
    })
    .assign(Timestamp=lambda x: pd.to_datetime(x['Timestamp'], format='%Y%m%d%H%M%S'))  # Convert 'dt' to datetime
    for date in date_list
    if os.path.exists(os.path.join(featureDir, date+'.npy'))
    for data in [np.load(os.path.join(featureDir, date+'.npy'), allow_pickle=True)]
)
data = pd.concat(dfs_generator, ignore_index=True)
data = data.sort_values(by='Timestamp')
# Remove rows with NaN values - It seems that 29 March 2021 has NaN values in GHI Column
data.dropna(inplace=True)
data = data.reset_index(drop=True)

# Check if there are no NaN values in any row
assert not data.isnull().any(axis=1).any(), "There are rows with NaN values"

# Convert 'SZA', 'SAA', 'CSM', 'GHI' columns to numeric
numeric_cols = ['SZA', 'SAA', 'CSM', 'GHI']
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

# print(data.describe())


### Obtain reducedData
reducedData, csmData, szaData, saaData, ghiData = [], [], [], [], []
start_eval = time.time()
progress_bar = tqdm(enumerate(yearDataLoader, 0), desc=f'Year Predictions', ncols=100, unit="batch", total=len(yearDataLoader))
for i,batch in progress_bar:
    # Transfer batch data to device
    cloudFraction = batch['cloudFraction'].float().to(device)
    cloudClass = batch['cloudClass'].float().to(device)
    csm = batch['csm'].float().to(device)
    sza = batch['sza'].float().to(device)
    saa = batch['saa'].float().to(device)
    target = batch['target'].float().to(device)
    # Forward pass
    impactVector, _ = model.getImpactVector(cloudFraction, cloudClass, csm, sza, saa)
    # Add data to lists
    csmData.append(csm.cpu().detach().numpy())
    szaData.append(sza.cpu().detach().numpy())
    saaData.append(saa.cpu().detach().numpy())
    ghiData.append(target.cpu().detach().numpy())
    reducedData.append(impactVector.cpu().detach().numpy())

reducedData = np.concatenate(reducedData)
print('Reduced Data:', reducedData.shape)
csmData = np.concatenate(csmData)
csmData = denormalize_elements(csmData, normalizationParams['CSM_mean'], normalizationParams['CSM_std'])
print('CSM:', csmData.shape, np.min(csmData), np.max(csmData))
szaData = np.concatenate(szaData)
szaData = np.rad2deg(np.arcsin(szaData))
print('SZA:', szaData.shape, np.min(szaData), np.max(szaData))
saaData = np.concatenate(saaData)
saaData = 2*np.rad2deg(np.arccos(saaData))
print('SAA:', saaData.shape, np.min(saaData), np.max(saaData))
ghiData = np.concatenate(ghiData)
ghiData = denormalize_elements(ghiData, normalizationParams['GHI_mean'], normalizationParams['GHI_std'])
print('GHI:', ghiData.shape, np.min(ghiData), np.max(ghiData))

timestamps_list = data['Timestamp'].tolist()
print('Timestamp:', len(timestamps_list), timestamps_list[0], timestamps_list[-1])


# Save the data in .npz format
timestamps_numeric = pd.to_datetime(timestamps_list).view('int64')
np.savez_compressed(save_data_path, timestamps=timestamps_numeric, reduced_data=reducedData,
                    csm_data=csmData, sza_data=szaData, saa_data=saaData, ghi_data=ghiData)

# Load the data from the .npz file
loaded_data = np.load(save_data_path)
loaded_timestamps_numeric = loaded_data['timestamps']
loaded_timestamps = pd.to_datetime(loaded_timestamps_numeric)
loaded_reduced_data = loaded_data['reduced_data']
loaded_csm_data = loaded_data['csm_data']
loaded_sza_data = loaded_data['sza_data']
loaded_saa_data = loaded_data['saa_data']
loaded_ghi_data = loaded_data['ghi_data']

assert loaded_timestamps.equals(pd.to_datetime(timestamps_list))
assert np.array_equal(loaded_reduced_data, reducedData)
assert np.array_equal(loaded_csm_data, csmData)
assert np.array_equal(loaded_sza_data, szaData)
assert np.array_equal(loaded_saa_data, saaData)
assert np.array_equal(loaded_ghi_data, ghiData)
print('Loaded data is an exact match to the data that was saved!')