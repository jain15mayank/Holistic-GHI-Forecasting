import os
import sys
sys.path.append('./datasets')
sys.path.append('./models')
sys.path.append('./utils')
from sirtaGSIGHI import denormalize_elements, makeDatasetGSIfeatureGHI
from sirtaGSIGHImodel import CloudImpactNNtoGHI
from trainFromFeatures import generate_train_val_dates

from datetime import datetime
import random
import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
from torch.utils.data import DataLoader

from torchinfo import summary
from tqdm import tqdm

'''
DEFINE CONSTANTS FOR EVALUATION
'''

manualSeed = 51
batchSize = 256

# modelVersion = '' # V1
modelVersion = 'V2' # V2
# modelVersion = 'V3' # V3
# modelVersion = 'V4' # V4
# modelVersion = 'V5' # V5
# modelVersion = 'V6' # V6
# modelVersion = 'V7' # V7

modelSavePath = './models/savedWeights/GSIGHI_processedFeatures'+modelVersion+'.pth'
ckptSavePath = './models/savedWeights/ckpts/GSIGHI_processedFeatures_ckpt'+modelVersion+'.pth'
featureDir = './datasets/segNclsData/2020to2022/'

resultsDir = './results/'

trainStartDate = '20200101'
trainEndDate = '20210630'
testStartDate = '20210701'
testEndDate = '20211231'

validationSplit = 0.2

plotDate = '20211014' # Date to plot the prediction curve for


'''
MAIN CODE STARTS HERE
'''

print("Random Seed:\t", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

useValSet = False
if validationSplit > 0 and validationSplit < 1:
    useValSet = True
    trainStartDate, trainEndDate, valStartDate, valEndDate = generate_train_val_dates(
        trainStartDate, trainEndDate, validationSplit)
# Find Normalization Parameters used in Training Data
trainDataset = makeDatasetGSIfeatureGHI(featureDir, trainStartDate, trainEndDate)
normalizationParams = trainDataset.getNormalizationParams()

# Make Dataset
testDataset = makeDatasetGSIfeatureGHI(featureDir, testStartDate, testEndDate, normalization_params = normalizationParams)
# Create DataLoader
print("Batch Size:\t", batchSize)
testDataLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)
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
# checkpoint = torch.load(ckptSavePath)
# finEpoch = checkpoint['epoch']
# model.load_state_dict(checkpoint['model_state_dict'])
### Set model to evaluation mode
model.eval()
### Obtain predictions
predictions, targets = [], []
start_eval = time.time()
progress_bar = tqdm(enumerate(testDataLoader, 0), desc=f'Test Predictions', ncols=100, unit="batch", total=len(testDataLoader))
for i,batch in progress_bar:
    # Transfer batch data to device
    cloudFraction = batch['cloudFraction'].float().to(device)
    cloudClass = batch['cloudClass'].float().to(device)
    csm = batch['csm'].float().to(device)
    sza = batch['sza'].float().to(device)
    saa = batch['saa'].float().to(device)
    target = batch['target'].float().to(device)
    targets.append(target.cpu().detach().numpy())
    # Forward pass
    prediction = model.forward(cloudFraction, cloudClass, csm, sza, saa)
    predictions.append(prediction.cpu().detach().numpy())
targets = np.concatenate(targets)
targets = denormalize_elements(targets, normalizationParams['GHI_mean'], normalizationParams['GHI_std'])
predictions = np.concatenate(predictions)
predictions = denormalize_elements(predictions, normalizationParams['GHI_mean'], normalizationParams['GHI_std'])
end_eval = time.time()
elapsed = end_eval - start_eval
print("Time Consumed in Making Predictions:\t", elapsed)
print(len(testDataset), len(predictions), len(targets))
assert len(testDataset)==len(predictions), "Number of predictions does not match with the number of elements in test dataset!"
### Evaluate the Metrics on the entire test set
rmse = np.sqrt(mean_squared_error(targets, predictions))
mae = mean_absolute_error(targets, predictions)
r2 = r2_score(targets, predictions)
print(f"RMSE:\t\t {rmse}")
print(f"MAE:\t\t {mae}")
print(f"R2 Score:\t {r2}")
### Prepare to plot for plot_date
timestamp_list, csm_list, ghi_list, idx_list = testDataset.getDataByDate(plotDate)
predictions_list = [predictions[idx] for idx in idx_list]
# Plot predictions and ground truth against 'Timestamp' values
fig, ax = plt.subplots(figsize=(10, 6))
plt.rc('font', size=12)         # controls default text sizes
plt.rc('axes', titlesize=17)    # fontsize of the axes title
plt.rc('axes', labelsize=17)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)   # fontsize of the tick labels
plt.rc('ytick', labelsize=14)   # fontsize of the tick labels
plt.rc('legend', fontsize=15)   # legend fontsize
plt.rc('figure', titlesize=17)  # fontsize of the figure title
plt.plot(timestamp_list, ghi_list, label='Ground Truth', marker='x')
plt.plot(timestamp_list, csm_list, label='Clear Sky Model', marker='1')
plt.plot(timestamp_list, predictions_list, label='Predictions', marker='o')
ax.set_xlabel('Timestamp', fontsize=16)
ax.set_ylabel('Values', fontsize=16)
plt.title('GHI Predictions vs. Ground Truth')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
ax.tick_params(labelsize=14)
fig.tight_layout()
plt.savefig(os.path.join(resultsDir, plotDate+modelVersion+'.pdf'), bbox_inches = 'tight', pad_inches = 0.05)
plt.close()
