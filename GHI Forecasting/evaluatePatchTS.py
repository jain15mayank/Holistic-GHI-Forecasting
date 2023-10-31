import os
import sys
sys.path.append('./datasets')
sys.path.append('./models')
from TSforecastingDatasets import PatchTSDataset, denormalize_elements
from TSforecastingModels import PatchTransformer
from trainPatchTS import generate_train_val_dates

from datetime import datetime, timedelta
import numpy as np
import random
import time
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

dataDir = './datasets/data/'
testStartDate = '20220101'
testEndDate = '20221231'

trainStartDate = '20200101'
trainEndDate = '20211231'
validationSplit = 0.2

modelSavePath = './models/savedWeights/GHIforecastingPatchTS.pth'

plotDate = '20221001' # Date to plot the prediction curve for

print("Random Seed:\t", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

useValSet = False
trainStartDate = trainStartDate
trainEndDate = trainEndDate
if validationSplit > 0 and validationSplit < 1:
    useValSet = True
    trainStartDate, trainEndDate, valStartDate, valEndDate = generate_train_val_dates(
        trainStartDate, trainEndDate, validationSplit)
    print(trainStartDate, trainEndDate, valStartDate, valEndDate)

# Make Dataset
trainDataset = PatchTSDataset(dataDir, trainStartDate, trainEndDate,
                              contWindowsFilePath=os.path.join(os.getcwd(), 'datasets', 'data', 'contWindows.pkl'))
normalizationParams = trainDataset.getNormalizationParams()

# Make Dataset
testDataset = PatchTSDataset(dataDir, testStartDate, testEndDate,
                             normalization_params = normalizationParams,
                             contWindowsFilePath=os.path.join(os.getcwd(), 'datasets', 'data', 'contWindows2022.pkl'))
# Create DataLoader
testDataLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)

### Setup gpu device
device = torch.device('cuda', 0)
### Create model
history_shape = testDataset.__getitem__(0)[0].shape
target_shape = testDataset.__getitem__(0)[1].shape
print(history_shape, target_shape)
input_features = history_shape[1]
output_steps = target_shape[0]
model = PatchTransformer(in_features=input_features, context_window=history_shape[0],
                         d_model=128, nhead=16, num_layers=3, target_window=output_steps)
### Put the model to the device
model.to(device)
### Print summary of the model
summary(model, [(batchSize, history_shape[0], history_shape[1])])
### Load the saved model
model.load_state_dict(torch.load(modelSavePath))#['model_state_dict'])
### Set model to evaluation mode
model.eval()
### Obtain predictions
predictions, targets = [], []
start_eval = time.time()
progress_bar = tqdm(enumerate(testDataLoader, 0), desc=f'Test Predictions', ncols=100, unit="batch", total=len(testDataLoader))
for i,batch in progress_bar:
    # Transfer batch data to device
    history = batch[0].float().to(device)
    target = batch[1].float().to(device)
    # Forward pass
    outputs = model(history)
    predictions.append(outputs[:,:,-1].cpu().detach().numpy())
    targets.append(target[:,:,-1].cpu().detach().numpy())
print(len(targets), len(predictions), targets[0].shape, predictions[0].shape)
targets = np.concatenate(targets)
targets = denormalize_elements(targets, normalizationParams['GHI_mean'], normalizationParams['GHI_std'])
predictions = np.concatenate(predictions)
predictions = denormalize_elements(predictions, normalizationParams['GHI_mean'], normalizationParams['GHI_std'])
print(targets.shape, predictions.shape)
end_eval = time.time()
elapsed = end_eval - start_eval
print("Time Consumed in Making Predictions:\t", elapsed)
print(len(testDataset), len(predictions), len(targets))
assert len(testDataset)==len(predictions), "Number of predictions does not match with the number of elements in test dataset!"
### Evaluate the Metrics on the entire test set
print('Metrics for complete 1 hour ahead forecast at 2 minute temporal resolution:')
rmse = np.sqrt(mean_squared_error(targets, predictions))
mae = mean_absolute_error(targets, predictions)
r2 = r2_score(targets, predictions)
print(f"RMSE:\t\t {rmse}")
print(f"MAE:\t\t {mae}")
print(f"R2 Score:\t {r2}")
for i in range(6):
    print(f'Metrics for single value after {(i+1)*10} minutes:')
    rmse = np.sqrt(mean_squared_error(targets[:,((i+1)*5)-1], predictions[:,((i+1)*5)-1]))
    mae = mean_absolute_error(targets[:,((i+1)*5)-1], predictions[:,((i+1)*5)-1])
    r2 = r2_score(targets[:,((i+1)*5)-1], predictions[:,((i+1)*5)-1])
    print(f"RMSE ({(i+1)*10}-min):\t\t {rmse}")
    print(f"MAE ({(i+1)*10}-min):\t\t {mae}")
    print(f"R2 Score ({(i+1)*10}-min):\t {r2}")

### Prepare to plot for plot_date