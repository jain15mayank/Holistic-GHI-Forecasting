import os
import sys
sys.path.append('./datasets')
sys.path.append('./models')
sys.path.append('./utils')
from sirtaGSIGHI import normalize_elements, denormalize_elements, makeDatasetGSIfeatureGHI
from sirtaGSIGHImodel import CloudImpactNNtoGHI
from trainFromFeatures import generate_train_val_dates

import torch
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from copy import copy, deepcopy

def plotTrainingCharacteristics(modelCkptPath, plotSavePath):
    ### Create model
    model = CloudImpactNNtoGHI(gridSize=4, spFlag = False)
    ### Load Checkpoint
    checkpoint = torch.load(modelCkptPath, map_location=torch.device('cpu'))
    finEpoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    loss_y = checkpoint['lossHistory']
    val_loss_y = checkpoint['valLossHistory']
    # Plot training loss characteristics for each epoch
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rc('font', size=12)         # controls default text sizes
    plt.rc('axes', titlesize=17)    # fontsize of the axes title
    plt.rc('axes', labelsize=17)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)   # fontsize of the tick labels
    plt.rc('legend', fontsize=15)   # legend fontsize
    plt.rc('figure', titlesize=17)  # fontsize of the figure title
    plt.plot(np.arange(finEpoch)+1, loss_y, label='Training Loss')
    if not None in val_loss_y:
        plt.plot(np.arange(finEpoch)+1, val_loss_y, label='Validation Loss')
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Loss', fontsize=16)
    ax.set_yscale('log')
    plt.title('Training Loss Characteristics')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    ax.tick_params(labelsize=14)
    fig.tight_layout()
    plt.savefig(plotSavePath, bbox_inches = 'tight', pad_inches = 0.05)
    plt.close()

def createImagePathDFforDate(imageDir, date):
    year, month, day = date[:4], date[4:6], date[6:]
    if int(month) <= 6:
        iDir = f'{year}-0106'
    else:
        iDir = f'{year}-0712'
    iDir = os.path.join(imageDir, iDir)
    imagePaths = [os.path.join(iDir, f) for f in sorted(os.listdir(iDir)) if f.endswith('.png') and f.startswith(f'{year}{month}{day}')]
    imageNames = [iP.rsplit('/',1)[-1] for iP in imagePaths]
    imgDF = pd.DataFrame({'imagePath': imagePaths})
    imgDF['Timestamp'] = pd.to_datetime(imageNames, format="%Y%m%dT%H%M%S.png")
    return imgDF

def showImageAndVectors(image_paths, vector_lists, savePath, vecTitles=None):
    gMin = []
    gMax = []
    for i in range(len(vector_lists)):
        assert len(image_paths) == len(vector_lists[i])
        # Determine global min and max values for color limits
        global_min = float('inf')
        global_max = float('-inf')
        for vector in vector_lists[i]:
            global_min = min(global_min, np.min(vector))
            global_max = max(global_max, np.max(vector))
        gMin.append(global_min)
        gMax.append(global_max)

    # Create a figure with subplots
    num_pairs = len(image_paths)
    num_cols = len(vector_lists) + 1
    _, axs = plt.subplots(num_pairs, num_cols, figsize=(5 * num_cols, 5 * num_pairs))

    for i in range(num_pairs):
        image_path = image_paths[i]
        timestamp = datetime.strptime(os.path.splitext(os.path.basename(image_path))[0], "%Y%m%dT%H%M%S")
        # Load the image
        image = plt.imread(image_path)
        # Plot the image on the left-most subplot
        axs[i, 0].imshow(image)
        axs[i, 0].set_title(f'Image at {timestamp}')
        axs[i, 0].axis('off')  # Hide axis for the image
        # Plot the vector heatmaps one by one
        for j in range(num_cols - 1):
            # Reshape the vector into a 4x4 square
            heatmap_data = vector_lists[j][i,:].reshape(4, 4)
            # Create a heatmap of the reshaped vector on the (j+1)th subplot
            cmap = ListedColormap(['blue', 'green', 'yellow', 'red'])  # Define a custom colormap
            heatmap = axs[i, j+1].matshow(heatmap_data, cmap=cmap, vmin=gMin[j], vmax=gMax[j])
            if vecTitles is not None and len(vecTitles) == len(vector_lists):
                axs[i, j+1].set_title(vecTitles[j])
            else:
                axs[i, j+1].set_title('Heatmap')
            plt.colorbar(heatmap, ax=axs[i, j+1])  # Add a colorbar to the heatmap
            axs[i, j+1].axis('off')  # Hide axis for the heatmap
    plt.tight_layout()
    plt.savefig(savePath, bbox_inches = 'tight', pad_inches = 0.05)
    plt.close()

def visualize(savePath, imageDir, featureDir, modelPath, normalizationParams, date, numImages=5, maxAttempts=10):
    completeDataset = makeDatasetGSIfeatureGHI(featureDir, date, date)
    # Extract random indices
    attempts = 0
    while True:
        visIndices = np.sort(np.random.choice(len(completeDataset), numImages, replace=False))
        imgDF = createImagePathDFforDate(imageDir, date)
        dataDF = deepcopy(completeDataset.data)
        dataDF = dataDF.loc[visIndices]
        dataDF = dataDF.reset_index().merge(imgDF, on='Timestamp').set_index('index')
        if len(dataDF) == len(visIndices):
            break
        attempts += 1
        if attempts > maxAttempts:
            raise ValueError('Unable to get raw images for the given date!')
    # Setup model
    device = torch.device('cuda', 0)
    model = CloudImpactNNtoGHI(gridSize=4, spFlag = False)
    model.to(device)
    model.load_state_dict(torch.load(modelPath))
    predictions, targets, cloudVectors, cloudScalars, solarPosVectors = [], [], [], [], []
    # Prepare and transfer all data to device
    dataDF['Proc_CSM'] = dataDF['CSM'].apply(lambda x: normalize_elements(x, normalizationParams['CSM_mean'], normalizationParams['CSM_std']).astype(np.float32))
    dataDF['Proc_GHI'] = dataDF['GHI'].apply(lambda x: normalize_elements(x, normalizationParams['CSM_mean'], normalizationParams['CSM_std']).astype(np.float32))
    dataDF['Proc_SZA'] = dataDF['SZA'].apply(lambda x: np.sin(np.deg2rad(x)).astype(np.float32))
    dataDF['Proc_SAA'] = dataDF['SAA'].apply(lambda x: np.cos(np.deg2rad(x/2)).astype(np.float32))
    csm = torch.tensor(dataDF['Proc_CSM'].values, dtype=torch.float32).unsqueeze(1).to(device)
    sza = torch.tensor(dataDF['Proc_SZA'].values, dtype=torch.float32).unsqueeze(1).to(device)
    saa = torch.tensor(dataDF['Proc_SAA'].values, dtype=torch.float32).unsqueeze(1).to(device)
    target = torch.tensor(dataDF['Proc_GHI'].values, dtype=torch.float32).unsqueeze(1).to(device)
    targets.append(target.cpu().detach().numpy())
    targets = np.concatenate(targets)
    targets = denormalize_elements(targets, normalizationParams['GHI_mean'], normalizationParams['GHI_std'])
    cloudFraction = torch.tensor(np.array([np.array(x) for x in dataDF['CloudFraction']]), dtype=torch.float32).to(device)
    cloudClass = torch.tensor(np.array([np.array(x) for x in dataDF['CloudClass']]), dtype=torch.float32).to(device)
    # Forward pass
    prediction = model.forward(cloudFraction, cloudClass, csm, sza, saa)
    predictions.append(prediction.cpu().detach().numpy())
    predictions = np.concatenate(predictions)
    predictions = denormalize_elements(predictions, normalizationParams['GHI_mean'], normalizationParams['GHI_std'])
    cloudImpactVector, weightString = model.getImpactVector(cloudFraction, cloudClass, csm, sza, saa)
    cloudVectors.append(cloudImpactVector.cpu().detach().numpy())
    cloudVectors = np.concatenate(cloudVectors)
    solarPosVector = model.getSZASAAoutput(sza, saa)
    solarPosVectors.append(solarPosVector.cpu().detach().numpy())
    solarPosVectors = np.concatenate(solarPosVectors)
    cloudImpactScalar = model.getImpactScalar(cloudFraction, cloudClass, csm, sza, saa)
    cloudScalars.append(cloudImpactScalar.cpu().detach().numpy())
    cloudScalars = np.concatenate(cloudScalars)
    cloudScalars = denormalize_elements(cloudScalars, normalizationParams['CSM_mean'], normalizationParams['CSM_std'])
    print(' Targets | Net O/P | CSMvals | cloudSc |   SZA   |   SAA   | ProcSZA | ProcSAA ')
    print('-------------------------------------------------------------------------------')
    [print(f'{t[0]:09.5f}|{p[0]:09.5f}|{c:09.5f}|{cS[0]:09.5f}|{sza:09.5f}|{saa:09.5f}|{psza:09.5f}|{psaa:09.5f}') for t, p, c, cS, sza, saa, psza, psaa in zip(targets, predictions, dataDF['CSM'], cloudScalars, dataDF['SZA'], dataDF['SAA'], dataDF['Proc_SZA'], dataDF['Proc_SAA'])]
    print('\n' + weightString)
    imagePaths = [x for x in dataDF['imagePath']]
    showImageAndVectors(imagePaths, [solarPosVectors, cloudVectors], savePath, vecTitles=['Solar Position Vector', 'Final Cloud Impact Vector'])
    

if __name__=="__main__":
    # modelVersion = '' # V1
    # modelVersion = 'V2' # V2
    # modelVersion = 'V3' # V3
    # modelVersion = 'V4' # V4
    # modelVersion = 'V5' # V5
    # modelVersion = 'V6' # V6
    modelVersion = 'V7' # V7

    modelSavePath = './models/savedWeights/GSIGHI_processedFeatures'+modelVersion+'.pth'
    ckptSavePath = './models/savedWeights/ckpts/GSIGHI_processedFeatures_ckpt'+modelVersion+'.pth'
    featureDir = './datasets/segNclsData/2020to2022/'
    imageDir = '../datasets/SIRTACAM/data/'

    trainStartDate = '20200101'
    trainEndDate = '20210630'

    validationSplit = 0.2

    resultsDir = './results/'
    
    plotTrainingCharacteristics(ckptSavePath, os.path.join(resultsDir, 'lossCharacteristics'+ modelVersion +'.pdf'))

    useValSet = False
    if validationSplit > 0 and validationSplit < 1:
        useValSet = True
        trainStartDate, trainEndDate, valStartDate, valEndDate = generate_train_val_dates(
            trainStartDate, trainEndDate, validationSplit)
    # Find Normalization Parameters used in Training Data
    trainDataset = makeDatasetGSIfeatureGHI(featureDir, trainStartDate, trainEndDate)
    normalizationParams = trainDataset.getNormalizationParams()

    date = '20211001'
    numImages = 5
    visualize(os.path.join(resultsDir, 'vis'+ modelVersion +'.pdf'), imageDir, featureDir,
              modelSavePath, normalizationParams, date, numImages=numImages)