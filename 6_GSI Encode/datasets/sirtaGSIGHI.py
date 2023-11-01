import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from datetime import datetime, timedelta
import sys
#sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'utils'))
sys.path.append('../utils')
from identifyClearSkyModel import addBestCSMvaluesToDataframe
#from ..utils.identifyClearSkyModel import addBestCSMvaluesToDataframe

# Define a function to create data range between start and end dates ("YYYYMMDD")
def generate_date_range(start_date_str, end_date_str):
    # Convert start_date and end_date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")
    # Initialize an empty list to store the date range
    date_range = []
    # Iterate through the date range and add each date to the list
    current_date = start_date
    while current_date <= end_date:
        date_range.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)
    return date_range

# Define the Min-Max scaling function
def min_max_scale(tensor, min_val, max_val):
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def inverse_min_max_scale(normalized_tensor, min_val, max_val):
    original_tensor = normalized_tensor * (max_val - min_val) + min_val
    return original_tensor

# Define the Standard normalization function
def normalize_elements(tensor, mean, std):
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor

def denormalize_elements(normalized_value, mean, std):
    denormalized_value = (normalized_value * std) + mean
    return denormalized_value


class makeDatasetGSIGHI(Dataset):
    def __init__(self, csv_file, image_dir, start_date, end_date, latitude, longitude, elevation, img_transform=None, normalization_params = None, scaling_params = None):
        self.data = pd.read_csv(csv_file, names=['Timestamp', 'GHI'])
        self.data = self.data.dropna()
        self.data.GHI[self.data.GHI.lt(0)] = 0
        self.data = self.data.reset_index(drop = True)
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'], format="%Y%m%d-%H%M%S")
        self.start_date = datetime.strptime(start_date, '%Y%m%d')
        self.end_date = datetime.strptime(end_date, '%Y%m%d')
        self.transform = img_transform

        # Filter data based on date range
        self.data = self.data[(self.data['Timestamp'] >= self.start_date) & (self.data['Timestamp'] <= self.end_date)]
        self.data = self.data.reset_index(drop = True)

        # Add CSM, SZA and SAA columns to the DataFrame
        #start_time = time.time() # import time
        self.data = addBestCSMvaluesToDataframe(self.data, latitude, longitude, elevation)
        #print("\nTime to add CSM, SZA, SAA = --- %s seconds ---\n" % (time.time() - start_time))

        # Add imagePath to self.data on common timestamps - remove rest of the rows
        imagePaths = []
        for iDir in sorted(os.listdir(image_dir)):
            imagePaths += list(map(lambda x: image_dir+iDir+"/" + x, sorted(os.listdir(image_dir+iDir))))
        imageNames = [iP.rsplit('/',1)[-1] for iP in imagePaths]
        imageDataCT = pd.DataFrame({'imagePath': imagePaths})
        imageDataCT['Timestamp'] = pd.to_datetime(imageNames, format="%Y%m%dT%H%M%S.png")
        self.data = self.data.merge(imageDataCT, on='Timestamp')
        self.data = self.data.reset_index(drop=True)

        # Convert 'SZA', 'SAA', 'CSM', 'GHI' columns to numeric
        numeric_cols = ['SZA', 'SAA', 'CSM', 'GHI']
        self.data[numeric_cols] = self.data[numeric_cols].apply(pd.to_numeric, errors='coerce')

        if normalization_params is None:
            # Calculate mean and standard deviation for normalization
            self.csm_mean, self.csm_std = np.mean(self.data['CSM']), np.std(self.data['CSM'])
            self.ghi_mean, self.ghi_std = np.mean(self.data['CSM']), np.std(self.data['CSM'])
        else:
            self.csm_mean, self.csm_std = normalization_params['CSM_mean'], normalization_params['CSM_std']
            self.ghi_mean, self.ghi_std = normalization_params['GHI_mean'], normalization_params['GHI_std']
        
        if scaling_params is None:
            # Calculate min and max for min-max scaling
            self.csm_min, self.csm_max = np.min(self.data['CSM']), np.max(self.data['CSM'])
            self.ghi_min, self.ghi_max = np.min(self.data['CSM']), np.max(self.data['CSM'])
        else:
            self.csm_min, self.csm_max = scaling_params['CSM_min'], scaling_params['CSM_max']
            self.ghi_min, self.ghi_max = scaling_params['GHI_min'], scaling_params['GHI_max']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.loc[idx, 'imagePath']
        image = Image.open(image_path)

        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)
        
        # Extract other features - csm, sza and saa
        CSM = normalize_elements(self.data.loc[idx, 'CSM'].reshape(-1), self.csm_mean, self.csm_std).astype(np.float32)
        SZA = np.sin(np.deg2rad(self.data.loc[idx, 'SZA'])).reshape(-1).astype(np.float32)
        SAA = np.cos(np.deg2rad(self.data.loc[idx, 'SAA']/2)).reshape(-1).astype(np.float32)
        features = np.array([SZA, SAA, CSM])

        # Extract the target variable (GHI)
        target = normalize_elements(self.data.loc[idx, 'GHI'].reshape(-1), self.csm_mean, self.csm_std).astype(np.float32)

        return {'image': image, 'features': features, 'target': target}
    
    def getNormalizationParams(self):
        return {'CSM_mean': self.csm_mean, 'CSM_std': self.csm_std, 'GHI_mean': self.ghi_mean, 'GHI_std': self.ghi_std}
    
    def getScalingParams(self):
        return {'CSM_min': self.csm_min, 'CSM_max': self.csm_max, 'GHI_min': self.ghi_min, 'GHI_max': self.ghi_max}

class makeDatasetGSIfeatureGHI(Dataset):
    def __init__(self, feature_dir, start_date, end_date, normalization_params = None, scaling_params = None):
        date_list = generate_date_range(start_date, end_date)
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
            if os.path.exists(os.path.join(feature_dir, date+'.npy'))
            for data in [np.load(os.path.join(feature_dir, date+'.npy'), allow_pickle=True)]
        )
        self.data = pd.concat(dfs_generator, ignore_index=True)
        self.data = self.data.sort_values(by='Timestamp')
        # Remove rows with NaN values - It seems that 29 March 2021 has NaN values in GHI Column
        self.data.dropna(inplace=True)
        self.data = self.data.reset_index(drop=True)

        # Check if there are no NaN values in any row
        assert not self.data.isnull().any(axis=1).any(), "There are rows with NaN values"

        # Convert 'SZA', 'SAA', 'CSM', 'GHI' columns to numeric
        numeric_cols = ['SZA', 'SAA', 'CSM', 'GHI']
        self.data[numeric_cols] = self.data[numeric_cols].apply(pd.to_numeric, errors='coerce')

        if normalization_params is None:
            # Calculate mean and standard deviation for normalization
            self.csm_mean, self.csm_std = np.mean(self.data['CSM']), np.std(self.data['CSM'])
            self.ghi_mean, self.ghi_std = np.mean(self.data['CSM']), np.std(self.data['CSM'])
        else:
            self.csm_mean, self.csm_std = normalization_params['CSM_mean'], normalization_params['CSM_std']
            self.ghi_mean, self.ghi_std = normalization_params['GHI_mean'], normalization_params['GHI_std']
        
        if scaling_params is None:
            # Calculate min and max for min-max scaling
            self.csm_min, self.csm_max = np.min(self.data['CSM']), np.max(self.data['CSM'])
            self.ghi_min, self.ghi_max = np.min(self.data['CSM']), np.max(self.data['CSM'])
        else:
            self.csm_min, self.csm_max = scaling_params['CSM_min'], scaling_params['CSM_max']
            self.ghi_min, self.ghi_max = scaling_params['GHI_min'], scaling_params['GHI_max']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract the cloudFraction
        cloudFraction = self.data.loc[idx, 'CloudFraction'].astype(np.float32)
        # Extract the cloudClass
        cloudClass = self.data.loc[idx, 'CloudClass'].astype(np.float32)
        # Extract the csm, sza and saa
        CSM = normalize_elements(self.data.loc[idx, 'CSM'].reshape(-1), self.csm_mean, self.csm_std).astype(np.float32)
        SZA = np.sin(np.deg2rad(self.data.loc[idx, 'SZA'])).reshape(-1).astype(np.float32)
        SAA = np.cos(np.deg2rad(self.data.loc[idx, 'SAA']/2)).reshape(-1).astype(np.float32)

        # Extract the target variable (GHI)
        target = normalize_elements(self.data.loc[idx, 'GHI'].reshape(-1), self.csm_mean, self.csm_std).astype(np.float32)

        return {'cloudFraction': cloudFraction, 'cloudClass': cloudClass, 'csm': CSM, 'sza': SZA, 'saa': SAA, 'target': target}
    
    def getNormalizationParams(self):
        return {'CSM_mean': self.csm_mean, 'CSM_std': self.csm_std, 'GHI_mean': self.ghi_mean, 'GHI_std': self.ghi_std}
    
    def getScalingParams(self):
        return {'CSM_min': self.csm_min, 'CSM_max': self.csm_max, 'GHI_min': self.ghi_min, 'GHI_max': self.ghi_max}
    
    def getDataByDate(self, date):
        """
        Get Timestamp, CSM, GHI values, and their row index corresponding to the given date.
        Args:
            date (str): Date in "YYYYMMDD" format.
        Returns:
            4 lists : 4 Lists of variables - (Timestamp, CSM, GHI, row index) for the given date.
        """
        date_df = self.data[self.data['Timestamp'].dt.strftime('%Y%m%d') == date]
        timestamp_list, csm_list, ghi_list, idx_list = [], [], [], []
        for idx, row in date_df.iterrows():
            timestamp_list.append(row['Timestamp'])
            csm_list.append(row['CSM'])
            ghi_list.append(row['GHI'])
            idx_list.append(idx)
        return (timestamp_list, csm_list, ghi_list, idx_list)


if __name__ == "__main__":
    imageDir = "../../datasets/SIRTACAM/data/"
    ghiDataFile = "./GHIdata2020-22.csv"
    featureDir = "./segNclsData/2020to2022/"

    startDate = '20200101'  #YYYYMMDD
    endDate = '20211231'  #YYYYMMDD

    # latitude = 48.7
    # longitude = 2.2
    # elevation = 176

    # startDate = datetime.strptime(startDate, '%Y%m%d')
    # endDate = datetime.strptime(endDate, '%Y%m%d')

    # transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    # ])
    print('Starting to Initialize Dataset Object!')
    # datasetObj = makeDatasetGSIGHI(ghiDataFile, imageDir, startDate, endDate, latitude, longitude, elevation, img_transform=transform)
    datasetObj = makeDatasetGSIfeatureGHI(featureDir, startDate, endDate)
    print(datasetObj.__len__())
    print(datasetObj.__getitem__(0))