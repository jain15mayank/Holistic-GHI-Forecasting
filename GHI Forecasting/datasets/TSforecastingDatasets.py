import os
import time
import pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# Define the Standard normalization function
def normalize_elements(tensor, mean, std):
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor
# Define the function to undo Standard normalization
def denormalize_elements(normalized_value, mean, std):
    denormalized_value = (normalized_value * std) + mean
    return denormalized_value

def createRollingWindowIndices(max_steps, history_steps, future_steps, rolling_steps=1):
    indices = [(list(range(i, i + history_steps)), list(range(i + history_steps, i + history_steps + future_steps)))
               for i in range(0, max_steps - history_steps - future_steps + 1, rolling_steps)]
    return indices

def checkContinuousWindow(df, window_indices, temporal_resolution='2min'):
    time_diff = pd.to_timedelta(temporal_resolution)
    timestamps = df.iloc[window_indices]['Timestamp'].reset_index(drop=True)
    timestamps = pd.to_datetime(timestamps)
    time_diffs = timestamps.diff().dropna()
    return (time_diffs <= time_diff).all()

def createContinuousRollingWindows(df, history_steps, future_steps, max_steps=None,
                                   rolling_steps=1, temporal_resolution='2min'):
    max_steps = len(df) if max_steps is None else max_steps
    windowIndices = createRollingWindowIndices(max_steps, history_steps, future_steps, rolling_steps=rolling_steps)
    condition = lambda x: checkContinuousWindow(df, x[0]+x[1], temporal_resolution=temporal_resolution)
    contWindowIndices = [window[0][0] for window in windowIndices if condition(window)]
    return contWindowIndices

class TimeSeriesDataset(Dataset):
    def __init__(self, data_path, startDate, endDate, history_len=6 * 30, forecast_len=1 * 30,
                 normalization_params=None, contWindowsFilePath=None):
        self.start_date = datetime.strptime(startDate, '%Y%m%d')
        self.end_date = datetime.strptime(endDate, '%Y%m%d')
        years = np.unique([self.start_date.year, self.end_date.year])
        self.data = None
        # Read data from the data_path for relevant years
        for year in years:
            if self.data is None:
                self.data = pd.read_parquet(os.path.join(data_path, f'{year}.parquet'))
            else:
                self.data = pd.concat([self.data, pd.read_parquet(os.path.join(data_path, f'{year}.parquet'))])
        # Filter data based on date range
        self.data = self.data[(self.data['Timestamp'] >= self.start_date) & 
                              (self.data['Timestamp'] <= self.end_date + timedelta(hours=23, minutes=59))]
        # Place GHI in the end
        self.data = self.data[[col for col in self.data.columns if col != 'GHI'] + ['GHI']]
        self.data = self.data.reset_index(drop = True)

        # Perform Normalization
        if normalization_params is None:
            # Calculate mean and standard deviation for normalization
            self.satMask_means = [np.mean(self.data[col]) for col in self.data.columns if col.startswith('SatMask')]
            self.satMask_stds = [np.std(self.data[col]) for col in self.data.columns if col.startswith('SatMask')]
            self.csm_mean, self.csm_std = np.mean(self.data['CSM']), np.std(self.data['CSM'])
            self.ghi_mean, self.ghi_std = np.mean(self.data['GHI']), np.std(self.data['GHI'])
            self.airTemp_mean, self.airTemp_std = np.mean(self.data['AirTemperature']), np.std(self.data['AirTemperature'])
            self.relHum_mean, self.relHum_std = np.mean(self.data['RelativeHumidity']), np.std(self.data['RelativeHumidity'])
        else:
            self.satMask_means, self.satMask_stds = normalization_params['SatMask_means'], normalization_params['SatMask_stds']
            self.csm_mean, self.csm_std = normalization_params['CSM_mean'], normalization_params['CSM_std']
            self.ghi_mean, self.ghi_std = normalization_params['GHI_mean'], normalization_params['GHI_std']
            self.airTemp_mean, self.airTemp_std = normalization_params['AirTemp_mean'], normalization_params['AirTemp_std']
            self.relHum_mean, self.relHum_std = normalization_params['RelHum_mean'], normalization_params['RelHum_std']
        
        self.history_len = history_len
        self.forecast_len = forecast_len

        # start_time = time.time()
        if contWindowsFilePath is None:
            self.contWindows = createContinuousRollingWindows(self.data, self.history_len, self.forecast_len)
        else:
            if os.path.exists(contWindowsFilePath) and os.path.isfile(contWindowsFilePath):
                with open(contWindowsFilePath, 'rb') as file:
                    loaded_data = pickle.load(file)
                if loaded_data['startDate'] == self.start_date and loaded_data['endDate'] == self.end_date:
                    print('Loaded Continuous Windows From File!')
                    self.contWindows = loaded_data['contWindows']
                else:
                    self.contWindows = createContinuousRollingWindows(self.data, self.history_len, self.forecast_len)
                    with open(contWindowsFilePath, 'wb') as file:
                        pickle.dump({'contWindows': self.contWindows, 'startDate': self.start_date, 'endDate': self.end_date}, file)
            else:
                self.contWindows = createContinuousRollingWindows(self.data, self.history_len, self.forecast_len)
                with open(contWindowsFilePath, 'wb') as file:
                    pickle.dump({'contWindows': self.contWindows, 'startDate': self.start_date, 'endDate': self.end_date}, file)

        # print(f"Elapsed time: {time.time()-start_time} seconds")

        # print(self.data.head())
        # print(self.data.describe())
        # print(self.data.tail())
        # print(len(self.data))
        # print(len(self.contWindows))

    def __len__(self):
        return len(self.contWindows)

    def __getitem__(self, idx):
        history_start = self.contWindows[idx]
        history_end = history_start + self.history_len
        forecast_start = history_end
        forecast_end = forecast_start + self.forecast_len

        history = self.data.iloc[history_start:history_end, 1:]  # Exclude timestamp
        target = self.data.iloc[forecast_start:forecast_end, -1]  # GHI

        for col in history.columns:
            if col.startswith('SatMask'):
                history[col] = normalize_elements(history[col], self.satMask_means[int(col.split('SatMask')[1])-1],
                                                  self.satMask_stds[int(col.split('SatMask')[1])-1])
        history['CSM'] = normalize_elements(history['CSM'], self.csm_mean, self.csm_std)
        history['GHI'] = normalize_elements(history['GHI'], self.ghi_mean, self.ghi_std)
        history['AirTemperature'] = normalize_elements(history['AirTemperature'], self.airTemp_mean, self.airTemp_mean)
        history['RelativeHumidity'] = normalize_elements(history['RelativeHumidity'], self.relHum_mean, self.relHum_std)
        history['SZA'] = np.sin(np.deg2rad(history['SZA']))
        history['SAA'] = np.cos(np.deg2rad(history['SAA']/2))

        target = normalize_elements(target, self.ghi_mean, self.ghi_std)

        return history.values.astype('float32'), target.values.astype('float32')
    
    def getNormalizationParams(self):
        return {'SatMask_means': self.satMask_means, 'SatMask_stds': self.satMask_stds,
                'CSM_mean': self.csm_mean, 'CSM_std': self.csm_std, 'GHI_mean': self.ghi_mean, 'GHI_std': self.ghi_std,
                'AirTemp_mean': self.airTemp_mean, 'AirTemp_std': self.airTemp_std,
                'RelHum_mean': self.relHum_mean, 'RelHum_std': self.relHum_std}

class PatchTSDataset(Dataset):
    def __init__(self, data_path, startDate, endDate, history_len=6 * 30, forecast_len=1 * 30,
                 normalization_params=None, contWindowsFilePath=None):
        self.start_date = datetime.strptime(startDate, '%Y%m%d')
        self.end_date = datetime.strptime(endDate, '%Y%m%d')
        years = np.unique([self.start_date.year, self.end_date.year])
        self.data = None
        # Read data from the data_path for relevant years
        for year in years:
            if self.data is None:
                self.data = pd.read_parquet(os.path.join(data_path, f'{year}.parquet'))
            else:
                self.data = pd.concat([self.data, pd.read_parquet(os.path.join(data_path, f'{year}.parquet'))])
        # Filter data based on date range
        self.data = self.data[(self.data['Timestamp'] >= self.start_date) & 
                              (self.data['Timestamp'] <= self.end_date + timedelta(hours=23, minutes=59))]
        # Place GHI in the end
        self.data = self.data[[col for col in self.data.columns if col != 'GHI'] + ['GHI']]
        self.data = self.data.reset_index(drop = True)

        # Perform Normalization
        if normalization_params is None:
            # Calculate mean and standard deviation for normalization
            self.satMask_means = [np.mean(self.data[col]) for col in self.data.columns if col.startswith('SatMask')]
            self.satMask_stds = [np.std(self.data[col]) for col in self.data.columns if col.startswith('SatMask')]
            self.csm_mean, self.csm_std = np.mean(self.data['CSM']), np.std(self.data['CSM'])
            self.ghi_mean, self.ghi_std = np.mean(self.data['GHI']), np.std(self.data['GHI'])
            self.airTemp_mean, self.airTemp_std = np.mean(self.data['AirTemperature']), np.std(self.data['AirTemperature'])
            self.relHum_mean, self.relHum_std = np.mean(self.data['RelativeHumidity']), np.std(self.data['RelativeHumidity'])
        else:
            self.satMask_means, self.satMask_stds = normalization_params['SatMask_means'], normalization_params['SatMask_stds']
            self.csm_mean, self.csm_std = normalization_params['CSM_mean'], normalization_params['CSM_std']
            self.ghi_mean, self.ghi_std = normalization_params['GHI_mean'], normalization_params['GHI_std']
            self.airTemp_mean, self.airTemp_std = normalization_params['AirTemp_mean'], normalization_params['AirTemp_std']
            self.relHum_mean, self.relHum_std = normalization_params['RelHum_mean'], normalization_params['RelHum_std']
        
        self.history_len = history_len
        self.forecast_len = forecast_len

        # start_time = time.time()
        if contWindowsFilePath is None:
            self.contWindows = createContinuousRollingWindows(self.data, self.history_len, self.forecast_len)
        else:
            if os.path.exists(contWindowsFilePath) and os.path.isfile(contWindowsFilePath):
                with open(contWindowsFilePath, 'rb') as file:
                    loaded_data = pickle.load(file)
                if loaded_data['startDate'] == self.start_date and loaded_data['endDate'] == self.end_date:
                    print('Loaded Continuous Windows From File!')
                    self.contWindows = loaded_data['contWindows']
                else:
                    self.contWindows = createContinuousRollingWindows(self.data, self.history_len, self.forecast_len)
                    with open(contWindowsFilePath, 'wb') as file:
                        pickle.dump({'contWindows': self.contWindows, 'startDate': self.start_date, 'endDate': self.end_date}, file)
            else:
                self.contWindows = createContinuousRollingWindows(self.data, self.history_len, self.forecast_len)
                with open(contWindowsFilePath, 'wb') as file:
                    pickle.dump({'contWindows': self.contWindows, 'startDate': self.start_date, 'endDate': self.end_date}, file)

        # print(f"Elapsed time: {time.time()-start_time} seconds")

        # print(self.data.head())
        # print(self.data.describe())
        # print(self.data.tail())
        # print(len(self.data))
        # print(len(self.contWindows))

    def __len__(self):
        return len(self.contWindows)

    def __getitem__(self, idx):
        history_start = self.contWindows[idx]
        history_end = history_start + self.history_len
        forecast_start = history_end
        forecast_end = forecast_start + self.forecast_len

        history = self.data.iloc[history_start:history_end, 1:]     # Exclude timestamp
        target = self.data.iloc[forecast_start:forecast_end, 1:]    # All features as GHI

        for col in history.columns:
            if col.startswith('SatMask'):
                history[col] = normalize_elements(history[col], self.satMask_means[int(col.split('SatMask')[1])-1],
                                                  self.satMask_stds[int(col.split('SatMask')[1])-1])
                target[col] = normalize_elements(target[col], self.satMask_means[int(col.split('SatMask')[1])-1],
                                                 self.satMask_stds[int(col.split('SatMask')[1])-1])
        history['CSM'] = normalize_elements(history['CSM'], self.csm_mean, self.csm_std)
        history['GHI'] = normalize_elements(history['GHI'], self.ghi_mean, self.ghi_std)
        history['AirTemperature'] = normalize_elements(history['AirTemperature'], self.airTemp_mean, self.airTemp_mean)
        history['RelativeHumidity'] = normalize_elements(history['RelativeHumidity'], self.relHum_mean, self.relHum_std)
        history['SZA'] = np.sin(np.deg2rad(history['SZA']))
        history['SAA'] = np.cos(np.deg2rad(history['SAA']/2))
        
        target['CSM'] = normalize_elements(target['CSM'], self.csm_mean, self.csm_std)
        target['GHI'] = normalize_elements(target['GHI'], self.ghi_mean, self.ghi_std)
        target['AirTemperature'] = normalize_elements(target['AirTemperature'], self.airTemp_mean, self.airTemp_mean)
        target['RelativeHumidity'] = normalize_elements(target['RelativeHumidity'], self.relHum_mean, self.relHum_std)
        target['SZA'] = np.sin(np.deg2rad(target['SZA']))
        target['SAA'] = np.cos(np.deg2rad(target['SAA']/2))

        return history.values.astype('float32'), target.values.astype('float32')
    
    def getNormalizationParams(self):
        return {'SatMask_means': self.satMask_means, 'SatMask_stds': self.satMask_stds,
                'CSM_mean': self.csm_mean, 'CSM_std': self.csm_std, 'GHI_mean': self.ghi_mean, 'GHI_std': self.ghi_std,
                'AirTemp_mean': self.airTemp_mean, 'AirTemp_std': self.airTemp_std,
                'RelHum_mean': self.relHum_mean, 'RelHum_std': self.relHum_std}

if __name__ == "__main__":
    dataPath = os.path.join(os.getcwd(), 'data')
    startDate = "20200101"
    endDate = "20210807"
    contWindowsFilePath = os.path.join(os.getcwd(), 'data', 'contWindows.pkl')

    dataset = TimeSeriesDataset(dataPath, startDate, endDate, contWindowsFilePath=contWindowsFilePath)

    history, target = dataset.__getitem__(0)
    print(history.shape, history.dtype, target.shape, target.dtype)
    print(dataset.__len__())
    # print(history)
    # print(target)

    windows = createRollingWindowIndices(15, 4, 3, rolling_steps=2)
    for window in windows:
        print(window[0] + window[1])

    data = {'Timestamp': ['2020-06-30 19:40:00', '2020-06-30 19:42:00', '2020-06-30 19:46:00', '2020-06-30 19:48:00', '2020-06-30 19:50:00', '2020-06-30 19:52:00', '2020-06-30 19:54:00', '2020-06-30 19:56:00', '2020-06-30 19:58:00']}
    df = pd.DataFrame(data)
    windowIndices1 = [0, 1, 2, 3, 4]
    windowIndices2 = [2, 3, 4, 5, 6]

    result1 = checkContinuousWindow(df, windowIndices1, temporal_resolution='2min')
    result2 = checkContinuousWindow(df, windowIndices2, temporal_resolution='2min')

    print(result1)  # Output: False
    print(result2)  # Output: True

    windows = createContinuousRollingWindows(df, 3, 2, rolling_steps=1)
    for window in windows:
        print(window)