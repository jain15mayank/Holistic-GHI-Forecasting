import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# Location where checkpoints are saved
latent_dim = 64
nEpochs = 2000
nckpt = 1000      # After 1000 epochs
ckptSavePath = './models/Conv/ckpts/autoencoder_'+str(latent_dim)+'_'+str(nEpochs)+'ckpt_'+str(nckpt)

# Year to work with
year = 2020

# Location of Data Files - Npz and timestamp
npyFileAddress = './data/'+str(year)+'/images'+str(year)+'.npz'
dtsFileAddress = './data/'+str(year)+'/timestamps'+str(year)#+'.pkl' # Add .pkl extension for 2022

# Define your batch size
batch_size = 32

# Final location to save the encoded vectors and associated timestamps
save_data_path = './data/'+str(year)+'/encoded_data_'+str(year)+'.npz'

# Load images and timestamps
loadedImgs = np.load(npyFileAddress)['imgs']
with open(dtsFileAddress, 'rb') as fp:
    loadedDats = pickle.load(fp)

assert(len(loadedImgs) == len(loadedDats))

# Initialize arrays to store data
num_samples = len(loadedDats)
encoded_imgs_list = np.empty((num_samples, latent_dim), dtype=np.float32)
timestamps_list = np.empty(num_samples, dtype=object)

# Load the trained autoencoder model
loaded_model = load_model(ckptSavePath)

# Initialize an index to keep track of the current position
index = 0
# Iterate over the test set in batches
for i in range(0, num_samples, batch_size):
    x_batch = loadedImgs[i:i + batch_size]  # Extract a batch of images
    timestamps_batch = loadedDats[i:i + batch_size]  # Extract corresponding timestamps

    # Generate predictions for the current batch
    encoded_batch = loaded_model.encoder(x_batch).numpy()

    # Store the batch encoded vectors and timestamps in the pre-allocated arrays
    valid_batch_size = min(batch_size, num_samples - index)
    encoded_imgs_list[index:index + valid_batch_size] = encoded_batch[:valid_batch_size]
    timestamps_list[index:index + valid_batch_size] = timestamps_batch[:valid_batch_size]

    index += valid_batch_size

# Save the data in .npz format
timestamps_numeric = pd.to_datetime(timestamps_list).view('int64')
np.savez_compressed(save_data_path, timestamps=timestamps_numeric, encoded_data=encoded_imgs_list)


# Load the data from the .npz file
loaded_data = np.load(save_data_path)
loaded_timestamps_numeric = loaded_data['timestamps']
loaded_timestamps = pd.to_datetime(loaded_timestamps_numeric)
loaded_encoded_data = loaded_data['encoded_data']

assert np.array_equal(loaded_encoded_data, encoded_imgs_list)
assert loaded_timestamps.equals(pd.to_datetime(timestamps_list))
print('Loaded data is an exact match to the data that was saved!')