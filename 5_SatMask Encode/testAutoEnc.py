import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

'''#################################################################################
# SPECIFY ARGUMENTS
#################################################################################'''
# For loading the dataset
testYearList = [2020]

# General random seed value to be used in the script
randomSeed = 51

# Number of Latent Space Dimensions in the AutoEncoder
latent_dim = 64
# Number of epochs to train for
nEpochs = 1000
# Learning rate to be used
lRate = 1e-5
# Save checkpoints with this period
ckpt_period = 50

# Location where checkpoints are saved
ckptSavePath = './models/Conv2/ckpts/autoencoder_'+str(latent_dim)+'_'+str(nEpochs)+'ckpt_'

# Location where checkpoints are saved
ckptPlotsPath = './results/Conv2/ckptResults/imageckpt_'



'''#################################################################################
# LOAD DATASET AND PERFORM TRAIN-TEST SPLIT
#################################################################################'''

x_test = None
for testYear in testYearList:
    npyFileAddress = './data/'+str(testYear)+'/images'+str(testYear)+'.npz'
    imgs = np.load(npyFileAddress)['imgs']
    if x_test is None:
        x_test = imgs
    else:
        x_test = np.concatenate((x_test, imgs))

print(x_test.shape, x_test.dtype)

x_test = x_test.astype('float32')
#Randomly Shuffle The Test Set
np.random.shuffle(x_test)
x_test = x_test[..., tf.newaxis]

print (x_test.shape, x_test.dtype)

print(np.max(x_test), np.min(x_test))



for nckpt in range(ckpt_period, nEpochs+1, ckpt_period):
    '''#################################################################################
    # LOAD AUTOENCODER
    #################################################################################'''

    print('At checkpoint with epoch number =', nckpt)
    loaded_model = load_model(ckptSavePath+str(nckpt))


    '''#################################################################################
    # TEST THE TRAINED AUTOENCODER MODEL
    #################################################################################'''
    batch_size = 32
    num_samples = 32
    encoded_imgs_list = []
    decoded_imgs_list = []

    # Iterate over the test set in batches
    for i in range(0, num_samples, batch_size):
        x_batch = x_test[i:i + batch_size]  # Extract a batch of data

        # Generate predictions for the current batch
        encoded_batch = loaded_model.encoder(x_batch).numpy().astype('float16')
        decoded_batch = loaded_model.decoder(encoded_batch).numpy().astype('float16')

        # Append the batch predictions to the lists
        encoded_imgs_list.append(encoded_batch)
        decoded_imgs_list.append(decoded_batch)

    # Concatenate the predictions from all batches
    encoded_imgs = np.vstack(encoded_imgs_list)
    decoded_imgs = np.vstack(decoded_imgs_list)

    # encoded_imgs = loaded_model.encoder(x_test).numpy()
    # decoded_imgs = loaded_model.decoder(encoded_imgs).numpy()

    n = 10
    fig = plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i], vmin=0, vmax=1)
        plt.title("original\nmn={0:1.1f}; mx={1:1.1f}".format(np.min(x_test[i]), np.max(x_test[i])))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i], vmin=0, vmax=1)
        plt.title("reconstructed\nmn={0:1.1f}; mx={1:1.1f}".format(np.min(decoded_imgs[i]), np.max(decoded_imgs[i])))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.suptitle('Epoch Number ='+str(nckpt))
    plt.savefig(ckptPlotsPath+"{:04d}".format(int(nckpt/ckpt_period))+'.png')
    plt.close()