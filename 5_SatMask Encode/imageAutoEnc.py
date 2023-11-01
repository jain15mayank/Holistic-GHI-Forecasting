import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score
#from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback

'''#################################################################################
# SPECIFY ARGUMENTS
#################################################################################'''
# For loading the dataset
trainYearList = [2020, 2021]
# valYearList = [2022]

# General random seed value to be used in the script
randomSeed = 51

# Define sample images to train autoencoder on
num_samples = 20000

# Number of Latent Space Dimensions in the AutoEncoder
latent_dim = 32
# Number of epochs to train for
prevEpochsDifferent = False # Whether, or not, previous run had different number of epochs
nEpochs = 2000
# Batch size for training and evaluation
BATCH_SIZE = 32
# Learning rate to be used
lRate = 1e-7
# Save checkpoints with this period
ckpt_period = 50
# Early stopping patience interval
esPatience = 20
# Whether or not to resume training from latest checkpoint?
resumeTrainingFlag = False

# Location where checkpoints are saved
ckptDir = './models/Conv2/ckpts/'+str(latent_dim)+'/'
if prevEpochsDifferent and resumeTrainingFlag:
    for ckptPath in os.listdir(ckptDir):
        os.rename(ckptDir+ckptPath, ckptDir+ckptPath.split('_')[0]+'_'+ckptPath.split('_')[1]+'_'+str(nEpochs)+'ckpt_'+ckptPath.split('_')[3])

ckptSavePath = ckptDir+'autoencoder_'+str(latent_dim)+'_'+str(nEpochs)+'ckpt_'

# Location to save model
modelSavePath = './models/Conv2/'+str(latent_dim)+'/autoencoder_'+str(latent_dim)+'_'+str(nEpochs)

# Results Path
resultsFolder = './results/Conv2/'
jsonFilePath = resultsFolder+'loss_log_'+str(latent_dim)+'.json'



'''#################################################################################
# LOAD DATASET AND PERFORM TRAIN-TEST SPLIT
#################################################################################'''

x_train = None
for trainYear in trainYearList:
    npyFileAddress = './data/'+str(trainYear)+'/images'+str(trainYear)+'.npz'
    imgs = np.load(npyFileAddress)['imgs']
    if x_train is None:
        x_train = imgs
    else:
        x_train = np.concatenate((x_train, imgs))

del imgs

# x_val = None
# for valYear in valYearList:
#     npyFileAddress = './data/'+str(valYear)+'/images'+str(valYear)+'.npz'
#     imgs = np.load(npyFileAddress)['imgs']
#     if x_val is None:
#         x_val = imgs
#     else:
#         x_val = np.concatenate((x_val, imgs))

print(x_train.shape, x_train.dtype)
# print(x_val.shape, x_val.dtype)

# Extract random samples from x_train
sampleIndices = np.arange(x_train.shape[0])
np.random.shuffle(sampleIndices)
sampleIndices = sampleIndices[:num_samples]
x_train = x_train[sampleIndices]
del sampleIndices

x_train = x_train.astype(np.float32)
# x_val = x_val.astype('float32')

x_train = np.expand_dims(x_train, axis=-1)
# x_val = np.expand_dims(x_val, axis=-1)

print (x_train.shape)
# print (x_val.shape)

print(np.max(x_train), np.min(x_train))
# print(np.max(x_val), np.min(x_val))

imgShape = x_train.shape

# Convert the NumPy array to a TensorFlow dataset
x_train = tf.data.Dataset.from_tensor_slices(x_train)
shuffle_buffer_size = 10000  # Adjust this as needed
x_train = x_train.shuffle(shuffle_buffer_size)
x_train = x_train.batch(BATCH_SIZE)



'''#################################################################################
# DEFINE AUTOENCODER
#################################################################################'''
class ConvAutoencoder(Model):
    def __init__(self, latent_dim):
        super(ConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(imgShape[1], imgShape[2], 1)),
            layers.Conv2D(2, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(4, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(2, (3, 3), activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu')
        ])
        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(int(imgShape[1]/4)*int(imgShape[2]/4)*2, activation='relu'),
            layers.Reshape((int(imgShape[1]/4), int(imgShape[2]/4), 2)),
            layers.Conv2D(2, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(2, (3, 3), activation='relu', padding='same'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ConvAutoencoderOld(Model):
    def __init__(self, latent_dim):
        super(ConvAutoencoderOld, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(imgShape[1], imgShape[2], 1)),
            layers.Conv2D(4, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(2, (3, 3), activation='relu', padding='same', strides=2),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(int(imgShape[1]*imgShape[2]/8), activation='sigmoid'),
            layers.Reshape((int(imgShape[1]/4), int(imgShape[2]/4), 2)),
            layers.Conv2DTranspose(2, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(4, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class DenseAutoencoder(Model):
    def __init__(self, latent_dim):
        super(DenseAutoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(imgShape[1]*imgShape[2], activation='sigmoid'),
            layers.Reshape((imgShape[1], imgShape[2]))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# tf.keras.mixed_precision.set_global_policy('mixed_float16')

autoencoder = ConvAutoencoderOld(latent_dim)
autoencoder.compile(optimizer=Adam(learning_rate=lRate), loss=losses.MeanSquaredError())
autoencoder.build((BATCH_SIZE, imgShape[1], imgShape[2], 1))
autoencoder.encoder.summary()
autoencoder.decoder.summary()



'''#################################################################################
# TRAIN THE AUTOENCODER MODEL
#################################################################################'''
def deleteJSONrowsAfterEpoch(jsonFile, epoch):
    json_lines = []
    with open(jsonFile, mode='rt', buffering=1) as open_file:
        for line in open_file.readlines():
            j = json.loads(line)
            if int(j['epoch']) < epoch:
                json_lines.append(line)

    with open(jsonFile, mode='wt', buffering=1) as open_file:
        open_file.writelines(json_lines)

if resumeTrainingFlag:
    # Find the exact path of the latest saved checkpoint
    latestTrainedEpochs = max([int(ckptFile.split(ckptSavePath.split(ckptDir)[1])[1]) for ckptFile in os.listdir(ckptDir)])
    latestCkptPath = ckptSavePath+'{epoch:02d}'.format(epoch=latestTrainedEpochs)
    # Load latest saved checkpoint
    autoencoder = load_model(latestCkptPath)
    
    checkpoint = ModelCheckpoint(ckptSavePath+'{epoch:02d}' ,save_freq='epoch', period=ckpt_period, save_format='tf')

    # Delete JSON log epochs which happened after the latest checkpoint save
    deleteJSONrowsAfterEpoch(jsonFilePath, latestTrainedEpochs)

    json_log = open(jsonFilePath, mode='at', buffering=1)

    json_log = open(jsonFilePath, mode='at', buffering=1)
    json_logging_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: json_log.write(
            # json.dumps({'epoch': epoch, 'loss': logs['loss'], 'val_loss': logs['val_loss']}) + '\n'),
            json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
        on_train_end=lambda logs: json_log.close()
    )

    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=esPatience)

    callbacks_list = [checkpoint, json_logging_callback]#, es]

    print('Starting to fit the model!')
    autoencoder.fit(x_train,# x_train,
                    epochs=nEpochs,
                    # validation_data=(x_val, x_val),
                    callbacks=callbacks_list,
                    initial_epoch=latestTrainedEpochs,
                    verbose=2)

else:
    checkpoint = ModelCheckpoint(ckptSavePath+'{epoch:02d}' ,save_freq='epoch', period=ckpt_period, save_format='tf')

    json_log = open(jsonFilePath, mode='wt', buffering=1)
    json_logging_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: json_log.write(
            # json.dumps({'epoch': epoch, 'loss': logs['loss'], 'val_loss': logs['val_loss']}) + '\n'),
            json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
        on_train_end=lambda logs: json_log.close()
    )

    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=esPatience)

    callbacks_list = [checkpoint, json_logging_callback]#, es]
    
    print('Starting to fit the model!')
    autoencoder.fit(tf.data.Dataset.zip((x_train, x_train)),# x_train,
                    epochs=nEpochs,
                    # batch_size=BATCH_SIZE,
                    # shuffle=True,
                    # validation_data=(x_val, x_val),
                    callbacks=callbacks_list,
                    verbose=2)
print('Model Training Completed!')

print('Saving Model...')
autoencoder.save(modelSavePath, save_format='tf')
print('Model Saved!')

## Plot and Save Training Curves #######################################

training_logs = {'epoch'        : np.array([]),
                 'loss'         : np.array([])}#,
                #  'val_loss'     : np.array([])}
with open(resultsFolder+'loss_log_'+str(latent_dim)+'.json') as logsJSON:
    lines = logsJSON.readlines()
    for line in lines:
        logsDict = json.loads(line)
        training_logs['epoch'] = np.append(training_logs['epoch'], logsDict['epoch'])
        training_logs['loss'] = np.append(training_logs['loss'], logsDict['loss'])
        # training_logs['val_loss'] = np.append(training_logs['val_loss'], logsDict['val_loss'])

# Plot Loss Curves
plt.figure(figsize=(10, 6))
plt.rc('font', size=17)         # controls default text sizes
plt.rc('axes', titlesize=19)    # fontsize of the axes title
plt.rc('axes', labelsize=19)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=17)   # fontsize of the tick labels
plt.rc('ytick', labelsize=17)   # fontsize of the tick labels
plt.rc('legend', fontsize=17)   # legend fontsize
plt.rc('figure', titlesize=19)  # fontsize of the figure title
plt.plot(training_logs['epoch'], training_logs['loss'], 'r--', label='Training Loss')
# plt.plot(training_logs['epoch'], training_logs['val_loss'], 'b--', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(resultsFolder+'training_loss_characteristics_'+str(latent_dim)+'.pdf', bbox_inches = 'tight', pad_inches = 0.05)
plt.close()


del autoencoder # Delete autoencoder model
print('Loading Saved Model...')
loaded_model = load_model(modelSavePath)
print('Saved Model Loaded!')



'''#################################################################################
# TEST THE TRAINED AUTOENCODER MODEL
#################################################################################'''
# loaded_model.evaluate(x=x_val, y=x_val, batch_size=BATCH_SIZE, verbose=2)

# n = 10

# encoded_imgs = loaded_model.encoder(x_val[:n, :, :, :]).numpy()
# decoded_imgs = loaded_model.decoder(encoded_imgs).numpy()

# plt.figure(figsize=(20, 4))
# for i in range(n):
#     loss = loaded_model.evaluate(x=x_val[i][tf.newaxis, ...], y=x_val[i][tf.newaxis, ...], verbose=0)
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_val[i], vmin=0, vmax=1)
#     plt.title("original\nmn={0:1.1f}; mx={1:1.1f}".format(np.min(x_val[i]), np.max(x_val[i])))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i], vmin=0, vmax=1)
#     plt.title("reconstructed\nmn={0:1.1f}; mx={1:1.1f}".format(np.min(decoded_imgs[i]), np.max(decoded_imgs[i])))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.savefig(resultsFolder+'resultPlots.png')
# plt.close()