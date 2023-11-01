import numpy as np
import zipfile
import tarfile
import imageio.v3 as imageio
from dateutil.parser import isoparse
import time
import pickle
import gc

'''#################################################################################
# SPECIFY ARGUMENTS
#################################################################################'''

yearNum = 2022

zipFileAddress = './data/'+str(yearNum)+'/images'+str(yearNum)+'.zip'
tarFileAddress = './data/'+str(yearNum)+'/images'+str(yearNum)+'.tar.gz'
npyFileAddress = './data/'+str(yearNum)+'/images'+str(yearNum)+'.npz'
dtsFileAddress = './data/'+str(yearNum)+'/timestamps'+str(yearNum)+'.pkl'

'''#################################################################################
# MAIN SCRIPT STARTS HERE
#################################################################################'''

# archive = zipfile.ZipFile(zipFileAddress, 'r')
tar = tarfile.open(tarFileAddress, "r:gz")

# fileNames = archive.namelist()
# fileNames.sort()
fileNames = sorted(tar.getnames())

print('Allocating memory for the huge array to store images...')
start_time = time.time()
imgs = np.zeros((len(fileNames)-1,100,100)).astype('bool')
print('Array allocated in %s seconds!' % (time.time() - start_time))
dats = []
iters = 0
print('Reading images from the zip file and storing them in the bool array...')
start_time = time.time()
gc.disable()
for fileName in fileNames:
    if not fileName[-4:]=='.png':
        if iters > 0:
            raise Exception('Non-PNG file encountered in the .zip file!')
        continue
    else:
        # with archive.open(fileName) as imageFile:
        with tar.extractfile(fileName) as imageFile:
            imgs[iters,:,:] = np.array(imageio.imread(imageFile)).astype('bool')
        dats.append(isoparse(fileName.split('/')[1][:-4]))
    iters += 1
    if iters % 500 == 0:
        print(iters)
gc.enable()
print('Data reading and storing completed in %s seconds!' % (time.time() - start_time))

print('Saving the image data in compressed NumPy array...')
start_time = time.time()
np.savez_compressed(npyFileAddress, imgs=imgs)
print('Image data saved in %s seconds!' % (time.time() - start_time))

print('Loading the image data from the compressed NumPy array...')
start_time = time.time()
loaded = np.load(npyFileAddress)
print('Image data loaded in %s seconds!' % (time.time() - start_time))
print(np.array_equal(imgs, loaded['imgs']))


print('Saving the DateTime data as pickle dump...')
start_time = time.time()
with open(dtsFileAddress, 'wb') as fp:
    pickle.dump(dats, fp)
print('DateTime data saved in %s seconds!' % (time.time() - start_time))

print('Loading the DateTime data from the pickle dump...')
start_time = time.time()
with open(dtsFileAddress, 'rb') as fp:
    loadedDats = pickle.load(fp)
print('DateTime data loaded in %s seconds!' % (time.time() - start_time))
print(loadedDats == dats)