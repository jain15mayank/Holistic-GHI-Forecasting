import os
import numpy as np
from datetime import datetime
import zipfile
import pygrib
from PIL import Image
from copy import copy, deepcopy
from scipy import spatial
import shutil

from joblib import Parallel, delayed
nProcesses = int(os.cpu_count()/4) - 1 # Can change the number of processes if required
print("Num Processes = ", nProcesses)

year = 2022

# Specify the location of interest
int_lat = 48.7
int_lon = 2.2

# Specify the pixels you want around the location of interest - will generate a square image (2*nPXLS, 2*nPXLS)
nPXLS = 50

zip_file_address = os.path.join(os.getcwd(), "../EUMETSAT/data/", str(year))

IMAGE_DIR = os.path.join(os.getcwd(), "../EUMETSAT/data/", "images"+str(year))
# Check if IMAGE_DIR exists
if not os.path.exists(IMAGE_DIR):
    try:
        # Create the directory and any necessary parent directories
        os.makedirs(IMAGE_DIR)
        print(f"Directory {IMAGE_DIR} created.")
    except OSError as e:
        print(f"Error creating directory {IMAGE_DIR}: {e}")
else:
    print(f"Directory {IMAGE_DIR} already exists.")

TEMP_DIR = os.path.join(os.getcwd(), "../EUMETSAT/data/", "temp"+str(year))
# Check if TEMP_DIR exists
if os.path.exists(TEMP_DIR):
    try:
        # Remove the directory and its contents
        shutil.rmtree(TEMP_DIR)
        print(f"Directory {TEMP_DIR} and its contents removed.")
    except OSError as e:
        print(f"Error removing directory {TEMP_DIR}: {e}")
# Create the directory and any necessary parent directories
try:
    os.makedirs(TEMP_DIR)
    print(f"Directory {TEMP_DIR} created.")
except OSError as e:
    print(f"Error creating directory {TEMP_DIR}: {e}")


print(len([entry for entry in os.listdir(zip_file_address) if os.path.isfile(os.path.join(zip_file_address, entry))]))
print(len([entry for entry in os.listdir(zip_file_address) if os.path.isdir(os.path.join(zip_file_address, entry))]))


def parseZipFileAndExtractImage(zip_file_address, entry, pid):
    zip_file_path = os.path.join(zip_file_address, entry)
    print("Process " + str(pid) + " is working on " + zip_file_path, flush=True)
    if os.path.isfile(zip_file_path):
        timestamp = datetime.strptime(entry, "%Y%m%dT%H%M%S.zip")
        print(entry, timestamp)
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # List the contents of the zip file
                file_list = sorted(zip_ref.namelist())
                for file_name in file_list:
                    if file_name.endswith('.grb'):
                        # Extract the GRIB file to a temporary directory
                        temp_dir_path = os.path.join(TEMP_DIR, 'temp_dir' + str(pid))
                        zip_ref.extract(file_name, path=temp_dir_path)
                        # Parse the GRIB file
                        grib_file_path = os.path.join(temp_dir_path, file_name)
                        grib = pygrib.open(grib_file_path)
                        # Iterate through the GRIB messages
                        for msg in grib:
                            # Extract GRIB Object
                            grb = grib.select(name=msg.name)[0]
                            # Extract GRIB Values
                            gribdata = grb.values # same as grb['values']
                            gribdata = np.uint8(np.array(gribdata))
                            gribdata = deepcopy(np.flip(gribdata, axis=(0,1)))
                            # Locate the Point of Interest in the matrix of GRIB Values
                            lats, lons = grb.latlons()
                            newData = grb.data(lat1 = int_lat-2, lat2 = int_lat+2, lon1 = int_lon-2, lon2 = int_lon+2)
                            newVals = newData[0]
                            newLats = newData[1]
                            newLons = newData[2]
                            newLatsLons = np.concatenate((newLats[:, None], newLons[:, None]), axis=1)
                            [int_lat_exact, int_lon_exact] = newLatsLons[spatial.KDTree(newLatsLons).query([int_lat, int_lon])[1]]
                            all_idx = np.argwhere(np.logical_and(np.equal(lats,int_lat_exact), np.equal(lons,int_lon_exact)))[0]
                            int_row = lats.shape[0]-all_idx[0]
                            int_col = all_idx[1]
                            # Crop and save a binary cloud mask map (2*nPXLS, 2*nPXLS)
                            gribdata_cropped = gribdata[int_row-nPXLS:int_row+nPXLS, int_col-nPXLS:int_col+nPXLS]
                            image_like = np.uint8(np.zeros((gribdata_cropped.shape[0], gribdata_cropped.shape[1])))
                            image_like[:,:] = np.where(np.equal(gribdata_cropped,2), 255, image_like[:,:]) # Just 255 where there are clouds
                            im = Image.fromarray(image_like)
                            im.save(os.path.join(IMAGE_DIR, timestamp.strftime('%Y%m%dT%H%M%S')+".png"))
                        # Close the GRIB file
                        grib.close()
                        # Remove the temporary directory and extracted GRIB file
                        os.remove(grib_file_path)
                        os.rmdir(temp_dir_path)
        except zipfile.BadZipFile:
            print("Invalid ZIP file: " + zip_file_path)
        except FileNotFoundError:
            print("File not found: " + zip_file_path)


def process_entry(pid, entry, zip_file_address):
    try:
        parseZipFileAndExtractImage(zip_file_address, entry, pid)
    except Exception as e:
        print(f"On pid {pid}, error processing entry {entry}: {e}", flush=True)

entries = sorted(os.listdir(zip_file_address))

Parallel(n_jobs=nProcesses, verbose=11, batch_size=1)(
    delayed(parseZipFileAndExtractImage)(zip_file_address, entry, pid) for pid, entry in enumerate(entries)
    )