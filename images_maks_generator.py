import h5py
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import torch

def generate_data():

    #filename of the data set
    filename = "fission.hdf5"

    #opening hdf5 file
    f = h5py.File(filename, "r")

    #reading masks
    for key in list(f['time_lapse_train']['gt']):
        data = data = np.array(f['time_lapse_train']['gt'][key], dtype=np.int16)
        image_name = key.split('.')[0] + "_mask.tif"
        tifffile.imsave(image_name, data)

    #reading images
    for key in list(f['time_lapse_train']['img']):
        data = data = np.array(f['time_lapse_train']['img'][key], dtype=np.int16)
        image_name = key.split('.')[0] + "_image.tif"
        tifffile.imsave(image_name, data)