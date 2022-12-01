import h5py
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import torch

def generate_data():

    filename = "datasets.hdf5"

    #opening hdf5 file
    f = h5py.File(filename, "r")

    #initializations 
    gt = []
    img = []

    #reading masks
    for key in list(f['time_lapse_train']['gt']):
        data = data = np.array(f['time_lapse_train']['gt'][key], dtype=np.int16)
        image_name = key.split('.')[0] + "_mask.tif"
        tifffile.imsave(image_name, data)
        data = torch.from_numpy(data)
        gt.append(data)

    #reading images
    for key in list(f['time_lapse_train']['img']):
        data = data = np.array(f['time_lapse_train']['img'][key], dtype=np.int16)
        image_name = key.split('.')[0] + "_image.tif"
        tifffile.imsave(image_name, data)
        data = torch.from_numpy(data)
        img.append(data)

    #first try, we will only train on the first element of the list 
    return gt[0], img[0]

def data_loader(train_set, val_set, batch_size):
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,  # Shuffle the iteration order over the dataset
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        num_workers=2,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
    )
    
    return train_loader, val_loader