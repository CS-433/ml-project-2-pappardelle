from skimage import io
from patchify import patchify, unpatchify
import numpy as np
from keras import backend as K
from tensorflow.keras.utils import to_categorical
import random

def image_mask_patches(patch_size, step_size): 
    image = []

    image.append(io.imread('/images/wt_pom1D_01_07_R3D_REF_image.tif'))
    image.append(io.imread('/images/wt_pom1D_01_15_R3D_REF_image.tif'))
    image.append(io.imread('/images/wt_pom1D_01_20_R3D_REF_image.tif'))
    image.append(io.imread('/images/train/wt_pom1D_01_30_R3D_REF_image.tif'))

    img_patches = []
    img_patches.append(patchify(image[0], patch_size, step=step_size))
    img_patches.append(patchify(image[1], patch_size, step=step_size)) 
    img_patches.append(patchify(image[2], patch_size, step=step_size))  
    img_patches.append(patchify(image[3], patch_size, step=step_size))  
    
    mask = []

    mask.append(io.imread('/masks/wt_pom1D_01_07_R3D_REF_mask.tif'))
    mask.append(io.imread('/masks/wt_pom1D_01_15_R3D_REF_mask.tif'))
    mask.append(io.imread('/masks/wt_pom1D_01_20_R3D_REF_mask.tif'))
    mask.append(io.imread('/masks/wt_pom1D_01_30_R3D_REF_mask.tif'))

    mask_patches = []
    mask_patches.append(patchify(mask[0], patch_size, step=step_size))
    mask_patches.append(patchify(mask[1], patch_size, step=step_size))  
    mask_patches.append(patchify(mask[2], patch_size, step=step_size))  
    mask_patches.append(patchify(mask[3], patch_size, step=step_size))
    
    input_img = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], img_patches.shape[5]))
    input_mask = np.reshape(mask_patches, (-1, mask_patches.shape[3], mask_patches.shape[4], mask_patches.shape[5]))
    
    return input_img, input_mask 

def standardize(input_img, input_mask): 
    std = input_img.max() 
    train_img = input_img / std 
    train_img = np.expand_dims(train_img, axis=4)
    train_mask = np.expand_dims(input_mask, axis=4)
    
    return train_img, train_mask
    
def binarize(train_mask): 
    train_mask[train_mask>1] = 1
    return train_mask

def one_hot_encoding(train_mask, n_classes): 
    return to_categorical(train_mask, num_classes=n_classes)

def remove_empty(input_img, input_mask):
    idx_img = np.where(input_mask.mean(axis=(1,2,3)) != 0)[0]
    input_img = input_img[idx_img]
    input_mask = input_mask[idx_img]
    
    return input_img, input_mask

def random_image_mask(input_img, input_mask, factor)
    nb_samples = input_img.shape[0]
    sl = random.sample(range(nb_samples), nb_samples//factor)
    input_img = input_img[sl]
    input_mask = input_mask[sl]
    
    return input_img, input_mask
