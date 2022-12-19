import importlib
import os

import torch
import torch.nn as nn
import tensorflow as tf
import kera
from keras.optimizers import Adam
from keras.metrics import MeanIoU

from model import build_unet
import data_preprocessing as dp 
from dice_coefficient import dice_coefficient
from sklearn.model_selection import train_test_split
from images_maks_generator import generate_data

def main():
    
    #Generate the images and masks 
    generate_data() 
    
    #Generate images and patches 
    patch_size = (16, 64, 64)
    step_size = (5, 64, 64)
    n_classes = 2
    input_img, input_mask  = dp.image_mask_patches(patch_size, step_size)
    
    #Remove empty patches
    input_img, input_mask = dp.remove_empty(input_img, input_mask)
    
    #Standardize the arrays of pixel and expand the dimension 
    train_img, train_mask = dp.standardize(input_img, input_mask)
    
    #Binarize the mask (i.e. binary segmentation)
    train_mask = dp.binarize(train_mask)
    
    #Categories the masks 
    train_mask = dp.one_hot_encoding(train_mask, n_classes)
    
    #Spliting into training and validation set  
    X_train, X_test, y_train, y_test = train_test_split(train_img, train_mask, test_size = 0.20, random_state = 0)
    
    #Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    
    # ===== Model, Optimizer and Criterion =====
    patch_size1 = 16
    patch_size2 = 64
    patch_size3 = 64
    channels=1

    LR = 0.0001
    optim = keras.optimizers.Adam(LR)

    model = build_unet((patch_size1,patch_size2,patch_size3,channels), n_classes)

    model.compile(optimizer = optim, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=dice_coefficient)
    print(model.summary())
    
    # ===== Train Model =====
    print("Training the model")
    history=model.fit(X_train, 
          y_train,
          batch_size=8, 
          epochs=100,
          verbose=1,
          validation_data=(X_test, y_test))
    
    model_path = '/saved_models/3dunetmodel_leaky_bs8_16x64x64_100epochs.h5'
    model.save(model_path)
    
    # ===== Predict Model =====
    print("Predict the model")
    y_pred=model.predict(X_test)
    
    #Predict on the test data
    y_pred_argmax=np.argmax(y_pred, axis=4)
    y_test_argmax = np.argmax(y_test, axis=4)
    
    IOU_keras = MeanIoU(num_classes=n_classes)  
    IOU_keras.update_state(y_test_argmax, y_pred_argmax)
    print("Mean IoU =", IOU_keras.result().numpy())
    