import importlib
import os

import torch
import torch.nn as nn

import model
import train
import dataLoader
from sklearn.model_selection import train_test_split

def main():
    
    #Generate the data
    train_img, train_label = dataLoader.generate_data()
    
    #splitting the train set 
    X_train, X_test, y_train, y_test = train_test_split(train_img, train_mask_cat, test_size = 0.20, random_state = 0)
    
    #zipping to train and val set 
    train_set = torch.cat((X_train, y_train), axis=1)
    val_set = torch.cat((X_test, y_test), axis=1)
    
    #Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ===== Data Loading =====
    batch_size = ##to decide
    train_loader, val_loader = dataLoader.data_loader(train_set, val_set, batch_size)
    
    # ===== Model, Optimizer and Criterion =====
    optimizer_kwargs = dict(
        lr=1e-3,
        weight_decay=1e-2,
        )
    model = Model()
    model = model.to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    #criterion = torch.nn.functional.cross_entropy
    crossentropy = nn.CrossEntropyLoss(weight=[0.5, 0.5])
    dice = DiceLoss(apply_softmax=True, weight=[0.5, 0.5])
    criterion = CombinedLoss([crossentropy, dice], weight=[0.5, 0.5], device=device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(len(train_loader.dataset) * num_epochs) // train_loader.batch_size,
    )
    
    # ===== Train Model =====
    num_epochs = 100 #to modify
    
    trainer = train.Trainer(num_epochs=num_epochs, model=model, device=device, criterion=criterion, optimizer=optimizer, train_loader=train_loader, lr_scheduler=scheduler)
    
    trainer.training()
    