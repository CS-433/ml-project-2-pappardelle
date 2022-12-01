import importlib
import os

import torch
import torch.nn as nn

import model
import train
import dataLoader

logger = utils.get_logger('UNet3D')

def main():
    
    #Generate the data
    image, mask = dataLoader.generate_data()
    
    train_set = ##need to zip them 
    
    #need to split it into train and validate set 
    ############## TO DO ################################
    val_set =
    
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
    criterion = torch.nn.functional.cross_entropy
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(len(train_loader.dataset) * num_epochs) // train_loader.batch_size,
    )
    
    # ===== Train Model =====
    num_epochs = ##to define
    
    trainer = train.Trainer(num_epochs=num_epochs, model=model, device=device, criterion=criterion, optimizer=optimizer, train_loader=train_loader, lr_scheduler=scheduler)
    
    trainer.training()
    