from __future__ import print_function, division

import collections
from torch.utils.data.dataloader import default_collate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import os
import copy


def CreateDataLoader(opt):
    transform = transforms.Compose([
        transforms.Resize((170, 170)),
        transforms.CenterCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    image_datasets = datasets.ImageFolder('path/to/dataset', transform) 
    dataloader = DataLoader(image_datasets, batch_size=opt.batchsize, shuffle=True, num_workers=8, drop_last=True)
    
    return dataloader

def CreateDataLoader_pre(opt):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    image_datasets = datasets.ImageFolder('/path/to/dataset', transform) 
    dataloader = DataLoader(image_datasets, batch_size=opt.batchsize, shuffle=True, num_workers=8, drop_last=True)
    
    return dataloader

def CreateDataLoader_test(opt):
    transform = transforms.Compose([
        transforms.Resize((170, 170)),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    image_datasets = datasets.ImageFolder('path/to/dataset', transform) 
    dataloader = DataLoader(image_datasets, batch_size=1, shuffle=False, num_workers=8, drop_last=False)
    
    return dataloader