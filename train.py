# main program to train hed model
# Author: Nishanth
# Date: 2017/10/19

# import torch libraries
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np

# import utility functions
from model import HED
from trainer import Trainer
from dataproc import TrainDataset

# fix random seed
rng = np.random.RandomState(37148)

# GPU ID
gpuID = 0

# batch size
nBatch = 1

# load the images dataset
dataRoot = '../HED-BSDS/'
valPath = dataRoot+'val_pair.lst'
trainPath = dataRoot+'train_pair.lst'

# create data loaders from dataset
valDataset = TrainDataset(valPath, dataRoot)
trainDataset = TrainDataset(trainPath, dataRoot)

valDataloader = DataLoader(valDataset, shuffle=False)
trainDataloader = DataLoader(trainDataset, shuffle=False)

# initialize the network
net = HED(pretrained=False)
net.cuda(gpuID)

# define the optimizer
optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9, weight_decay=0.0002)

# initialize trainer class
trainer = Trainer(net, optimizer, trainDataloader, valDataloader, 
                  nBatch=nBatch, maxEpochs=10, cuda=True, gpuID=gpuID)

# train the network
trainer.train()
