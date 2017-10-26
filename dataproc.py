# dataproc.py: Dataset loader classes for BSDS
# Author: Nishanth Koganti
# Date: 2017/10/11

# Source: http://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# Issues:
# Merge TrainDataset and TestDataset classes

# import libraries
import os
import numpy as np
import pandas as pd
from PIL import Image
import skimage.io as io

# import torch modules
import torch
from torch.utils.data import Dataset

# BSDS dataset class for training data
class TrainDataset(Dataset):
    def __init__(self, fileNames, rootDir, transform=None):        
        self.rootDir = rootDir
        self.transform = transform
        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=' ')

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        # input and target images
        inputName = os.path.join(self.rootDir, self.frame.iloc[idx, 0])
        targetName = os.path.join(self.rootDir, self.frame.iloc[idx, 1])
        
        # process the images
        inputImage = np.asarray(Image.open(inputName).convert('RGB'))
        inputImage = inputImage.astype(np.float32)
        inputImage = inputImage/255.0
        inputImage -= np.array((0.485, 0.456, 0.406))
        inputImage /= np.array((0.229, 0.224, 0.225))
        inputImage = inputImage.transpose((2,0,1))
        #inputImage = torch.from_numpy(inputImage)
        
        targetImage = io.imread(targetName)
        if len(targetImage.shape) == 3:
            targetImage = targetImage[:,:,0]
        targetImage = targetImage/255.0
        targetImage = targetImage > 0.0
        targetImage = targetImage.astype(np.float32)
        targetImage = np.expand_dims(targetImage, axis=0)        
        #targetImage = torch.from_numpy(targetImage)
        
        return inputImage, targetImage
    
# dataset class for test dataset
class TestDataset(Dataset):
    def __init__(self, fileNames, rootDir, transform=None):
        self.rootDir = rootDir
        self.transform = transform
        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=' ')

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        # input and target images
        fname = self.frame.iloc[idx, 0]
        inputName = os.path.join(self.rootDir, fname)
        
        # process the images
        inputImage = np.asarray(Image.open(inputName).convert('RGB'))
        inputImage = inputImage.astype(np.float32)
        inputImage = inputImage/255.0
        inputImage -= np.array((0.485, 0.456, 0.406))
        inputImage /= np.array((0.229, 0.224, 0.225))
        inputImage = inputImage.transpose((2,0,1))
        #inputImage = torch.from_numpy(inputImage)
                
        return inputImage, fname