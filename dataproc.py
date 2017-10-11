# import torch libraries
import os
import torch
import numpy as np
import pandas as pd
import skimage.io as io
from torch.utils.data import Dataset

# create BSDS dataset class
class BSDSDataset(Dataset):
    """BSDS Dataset Class"""

    def __init__(self, fileNames, rootDir, transform=None):
        """
        Args:
            rootDir (string): Directory with all images.
            fileNames (string): Path to csv file with annotations.
            transform (callable, optional): Transform on a sample.
        """
        
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
        inputImage = io.imread(inputName)
        inputImage = inputImage.astype(np.float32)
        inputImage = inputImage[:,:,::-1]
        inputImage -= np.array((104.00698793,116.66876762,122.67891434))
        inputImage /= 255*np.array((0.225, 0.224, 0.229))
        inputImage = inputImage.transpose((2,0,1))
        inputImage = torch.from_numpy(np.flip(inputImage, axis=0).copy())
        
        targetImage = io.imread(targetName, as_grey=True)
        targetImage = targetImage > 0
        targetImage = targetImage.astype(np.float32)
        targetImage = np.expand_dims(targetImage, axis=0)        
        targetImage = torch.from_numpy(targetImage)
        
        # prepare dictionary
        sample = {'input': inputImage, 'target': targetImage}
        
        return inputImage, targetImage