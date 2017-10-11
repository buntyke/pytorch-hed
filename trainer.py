import os, time
import numpy as np
from PIL import Image
import os.path as osp

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"

# binary cross entropy loss in 2D
def bce2d(input, target):
    # do not compute gradients w.r.t target
    _assert_no_grad(target)
    
    beta = 1 - torch.mean(target)
    weights = 1 - beta + (2 * beta - 1) * target
    return F.binary_cross_entropy(input, target, weights, size_average=True)

# mean squared error in 2D
def ce2d(input, target):
    # do not compute gradients w.r.t target
    _assert_no_grad(target)

    return F.binary_cross_entropy(input, target, size_average=True)

# utility functions to visualize output
def rgb_trans(data):
    data = data.numpy()
    data = data[0]
    data = data.transpose(1,2,0)
    data *= 255*np.array((0.225, 0.224, 0.229))
    data += np.array([104.00698793, 116.66876762, 122.67891434])
    data = data[:, :, ::-1]
    data = data.astype(np.uint8)
    data = Image.fromarray(data, 'RGB')
    return data

def gray_trans(img):
    img = img.numpy()[0][0]*255
    img = img.astype(np.uint8)
    img = Image.fromarray(img, 'L')
    return img

# utility functions to set the learning rate
def adjust_learning_rate(optimizer, gamma):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= gamma 

def show_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        print(param_group['lr'])

class Trainer(object):

    def __init__(self, generator, optimizerG, trainDataloader, 
                 valDataloader, out='output', maxEpochs=10, cuda=True, gpuID=0):

        # set the GPU flag
        self.cuda = cuda
        self.gpuID = gpuID
        
        # define an optimizer
        self.optimG = optimizerG
        
        # set the network
        self.generator = generator
        
        # set the data loaders
        self.valDataloader = valDataloader
        self.trainDataloader = trainDataloader
        
        # set output directory
        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)
                
        # set training parameters
        self.step = 2
        self.gamma = 0.1
        
        self.epoch = 0

        self.dispInterval = 100
        self.nepochs = maxEpochs
        self.timeformat = '%Y-%m-%d %H:%M:%S'

    def train(self):
        # function to train network
        for epoch in range(self.epoch, self.nepochs):
            # set function to training mode
            self.generator.train()
            
            # initialize gradients
            self.optimG.zero_grad()
            
            # adjust hed learning rate
            if (epoch > 0) and (epoch % self.step) == 0:
                adjust_learning_rate(self.optimG, self.gamma)
            show_learning_rate(self.optimG)
            
            # train the network
            for i, sample in enumerate(self.trainDataloader, 0):
                # get the training batch
                data, target = sample
                
                if self.cuda:
                    data, target = data.cuda(self.gpuID), target.cuda(self.gpuID)
                data, target = Variable(data), Variable(target)
                
                # generator forward
                d1, d2, d3, d4, d5, d6 = self.generator(data, target) 
                
                # compute loss for batch
                loss1 = bce2d(d1, target)
                loss2 = bce2d(d2, target)
                loss3 = bce2d(d3, target)
                loss4 = bce2d(d4, target)
                loss5 = bce2d(d5, target)
                loss6 = ce2d(d6, target)
                
                # all components have equal weightage
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

                if np.isnan(float(loss.data[0])):
                    raise ValueError('loss is nan while training')
                
                # perform backpropogation and update network
                loss.backward()
                self.optimG.step()
                self.optimG.zero_grad()
                
                # visualize the loss
                if (i+1) % self.dispInterval == 0:
                    timestr = time.strftime(self.timeformat, time.localtime())
                    print("%s epoch: %d iter:%d loss:%.6f"%(timestr, epoch, i, loss.data[0]))
                    print("  loss1: %.3f max1:%.4f min1:%.4f"%(loss1.data[0], 
                                                               d1.data.max(), d1.data.min()))
                    print("  loss2: %.3f max2:%.4f min2:%.4f"%(loss2.data[0], 
                                                               d2.data.max(), d2.data.min()))
                    print("  loss3: %.3f max3:%.4f min3:%.4f"%(loss3.data[0], 
                                                               d3.data.max(), d3.data.min()))
                    print("  loss4: %.3f max4:%.4f min4:%.4f"%(loss4.data[0], 
                                                               d4.data.max(), d4.data.min()))
                    print("  loss5: %.3f max5:%.4f min5:%.4f"%(loss5.data[0], 
                                                               d5.data.max(), d5.data.min()))
                    print("  loss6: %.3f max6:%.4f min6:%.4f"%(loss6.data[0], 
                                                               d6.data.max(), d6.data.min()))
            
                # perform validation
                if (i+1) % 500 == 0:
                    self.valnet(epoch+1)
                    torch.save(self.generator.state_dict(), '%s/HED%d.pth' % (self.out, epoch))

    def valnet(self, epoch):
        # eval model on validation set
        print('Evaluation:')
        
        # convert to test mode
        self.generator.eval()
        
        # save the results
        if os.path.exists(self.out + '/epoch' + str(epoch)) == False:
            os.mkdir(self.out + '/epoch' + str(epoch))
        fullDirname = '%s/epoch%d'%(self.out, epoch)
        
        # perform test inference
        for i, sample in enumerate(self.valDataloader, 0):            
            # get the test sample
            data, target = sample
            
            if self.cuda:
                data, target = data.cuda(self.gpuID), target.cuda(self.gpuID)
            data, target = Variable(data), Variable(target)
            
            # perform forward computation
            d1, d2, d3, d4, d5, d6 = self.generator(data, target)
            
            # save the generated results
            rgb_trans( data.data.cpu()).save('%s/%dimg.png' %(fullDirname, i))
            gray_trans(1-d1.data.cpu()).save('%s/%dd1.png' % (fullDirname, i))
            gray_trans(1-d2.data.cpu()).save('%s/%dd2.png' % (fullDirname, i))
            gray_trans(1-d3.data.cpu()).save('%s/%dd3.png' % (fullDirname, i))
            gray_trans(1-d4.data.cpu()).save('%s/%dd4.png' % (fullDirname, i))
            gray_trans(1-d5.data.cpu()).save('%s/%dd5.png' % (fullDirname, i))
            gray_trans(1-d6.data.cpu()).save('%s/%dd6.png' % (fullDirname, i))
        print('evaluate done')
        self.generator.train()