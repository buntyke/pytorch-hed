# import libraries
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# function to crop the padding pixels
def crop(d, g):
    g_h, g_w = g.size()[2:4]
    d_h, d_w = d.size()[2:4]
    d1 = d[:, :, int(math.floor((d_h - g_h)/2.0)):int(math.floor((d_h - g_h)/2.0)) + g_h, int(math.floor((d_w - g_w)/2.0)):int(math.floor((d_w - g_w)/2.0)) + g_w]
    return d1

# VGG 16 Features Model definition
class VGG16features(nn.Module):
    def __init__(self, activation=F.relu):
        super(VGG16features, self).__init__()

        self.activation = activation

        self.init_pad = torch.nn.ReflectionPad2d(32)

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=(33, 33))
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.activation(x)
        x = self.conv1_2(x)
        c1 = self.activation(x)

        x = self.pool1(c1)

        x = self.conv2_1(x)
        x = self.activation(x)
        x = self.conv2_2(x)
        c2 = self.activation(x)

        x = self.pool2(c2)

        x = self.conv3_1(x)
        x = self.activation(x)
        x = self.conv3_2(x)
        x = self.activation(x)
        x = self.conv3_3(x)
        c3 = self.activation(x)

        x = self.pool3(c3)

        x = self.conv4_1(x)
        x = self.activation(x)
        x = self.conv4_2(x)
        x = self.activation(x)
        x = self.conv4_3(x)
        c4 = self.activation(x)

        x = self.pool4(c4)

        x = self.conv5_1(x)
        x = self.activation(x)
        x = self.conv5_2(x)
        x = self.activation(x)
        x = self.conv5_3(x)
        c5 = self.activation(x)

        x = self.pool5(c5)

        return x

    def forward_hypercol(self, x):
        x = self.conv1_1(x)
        x = self.activation(x)
        x = self.conv1_2(x)
        c1 = self.activation(x)

        x = self.pool1(c1)

        x = self.conv2_1(x)
        x = self.activation(x)
        x = self.conv2_2(x)
        c2 = self.activation(x)

        x = self.pool2(c2)

        x = self.conv3_1(x)
        x = self.activation(x)
        x = self.conv3_2(x)
        x = self.activation(x)
        x = self.conv3_3(x)
        c3 = self.activation(x)

        x = self.pool3(c3)

        x = self.conv4_1(x)
        x = self.activation(x)
        x = self.conv4_2(x)
        x = self.activation(x)
        x = self.conv4_3(x)
        c4 = self.activation(x)

        x = self.pool4(c4)

        x = self.conv5_1(x)
        x = self.activation(x)
        x = self.conv5_2(x)
        x = self.activation(x)
        x = self.conv5_3(x)
        c5 = self.activation(x)

        return c1, c2, c3, c4, c5

# VGG model definition
class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# load pre-trained VGG-16 model
def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    VGG16fs = VGG16features()
    model = VGG(VGG16fs, **kwargs)
    if pretrained:
        state_dict = torch.utils.model_zoo.load_url(torchvision.models.vgg.model_urls['vgg16'])
        new_state_dict = {}

        original_layer_ids = set()
        # copy the classifier entries and make a mapping for the feature mappings
        for key in state_dict.keys():
            if 'classifier' in key:
                new_state_dict[key] = state_dict[key]
            elif 'features' in key:
                original_layer_ids.add(int(key.split('.')[1]))
        sorted_original_layer_ids = sorted(list(original_layer_ids))

        layer_ids = set()
        for key in model.state_dict().keys():
            if 'classifier' in key:
                continue
            elif 'features' in key:
                layer_id = key.split('.')[1]
                layer_ids.add(layer_id)
        sorted_layer_ids = sorted(list(layer_ids))

        for key, value in state_dict.items():
            if 'features' in key:
                original_layer_id = int(key.split('.')[1])
                original_param_id = key.split('.')[2]
                idx = sorted_original_layer_ids.index(original_layer_id)
                new_layer_id = sorted_layer_ids[idx]
                new_key = 'features.' + new_layer_id + '.' + original_param_id
                new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)
    return model, VGG16fs

# definition of HED module
class HED(nn.Module):
    def __init__(self, dilation=0):
        # define VGG architecture and layers
        super(HED, self).__init__()
        
        # define fully-convolutional layers
        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(128, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn5 = nn.Conv2d(512, 1, 1)
        self.dsn6 = nn.Conv2d(5, 1, 1)
        
        # define upsampling/deconvolutional layers
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')              
        
        # initialize weights of layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        _, VGG16fs = vgg16(pretrained=True)
        self.VGG16fs = VGG16fs

    # define the computation graph
    def forward(self, x, gt):

        # get output from VGG model
        conv1, conv2, conv3, conv4, conv5 = self.VGG16fs.forward_hypercol(x)

        ## side output
        dsn5_up = self.upscore5(self.dsn5(conv5))
        d5 = crop(dsn5_up, gt)
        
        dsn4_up = self.upscore4(self.dsn4(conv4))
        d4 = crop(dsn4_up, gt)
        
        dsn3_up = self.upscore3(self.dsn3(conv3))
        d3 = crop(dsn3_up, gt)
        
        dsn2_up = self.upscore2(self.dsn2(conv2))
        d2 = crop(dsn2_up, gt)
        
        dsn1 = self.dsn1(conv1)
        d1 = crop(dsn1, gt)

        # equally weighted fusion
        d6 = 0.2*d1 + 0.2*d2 + 0.2*d3 + 0.2*d4 + 0.2*d5
        
        d1 = F.sigmoid(d1)
        d2 = F.sigmoid(d2)
        d3 = F.sigmoid(d3)
        d4 = F.sigmoid(d4)
        d5 = F.sigmoid(d5)
        d6 = F.sigmoid(d6)

        return d1, d2, d3, d4, d5, d6