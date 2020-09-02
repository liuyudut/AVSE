import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class MLP_D(nn.Module):
    def __init__(self, opt): 
        super(MLP_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h


class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h


class MLP_G_Att_Adaptive(nn.Module):
    def __init__(self, opt):
        super(MLP_G_Att_Adaptive, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h


class MLP_G_Img_Adaptive(nn.Module):
    def __init__(self, opt):
        super(MLP_G_Img_Adaptive, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.logic = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, res):
        h = self.lrelu(self.fc1(res))
        h = self.relu(self.fc2(h))
        return h


class MLP_E_Adaptive(nn.Module):
    def __init__(self, opt):
        super(MLP_E_Adaptive, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.attSize)
        self.logic = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, res):
        h = self.relu(self.fc1(res))

        return h

#######################################################

class MLP_R_Img(nn.Module):
    def __init__(self, opt):
        super(MLP_R_Img, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.resSize)
        self.logic = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, res):
        h = self.relu(self.fc1(res))

        return h


class MLP_Image_Encoder(nn.Module):
    def __init__(self, opt):
        super(MLP_Image_Encoder, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.logic = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, res):
        h = self.lrelu(self.fc1(res))
        h = self.relu(self.fc2(h))
        return h


class MLP_Att_Encoder(nn.Module):
    def __init__(self, opt):
        super(MLP_Att_Encoder, self).__init__()
        self.fc1 = nn.Linear(opt.attSize, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.logic = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, att):
        h = self.lrelu(self.fc1(att))
        h = self.relu(self.fc2(h))
        return h

