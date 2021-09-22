#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
from os.path import join
import imageio
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os,sys,os.path
import pandas as pd
import pickle
import pydicom
import glob
import collections
import pprint
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import warnings
import tarfile
import zipfile
import random
from skimage.io import imread, imsave
from skimage.transform import resize
import hashlib
import pickle
from tqdm import tqdm
import time
import math
from torch.utils.data import Dataset
import argparse
from sklearn.cluster import KMeans
import scipy
import torch.nn as nn
import torch.optim as optim
import copy
from scipy.optimize import linear_sum_assignment as linear_assignment




class ConvEncoder(nn.Module):
    def __init__(self,
                 in_channels=1,
                 input_shape=96,
                 code_length=128, 
                 drop_rate=0,
                 ):
        super(ConvEncoder,self).__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.code_length = code_length
        self.drop_rate = drop_rate
        # Convolutonal Layer 1 
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(in_channels,32, stride = 2, kernel_size = [7, 7], padding = 3, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(drop_rate, inplace=True)
            )
        self.size = int((self.input_shape - 7 + 3*2)/2 + 1)
        self.size = int((self.size - 3)/2+1)
        # Convolutonal Layer 2 
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(32,64, stride = 2, kernel_size = [5, 5],padding = 2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(drop_rate, inplace=True)
            )
        self.size = int((self.size - 5 + 2*2)/2 + 1)
        # Convolutonal Layer 3 
        self.convlayer3 = nn.Sequential(
            nn.Conv2d(64,128, stride = 2, kernel_size = [3, 3],padding = 1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Dropout(drop_rate, inplace=True)
            )
        self.size = int((self.size - 3 + 2*1)/2 + 1)
        # Dense Layer 1
        self.denselayer1 = nn.Sequential(
            nn.Linear(128*self.size*self.size,self.code_length),
        )
        self.out_channels = 128

    def forward(self,x): #x = [batch,time,freq]
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x = torch.flatten(x,start_dim=1)
        code = self.denselayer1(x)
        return code
    

class ConvDecoder(nn.Module):
    def __init__(self,
                 in_channels=128,
                 input_size=6,
                 code_length=128, 
                 drop_rate=0,
                 ):
        self.in_channels = in_channels
        self.input_size = input_size
        self.code_length = code_length
        self.drop_rate = drop_rate
        super(ConvDecoder,self).__init__()
        # Dense Layer 2
        self.denselayer2 = nn.Sequential(
            nn.Linear(code_length,128*self.input_size*self.input_size),
        )
        # Convolutonal Layer 4 
        self.convlayer4 = nn.Sequential(
            nn.ConvTranspose2d(128,64, stride = 2, kernel_size = [3, 3], padding = 1, output_padding = 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(drop_rate, inplace=True)
            )
        # Convolutonal Layer 5 
        self.convlayer5 = nn.Sequential(
            nn.ConvTranspose2d(64,32, stride = 2, kernel_size = [5, 5], padding = 2,output_padding = 1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(drop_rate, inplace=True)
            )
        # Convolutonal Layer 6
        self.convlayer6 = nn.Sequential(
            nn.ConvTranspose2d(32,32, stride = 2, kernel_size = [7, 7], padding = 3, output_padding = 1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout(drop_rate, inplace=True)
            )
        
         # Convolutonal Layer 7
        self.convlayer7 = nn.Sequential(
            nn.ConvTranspose2d(32,1, stride = 2, kernel_size = [7, 7], padding = 3, output_padding = 1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Dropout(drop_rate, inplace=True)
            )
    def forward(self,code): #code = [batch,code_length]
        x = self.denselayer2(code)
        batch = x.shape[0]
        x = torch.reshape(x,(batch,128,self.input_size,self.input_size))
        x = self.convlayer4(x)
        x = self.convlayer5(x)
        x = self.convlayer6(x)
        x = self.convlayer7(x)
        return x
    
class LogicLayer(nn.Module):
    def __init__(self,
                 n_classes=2,
                 code_length=128,
                 ):
        super(LogicLayer, self).__init__()
        self.n_classes = n_classes
        self.code_length = code_length
        self.denselayer3 = nn.Sequential(
            nn.Linear(self.code_length, self.n_classes),
            nn.Sigmoid(),
        )  
    def forward(self, code):
        p = self.denselayer3(code)
        return p
    
    
    
class LogicEncoder(nn.Module):
    def __init__(self,
                 in_channels=1,
                 input_shape=96,
                 code_length=128, 
                 drop_rate=0,
                 n_classes=2,
                 decoder=False,
                 ):
        super(LogicEncoder,self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.code_length = code_length
        self.drop_rate = drop_rate
        self.input_shape = input_shape
        self.decoder = decoder
        
        
        self.Encoder = ConvEncoder(self.in_channels,self.input_shape,self.code_length,self.drop_rate)
        in_channels = self.Encoder.out_channels
        input_shape = self.Encoder.size
        
        self.Decoder = ConvDecoder(in_channels,input_shape,self.code_length,self.drop_rate)
        
        self.LogicLayer = LogicLayer(self.n_classes,self.code_length)

    def forward(self,x): #x = [batch,time,freq]
        code = self.Encoder(x)
        p = self.LogicLayer(code)       
        if self.decoder == False:
            return p
        else:
            _x = self.Decoder(code)
            return _x,p

    def setDecoder(self, b):
        self.decoder = b
        return  

    
class ConvEncoder(nn.Module):
    def __init__(self,
                 in_channels=1,
                 input_shape=[96,96],
                 code_length=128, 
                 classes = 2,
                 drop_rate=0,
                 ):
        super(ConvEncoder,self).__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.code_length = code_length
        self.classes = classes
        self.drop_rate = drop_rate
        # Convolutonal Layer 1 
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(in_channels,32, stride = 2, kernel_size = [7, 7], padding = 3, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(drop_rate, inplace=True)
            )
        self.size = int((input_shape[0] - 7 + 3*2)/2 + 1)
        self.size = int((self.size - 3)/2+1)
        # Convolutonal Layer 2 
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(32,64, stride = 2, kernel_size = [5, 5],padding = 2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout(drop_rate, inplace=True)
            )
        self.size = int((self.size - 5 + 2*2)/2 + 1)
        # Convolutonal Layer 3 
        self.convlayer3 = nn.Sequential(
            nn.Conv2d(64,128, stride = 2, kernel_size = [3, 3],padding = 1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Dropout(drop_rate, inplace=True)
            )
        self.size = int((self.size - 3 + 2*1)/2 + 1)
        # Dense Layer 1
        self.denselayer1 = nn.Sequential(
            nn.Linear(128*self.size*self.size,self.code_length),
            nn.Linear(self.code_length, self.classes),
            nn.Sigmoid(),
        )

    def forward(self,x): #x = [batch,time,freq]
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x = torch.flatten(x,start_dim=1)
        p = self.denselayer1(x)
        return p
    
    
    
class CIFAR_ConvEncoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 input_shape=[32,32],
                 code_length=128, 
                 classes = 10,
                 drop_rate=0.2,
                 ):
        super(CIFAR_ConvEncoder,self).__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.code_length = code_length
        self.classes = classes
        self.drop_rate = drop_rate
        # Convolutonal Layer 1 
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(in_channels,32, kernel_size = 3, padding = 1, bias=False),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            )
        
        # Convolutonal Layer 2 
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(32,32, kernel_size = 3,padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.drop_rate, inplace=True)
            )

        # Convolutonal Layer 3 
        self.convlayer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size = 3,padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            )
        
        # Convolutonal Layer 4
        self.convlayer4 = nn.Sequential(
            nn.Conv2d(64,64, kernel_size = 3,padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size =2, stride=2),
            nn.Dropout(self.drop_rate, inplace=True)
            )
        
        
        # Dense Layer 1
        self.denselayer1 = nn.Sequential(
            nn.Linear(64*8*8, self.code_length),
        )

    def forward(self,x): #x = [batch,time,freq]
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x = self.convlayer4(x)
        x = torch.flatten(x,start_dim=1)
        code = self.denselayer1(x)
        return code

class CIFAR_ConvDecoder(nn.Module):
    def __init__(self,
                 in_channels=64,
                 input_shape=[8,8],
                 code_length=128, 
                 classes = 10,
                 drop_rate=0.2,
                 ):
        super(CIFAR_ConvDecoder,self).__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.code_length = code_length
        self.classes = classes
        self.drop_rate = drop_rate
        
        # Dense Layer 2
        self.denselayer2 = nn.Sequential(
            nn.Linear(code_length,self.in_channels*self.input_shape[0]*self.input_shape[1]),
        )

        # Transpose Convolutonal Layer 1 
        self.deconvlayer1 = nn.Sequential(
            nn.ConvTranspose2d(64,64, kernel_size = 3,padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            )
        
        # Transpose Convolutonal Layer 2
        self.deconvlayer2 = nn.Sequential(
            nn.ConvTranspose2d(64,32, kernel_size = 3,stride=2,padding = 1,output_padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(self.drop_rate, inplace=True)
            )
        
         # Transpose Convolutonal Layer 3 
        self.deconvlayer3 = nn.Sequential(
            nn.ConvTranspose2d(32,32, kernel_size = 3,padding = 1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            )
        
        # Transpose Convolutonal Layer 4
        self.deconvlayer4 = nn.Sequential(
            nn.ConvTranspose2d(32,3, kernel_size = 3,stride=2,padding = 1,output_padding = 1),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Dropout(self.drop_rate, inplace=True)
            )
        

    def forward(self,code): #x = [batch,time,freq]
        x = self.denselayer2(code)
        batch = x.shape[0]
        x = torch.reshape(x,(batch,self.in_channels,self.input_shape[0],self.input_shape[1]))
        x = self.deconvlayer1(x)
        x = self.deconvlayer2(x)
        x = self.deconvlayer3(x)
        x = self.deconvlayer4(x)
        return x
    
class CIFAR_LogicEncoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 input_shape=[32,32],
                 code_length=128, 
                 classes = 10,
                 drop_rate=0.2,
                 decoder = True
                 ):
        super(CIFAR_LogicEncoder,self).__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.code_length = code_length
        self.classes = classes
        self.drop_rate = drop_rate
        self.size = input_shape
        self.decoder = decoder
        
        self.Encoder = CIFAR_ConvEncoder(in_channels=in_channels,input_shape=input_shape,code_length=code_length, classes = classes,drop_rate=drop_rate)
        self.Decoder = CIFAR_ConvDecoder(in_channels=64,input_shape=[8,8],code_length=code_length, classes = classes,drop_rate=drop_rate)
        self.Classifier = nn.Linear(code_length,classes)
 
    def forward(self,x): #x = [batch,time,freq]
        code = self.Encoder(x)
        if self.decoder== True:
            _X = self.Decoder(code)
            return _X
        else:
            _X = self.Decoder(code)
            p = self.Classifier(code)
            return _X,p

    def SetDecoder(self,b):
        self.decoder = b
        return
    
class CIFAR_CNN(nn.Module):
    def __init__(self,
                 in_channels=3,
                 input_shape=[32,32],
                 code_length=128, 
                 classes = 10,
                 drop_rate=0.2,
                 ):
        super(CIFAR_CNN,self).__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.code_length = code_length
        self.classes = classes
        self.drop_rate = drop_rate
        
        self.Encoder = CIFAR_ConvEncoder(in_channels=in_channels,input_shape=input_shape,code_length=code_length, classes = classes,drop_rate=drop_rate)
        self.Classifier = nn.Linear(self.code_length,self.classes)
#         # Convolutonal Layer 1 
#         self.convlayer1 = nn.Sequential(
#             nn.Conv2d(in_channels,32, kernel_size = 3, padding = 1, bias=False),
#             nn.PReLU(),
#             nn.BatchNorm2d(32),
#             )
        
#         # Convolutonal Layer 2 
#         self.convlayer2 = nn.Sequential(
#             nn.Conv2d(32,32, kernel_size = 3,padding = 1),
#             nn.PReLU(),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(self.drop_rate, inplace=True)
#             )

#         # Convolutonal Layer 3 
#         self.convlayer3 = nn.Sequential(
#             nn.Conv2d(32,64, kernel_size = 3,padding = 1),
#             nn.PReLU(),
#             nn.BatchNorm2d(64),
#             )
        
#         # Convolutonal Layer 4
#         self.convlayer4 = nn.Sequential(
#             nn.Conv2d(64,64, kernel_size = 3,padding = 1),
#             nn.PReLU(),
#             nn.BatchNorm2d(64),
#             nn.AvgPool2d(kernel_size =2, stride=2),
#             nn.Dropout(self.drop_rate, inplace=True)
#             )
        
        
#         # Dense Layer 1
#         self.denselayer1 = nn.Sequential(
#             nn.Linear(64*8*8, self.code_length),
#         )

    def forward(self,x): #x = [batch,time,freq]
        code = self.Encoder(x)
        p = self.Classifier(code)
        return p
    
    
    
class CIFAR_CVAE_Encoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 input_shape=[32,32],
                 code_length=10, 
                 classes = 10,
                 drop_rate=0.2,
                 ):
        super(CIFAR_CVAE_Encoder,self).__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.code_length = code_length
        self.classes = classes
        self.drop_rate = drop_rate
       # Convolutonal Layer 1 
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(in_channels,32, kernel_size = 3, padding = 1, bias=False),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            )
        
        # Convolutonal Layer 2 
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(32,32, kernel_size = 3,padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.drop_rate, inplace=True)
            )

        # Convolutonal Layer 3 
        self.convlayer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size = 3,padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            )
        
        # Convolutonal Layer 4
        self.convlayer4 = nn.Sequential(
            nn.Conv2d(64,64, kernel_size = 3,padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size =2, stride=2),
            nn.Dropout(self.drop_rate, inplace=True)
            )
        
    
    def forward(self,x): #x = [batch,time,freq]
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x = self.convlayer4(x)
        x = torch.flatten(x,start_dim=1)
        return x

class CIFAR_CVAE_Decoder(nn.Module):
    def __init__(self,
                 in_channels=64,
                 input_shape=[8,8],
                 code_length=10, 
                 classes = 10,
                 drop_rate=0.2,
                 ):
        super(CIFAR_CVAE_Decoder,self).__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.code_length = code_length
        self.classes = classes
        self.drop_rate = drop_rate
        
        # Dense Layer 2
        self.denselayer = nn.Sequential(
            nn.Linear(code_length+classes,self.in_channels*self.input_shape[0]*self.input_shape[1]),
        )

        # Transpose Convolutonal Layer 1 
        self.deconvlayer1 = nn.Sequential(
            nn.ConvTranspose2d(64,64, kernel_size = 3,padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            )
        
        # Transpose Convolutonal Layer 2
        self.deconvlayer2 = nn.Sequential(
            nn.ConvTranspose2d(64,32, kernel_size = 3,stride=2,padding = 1,output_padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            )
        
         # Transpose Convolutonal Layer 3 
        self.deconvlayer3 = nn.Sequential(
            nn.ConvTranspose2d(32,32, kernel_size = 3,padding = 1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            )
        
        # Transpose Convolutonal Layer 4
        self.deconvlayer4 = nn.Sequential(
            nn.ConvTranspose2d(32,3, kernel_size = 3,stride=2,padding = 1,output_padding = 1),
            nn.Sigmoid(),
            )

    def forward(self,z): #x = [batch,time,freq]
        x = self.denselayer(z)
        batch = x.shape[0]
        x = torch.reshape(x,(batch,self.in_channels,self.input_shape[0],self.input_shape[1]))
        x = self.deconvlayer1(x)
        x = self.deconvlayer2(x)
        x = self.deconvlayer3(x)
        x = self.deconvlayer4(x)
        return x
    
class CIFAR_CVAE(nn.Module):
    def __init__(self,
                 in_channels=3,
                 input_shape=[32,32],
                 code_length=128, 
                 classes = 10,
                 drop_rate=0.2,
                 decoder = False,
                 ):
        super(CIFAR_CVAE,self).__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.code_length = code_length
        self.classes = classes
        self.drop_rate = drop_rate
        self.size = input_shape
        self.decoder = decoder
        
        self.Encoder = CIFAR_CVAE_Encoder(in_channels=in_channels,input_shape=input_shape,code_length=code_length, classes = classes,drop_rate=drop_rate)
        self.Decoder = CIFAR_CVAE_Decoder(in_channels=64,input_shape=[8,8],code_length=code_length, classes = classes,drop_rate=drop_rate)             
        self.Mu = nn.Sequential(
            nn.Linear(64*8*8, self.code_length),
        )
        
         # FC Layer 2
        self.LogVar = nn.Sequential(
            nn.Linear(64*8*8, self.code_length),
        )
        # FC Layer 3
        self.Classifier = nn.Sequential(
            nn.Linear(64*8*8, 32),
            nn.Linear(32,classes),
        )
        
        
    def reparameterize(self, mu, logVar):
        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps
    

    
    def forward(self,x): #x = [batch,time,freq]
        x = self.Encoder(x)
        categorical = self.Classifier(x)
        if self.decoder == False:
            return categorical
        else:
            mu = self.Mu(x)
            logVar  = self.LogVar(x)
            z = self.reparameterize(mu,logVar)
            c = F.softmax(categorical)
            z = torch.cat([z,c], 1)
            _x = self.Decoder(z)
            return mu,logVar,categorical,_x
        
class CIFAR_VAE_Encoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 input_shape=[32,32],
                 code_length=10, 
                 classes = 10,
                 drop_rate=0.2,
                 ):
        super(CIFAR_VAE_Encoder,self).__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.code_length = code_length
        self.classes = classes
        self.drop_rate = drop_rate
       # Convolutonal Layer 1 
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(in_channels,32, kernel_size = 3, padding = 1, bias=False),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            )
        
        # Convolutonal Layer 2 
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(32,32, kernel_size = 3,padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.drop_rate, inplace=True)
            )

        # Convolutonal Layer 3 
        self.convlayer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size = 3,padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            )
        
        # Convolutonal Layer 4
        self.convlayer4 = nn.Sequential(
            nn.Conv2d(64,64, kernel_size = 3,padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size =2, stride=2),
            nn.Dropout(self.drop_rate, inplace=True)
            )
        
    
    def forward(self,x): #x = [batch,time,freq]
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x = self.convlayer4(x)
        x = torch.flatten(x,start_dim=1)
        return x

class CIFAR_VAE_Decoder(nn.Module):
    def __init__(self,
                 in_channels=64,
                 input_shape=[8,8],
                 code_length=10, 
                 classes = 10,
                 drop_rate=0.2,
                 ):
        super(CIFAR_VAE_Decoder,self).__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.code_length = code_length
        self.classes = classes
        self.drop_rate = drop_rate
        
        # Dense Layer 2
        self.denselayer = nn.Sequential(
            nn.Linear(code_length,self.in_channels*self.input_shape[0]*self.input_shape[1]),
        )

        # Transpose Convolutonal Layer 1 
        self.deconvlayer1 = nn.Sequential(
            nn.ConvTranspose2d(64,64, kernel_size = 3,padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            )
        
        # Transpose Convolutonal Layer 2
        self.deconvlayer2 = nn.Sequential(
            nn.ConvTranspose2d(64,32, kernel_size = 3,stride=2,padding = 1,output_padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            )
        
         # Transpose Convolutonal Layer 3 
        self.deconvlayer3 = nn.Sequential(
            nn.ConvTranspose2d(32,32, kernel_size = 3,padding = 1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            )
        
        # Transpose Convolutonal Layer 4
        self.deconvlayer4 = nn.Sequential(
            nn.ConvTranspose2d(32,3, kernel_size = 3,stride=2,padding = 1,output_padding = 1),
            nn.Sigmoid(),
            )

    def forward(self,z): #x = [batch,time,freq]
        x = self.denselayer(z)
        batch = x.shape[0]
        x = torch.reshape(x,(batch,self.in_channels,self.input_shape[0],self.input_shape[1]))
        x = self.deconvlayer1(x)
        x = self.deconvlayer2(x)
        x = self.deconvlayer3(x)
        x = self.deconvlayer4(x)
        return x
    
class CIFAR_VAE(nn.Module):
    def __init__(self,
                 in_channels=3,
                 input_shape=[32,32],
                 code_length=128, 
                 classes = 10,
                 drop_rate=0.2,
                 decoder = False,
                 ):
        super(CIFAR_VAE,self).__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.code_length = code_length
        self.classes = classes
        self.drop_rate = drop_rate
        self.size = input_shape
        self.decoder = decoder
        
        self.Encoder = CIFAR_VAE_Encoder(in_channels=in_channels,input_shape=input_shape,code_length=code_length, classes = classes,drop_rate=drop_rate)
        self.Decoder = CIFAR_VAE_Decoder(in_channels=64,input_shape=[8,8],code_length=code_length, classes = classes,drop_rate=drop_rate)             
        self.Mu = nn.Sequential(
            nn.Linear(64*8*8, self.code_length),
        )
        
         # FC Layer 2
        self.LogVar = nn.Sequential(
            nn.Linear(64*8*8, self.code_length),
        )
        # FC Layer 3
        self.Classifier = nn.Sequential(
            nn.Linear(64*8*8, 32),
            nn.Linear(32,classes),
        )
        
        
    def reparameterize(self, mu, logVar):
        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std*eps
    

    
    def forward(self,x): #x = [batch,time,freq]
        x = self.Encoder(x)
        categorical = self.Classifier(x)
        if self.decoder == False:
            return categorical
        else:
            mu = self.Mu(x)
            logVar = self.LogVar(x)
            z = self.reparameterize(mu,logVar)
            _x = self.Decoder(z)
            return mu,logVar,categorical,_x
        
class MNIST_VAE_Encoder(nn.Module):
    def __init__(self,
                 in_channels=1,
                 input_shape=[28,28],
                 code_length=32, 
                 classes = 10,
                 drop_rate=0.2,
                 ):
        super(MNIST_VAE_Encoder,self).__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.code_length = code_length
        self.classes = classes
        self.drop_rate = drop_rate
       # Convolutonal Layer 1 
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1,out_channels=32, kernel_size = 3, padding = 1, bias=False),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            )
        
        # Convolutonal Layer 2 
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(32,32, kernel_size = 3,padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.drop_rate, inplace=True)
            )

        # Convolutonal Layer 3 
        self.convlayer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size = 3,padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            )
        
        # Convolutonal Layer 4
        self.convlayer4 = nn.Sequential(
            nn.Conv2d(64,64, kernel_size = 3,padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size =2, stride=2),
            nn.Dropout(self.drop_rate, inplace=True)
            )

    def forward(self,x): #x = [batch,time,freq]
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x = self.convlayer4(x)
        x = torch.flatten(x,start_dim=1)
        return x

class MNIST_VAE_Decoder(nn.Module):
    def __init__(self,
                 in_channels=64,
                 input_shape=[7,7],
                 code_length=32, 
                 classes = 10,
                 drop_rate=0.2,
                 ):
        super(MNIST_VAE_Decoder,self).__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.code_length = code_length
        self.classes = classes
        self.drop_rate = drop_rate
        
        # Dense Layer 2
        self.denselayer = nn.Sequential(
            nn.Linear(code_length,self.in_channels*self.input_shape[0]*self.input_shape[1]),
        )

        # Transpose Convolutonal Layer 1 
        self.deconvlayer1 = nn.Sequential(
            nn.ConvTranspose2d(64,64, kernel_size = 3,padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            )
        
        # Transpose Convolutonal Layer 2
        self.deconvlayer2 = nn.Sequential(
            nn.ConvTranspose2d(64,32, kernel_size = 3,stride=2,padding = 1,output_padding = 1),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            )
        
         # Transpose Convolutonal Layer 3 
        self.deconvlayer3 = nn.Sequential(
            nn.ConvTranspose2d(32,32, kernel_size = 3,padding = 1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            )
        
        # Transpose Convolutonal Layer 4
        self.deconvlayer4 = nn.Sequential(
            nn.ConvTranspose2d(32,1, kernel_size = 3,stride=2,padding = 1,output_padding = 1),
            nn.Sigmoid(),
            )

    def forward(self,z): #x = [batch,time,freq]
        x = self.denselayer(z)
        batch = x.shape[0]
        x = torch.reshape(x,(batch,self.in_channels,self.input_shape[0],self.input_shape[1]))
        x = self.deconvlayer1(x)
        x = self.deconvlayer2(x)
        x = self.deconvlayer3(x)
        x = self.deconvlayer4(x)
        return x
        
class MNIST_VAE(nn.Module):
    def __init__(self,
                 in_channels=3,
                 input_shape=[32,32],
                 code_length=32, 
                 classes = 10,
                 drop_rate=0.2,
                 decoder = False,
                 ):
        super(MNIST_VAE,self).__init__()
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.code_length = code_length
        self.classes = classes
        self.drop_rate = drop_rate
        self.size = input_shape
        self.decoder = decoder
        
        self.Encoder = MNIST_VAE_Encoder(in_channels=in_channels,input_shape=input_shape,code_length=code_length, classes = classes,drop_rate=drop_rate)
        self.Decoder = MNIST_VAE_Decoder(in_channels=64,input_shape=[7,7],code_length=code_length, classes = classes,drop_rate=drop_rate)             
        self.Mu = nn.Sequential(
            nn.Linear(64*7*7, self.code_length),
        )
        
         # FC Layer 2
        self.LogVar = nn.Sequential(
            nn.Linear(64*7*7, self.code_length),
        )
        # FC Layer 3
        self.Classifier = nn.Sequential(
            nn.Linear(64*7*7, 32),
            nn.Linear(32,classes),
        )
        
        
    def reparameterize(self, mu, logVar):
        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std*eps
    

    
    def forward(self,x): #x = [batch,time,freq]
        x = self.Encoder(x)
        categorical = self.Classifier(x)
        if self.decoder == False:
            return categorical
        else:
            mu = self.Mu(x)
            logVar = self.LogVar(x)
            z = self.reparameterize(mu,logVar)
            _x = self.Decoder(z)
            return mu,logVar,categorical,_x