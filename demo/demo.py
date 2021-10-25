# Reference: https://github.com/VDelv/EEGLearn-Pytorch.git

import numpy as np 
import scipy.io as sio
import torch
import os 

import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader,random_split
from collections import Counter
import torch

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from Utils import *
from Models import *

torch.manual_seed(1234)
np.random.seed(1234)

import warnings
warnings.simplefilter("ignore")

Mean_Images = sio.loadmat("./demo/Sample Data/images.mat")["img"] #corresponding to the images mean for all the seven windows
print(np.shape(Mean_Images)) 
Label = (sio.loadmat("./demo/Sample Data/FeatureMat_timeWin")["features"][:,-1]-1).astype(int) #corresponding to the signal label (i.e. load levels).
print(np.shape(Label)) 
Patient_id = sio.loadmat("./demo/Sample Data/trials_subNums.mat")['subjectNum'][0] #corresponding to the patient id
print(np.shape(Patient_id))


print("Choose among the patient : "+str(np.unique(Patient_id))+ " "+ str(np.unique(Label)))
print(Counter(Label))
choosen_patient = 9
train_part = 0.8
test_part = 0.2
batch_size = 32

EEG = EEGImagesDataset(label=Label[Patient_id==choosen_patient], image=Mean_Images[Patient_id==choosen_patient])
print(len(EEG))







# lengths = [int(len(EEG)*train_part+1), int(len(EEG)*test_part)]
# Train, Test = random_split(EEG, lengths) # random_split(datasets, [train_size, test_size])

# Trainloader = DataLoader(Train,batch_size=batch_size)
# Testloader = DataLoader(Test, batch_size=batch_size)
# res = TrainTest_Model(BasicCNN, Trainloader, Testloader, n_epoch=50, learning_rate=0.001, print_epoch=-1, opti='Adam')