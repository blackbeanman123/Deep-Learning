#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 14:43:50 2020

@author: blackbeanman
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms



# check accuracy on training set and test set(or on the validation set to tune the hyperparameter)
def check_accuracy(loader, model):  # the model correspond to the 
    num_correct = 0
    num_samples = 0
    model.eval()#!!important!! https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    
    with torch.no_grad():#tell it that there is no need to cal the gradient
        for x,y in loader:
            x = x.to(device = device)   #import the data to the device we are using
            y = y.to(device = device)
            
            y_ = model(x)
            _, preds = y_.max(1) #the dimension 1, the _ means this returned variable is unnecessary
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            
        model.train()   #return to the model.train for further training (model.eval()--model.train())
        acc = float(num_correct)/float(num_samples)
        print(acc)
        return acc
    
class AlexNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AlexNet, self).__init__()
        self.over_pool = nn.MaxPool2d(pool_size = 3, stride = 2)    #overlap pooling
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 96, kernel_size = 11, stride = 4, padding = 0)
        self.conv2 = nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, stride = 1, padding = 2)
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride = 1)#2 3 4 -- keep convolving
        self.conv4 = nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, stride = 1)#2 3 4 -- not changing img size
        self.conv5 = nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, stride = 1)
        self.fc1 = nn.Linear(227*227*3, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.out = nn.Softmax(4096, 1000)
    def forward(self,x):
        #conv1
        x = F.relu(self.conv1(x))
        x = self.over_pool(x)
        #conv2
        x = F.relu(self.conv2(x))
        x = self.over_pool(x)#I don't use Local Response Normalization
        #conv3
        x = F.relu(self.conv3(x))
        #this layer doesn't contain max_pool and LRN
        #conv4
        x = F.relu(self.conv4(x))
        #this layer doesn't contain max_pool and LRN
        #conv5
        x = F.relu(self.conv5(x))
        x = self.over_pool(x)
        #fc1, 1.reshape to 1d, 2.dropout
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.4, training = self.training)
        #fc2
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.4, training = self.training)
        #output
        x = self.out(x)
        

if __name__ == '__main__':
    #Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Set hyperparameters:
    in_channels = 3
    num_classes = 1000    #10 classes of output
    learning_rate = 0.001
    batch_size = 4
    num_epochs = 1
    
    
    #Set transform and Load Data
    preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])#why 0.485...
    
    train_set = datasets.ImageNet(root = 'imagenet/', train = True, transform = preprocess, download = True)
    train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True, workers = 2)
    test_set = datasets.ImageNet(root = 'imagenet/', train = False, transform = preprocess, download = True)
    test_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True, workers = 2)
    
    #initialize our NN model called 'model'
    model = AlexNet(in_channels = in_channels, num_classes = num_classes).to(device)

    #Loss and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)

    #Train Network
    for epoch in range(num_epochs):         #for each epoch
        for batch_idx, (x, y) in enumerate(train_loader):   #x is the data, y is the correct label
            #get data to cuda if possible
            x = x.to(device = device)   #import the data to the device we are using
            y = y.to(device = device)   
        
            # forward propagation
            y_ = model(x)       #the hypothesis of x or the score of x
            loss = loss_function(y_ , y)
        
            #backward propagation
            optimizer.zero_grad()       #in every batch we need to set our gradient to zero
            loss.backward()             #perform backward propagation on the loss 
        
            #gradient descent of adam step
            optimizer.step()

    #check accuracy
    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)
        