#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 08:09:14 2020

@author: blackbeanman
"""
#https://engmrk.com/lenet-5-a-classic-cnn-architecture/
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 6, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1, padding = 0)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = 5, stride = 1, padding = 0)
        self.pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)
    
        
    def forward(self,x):
        x = torch.tanh(self.conv1(x))   #C1 layer
        x = self.pool(x)    #C2 layer, pooling + some activation function(tanh in lenet)
        x = torch.tanh(self.conv2(x))   #C3 , my implementation is not the same as lenet; It's a simple conv layer
        x = self.pool(x)    #C4 pooling layer
        x = torch.tanh(self.conv3(x))   #C5 conv 
        x = x.reshape(x.shape[0], -1)  #unroll  
        x = torch.tanh(self.fc1(x))     #C6 FC layer
        x = torch.softmax(self.fc2(x), dim = 1)  #C7 FC layer& output layer, not exactly the same activition function used in lenet
        return x   
#97% accuracy on MNIST
        
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
    
    



if __name__ == '__main__':
    #Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Set hyperparameters:
    in_channels = 1 
    num_classes = 10    #10 classes of output
    learning_rate = 1e-3
    batch_size = 8  #mini batch
    num_epochs = 5

    #Set Transform and Load Data
    train_set = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
    test_set = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False)

    #initialize our NN model called 'model'
    model = LeNet(in_channels = in_channels, num_classes = num_classes).to(device)

    #Loss and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    #Train Network
    for epoch in range(num_epochs):         #for each epoch
        for batch_idx, (x, y) in enumerate(train_loader):   #x is the data, y is the correct label
            #get data to cuda if ppossible
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
        