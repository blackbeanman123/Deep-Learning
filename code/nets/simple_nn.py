#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 16:53:35 2020

@author: blackbeanman
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# the class of the plain NN
class NN(nn.Module):     #inherit the class nn.Module
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()  #call the parent constructor
        # the definition of different layers
        self.FC1 = nn.Linear(input_size, 30)
        self.FC2 = nn.Linear(30, num_classes)
        
    def forward(self, x):   #put your forward propagation structure here
        x = F.relu(self.FC1(x))
        x = F.softmax(self.FC2(x), dim = 1)
        #x = self.FC2(x)
        return x
    
    
        
        
# check accuracy on training set and test set(or on the validation set to tune the hyperparameter)
def check_accuracy(loader, model):  # the model correspond to the 
    num_correct = 0
    num_samples = 0
    model.eval()#!!important!! https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    
    with torch.no_grad():#tell it that there is no need to cal the gradient
        for x,y in loader:
            x = x.to(device = device)   #import the data to the device we are using
            y = y.to(device = device)
            x = x.reshape(x.shape[0],-1)
            
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
    input_size = 784 #the img is 28x28
    num_classes = 10    #10 classes of output
    learning_rate = 0.001
    batch_size = 4
    num_epochs = 1
    
    
    #Set transform and Load Data
    train_set = datasets.MNIST(root = 'dataset/', train = True, transform = transforms.ToTensor(), download = True)
    train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
    test_set = datasets.MNIST(root = 'dataset/', train = False, transform = transforms.ToTensor(), download = True)
    test_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle = True)

    #initialize our NN model called 'model'
    model = NN(input_size = input_size, num_classes = num_classes).to(device)

    #Loss and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    #Train Network
    for epoch in range(num_epochs):         #for each epoch
        for batch_idx, (x, y) in enumerate(train_loader):   #x is the data, y is the correct label
            #get data to cuda if ppossible
            x = x.to(device = device)   #import the data to the device we are using
            y = y.to(device = device)   
        
            # unroll the images to vectors for training
            x = x.reshape(x.shape[0], -1)
        
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
        