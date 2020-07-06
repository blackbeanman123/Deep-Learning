#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:52:15 2020

@author: blackbeanman
"""
import numpy as np
import cv2 as cv

def convolve(kernel, img):
    #add zero padding to convolve the features at the border
    img = cv.copyMakeBorder(img,1,1,1,1,cv.BORDER_CONSTANT,value=[0,0,0])

    #turn the kernel upside-down to perform convolve mathmatically
    #in CNN, whether we turn the kernel upside-down doesn't hurt
    kernel = np.flip(kernel)
    
    size = (np.shape(img)[0], np.shape(img)[1])
    
    new_img = np.zeros(np.shape(img))
    new_img.fill(255)
    
    #we won't change the size of the picture,so 3x3 stride 1
    #I have no idea about how to vectorlization the process
    #Winograd algorithm can help us cal it
    for i in range(1,size[0]-1):#[)
        for j in range(1,size[1]-1):     
            for k in range(0,3):
                sub_img = img[i-1:i+2,j-1:j+2,k]    
                new_img[i][j][k] = np.sum(kernel*sub_img)
    cv.imwrite('Fuzzification.png',new_img)
    
    

kernel1 = np.array([[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1]])#Fuzzification
kernel2 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])#Finds horizontals
kernel3 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])#Finds verticals
img = cv.imread('icon.png')
convolve(kernel1,img)
