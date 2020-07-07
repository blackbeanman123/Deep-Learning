#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:52:15 2020

@author: blackbeanman
"""
import numpy as np
import cv2 as cv

def convolve(kernel, img):
    new_img = np.zeros(np.shape(img))
    new_img.fill(255)
    #add zero padding to convolve the features at the border
    img = cv.copyMakeBorder(img,1,1,1,1,cv.BORDER_CONSTANT,value=[0,0,0])

    #turn the kernel upside-down to perform convolve mathmatically
    #in CNN, whether we turn the kernel upside-down doesn't hurt
    kernel = np.flip(kernel)
    
    size = (np.shape(img)[0], np.shape(img)[1])
    
    
    #we won't change the size of the picture,so 3x3 stride 1
    #I have no idea about how to vectorlization the process
    #Winograd algorithm can help us cal it
    for i in range(1,size[0]-1):#[)
        for j in range(1,size[1]-1):     
                sub_img = np.transpose(img[i-1:i+2,j-1:j+2,0:3])    #plain slice wouldn't slice the picture the way we want
                new_img[i-1][j-1] = np.sum(kernel*sub_img, axis = (1,2))
                print(sub_img)
                print(new_img[i-1][j-1])
    cv.imwrite('test.png',new_img)
    
    
#the kernel need to be the same depth as the image
kernel1 = np.array([[[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1]],
                    [[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1]],
                    [[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1]]])#Fuzzification
kernel2 = np.array([[[-1,-2,-1],[0,0,0],[1,2,1]],
                    [[-1,-2,-1],[0,0,0],[1,2,1]],
                    [[-1,-2,-1],[0,0,0],[1,2,1]]])#Finds verticals
kernel3 = np.array([[[-1,0,1],[-2,0,2],[-1,0,1]],
                    [[-1,0,1],[-2,0,2],[-1,0,1]],
                    [[-1,0,1],[-2,0,2],[-1,0,1]]])#Finds horizontals
kernel4 = np.array([[[-1,1,1],[-2,2,2],[-1,3,1]],
                    [[-1,4,1],[-2,5,2],[-1,6,1]],
                    [[-1,7,1],[-2,8,2],[-1,9,1]]])
img = cv.imread('icon.png')
convolve(kernel4,img)
