# Image Style Transfer Using Convolutional Neural Networks 

## Summary



## Research Object

Image style transfer 

## Main Problems

Limitations about texture transfer: Only low-level(pixel value) features were used to generate new image

## Method

### Architecture

><img src="../res/style2.png" width = "700" height = "400" align=center />

><img src="../res/style1.png" width = "700" height = "400" align=center />

a: the style image</br>
x: the white noise used to generate the transferred output</br>
p: the content image</br>

The features of style image, content image and the white noise are extracted by the convolutional layer.  

Core idea: 

1. Minimize the content loss(feature distance) and style loss(gram matrix distance) simultaneously in order to generate a transferred image. 
2. The style representations should be spatially invariant(->gram matrix of features is spatially invariant)    
3. The content representations and style representations can be somehow distangled.   

### Loss definition

P -> the features of the content image     
F -> the features of the white noise(generated image)     
A -> the features of the style image      
G -> the gram matrix of F(and there are other useful style representations)      

The content loss is the L2 loss between a single high level content feature map and a white noise feature map.      
![](../res/style7.gif)        

The Style Representations is defined as the gram matrix of the content representations which is spacially invariant.    

![](../res/style4.gif)

The style loss is the summation of the L2 losses      
![](../res/style3.gif)       
![](../res/style5.gif)

Finally, the total loss is controlled by the content weight and the style weight    
![](../res/style6.gif)     

Weight controlling examples:     
><img src="../res/style8.png" width = "470" height = "540" align=center />   

## Evaluation

### content representation in different layers

><img src="../res/style9.png" width = "370" height = "640" align=center />  

low level features: close to pixel space
high level features: more distortion and abstract

### different white noise -> different image

>It should be noted that only initialising with noise allows to generate an arbitrary number of new images (Fig 6 C). Initialising with a fixed image always deterministically leads to the same outcome (up to stochasticity in the gradi- ent descent procedure)
><img src="../res/style10.png" width = "370" height = "540" align=center />  

### noisy Photorealistic style transfer

You can see that there are many pixel noises, which means that it cannot preserve the high resolution features perfectly.        
><img src="../res/style11.png" width = "370" height = "440" align=center />      

### redundant computations -> extremely slow

This method need to iterate 200~300 times to generate a single high resolution image.   
