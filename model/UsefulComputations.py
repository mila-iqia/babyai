# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:47:53 2017

@author: simon
"""

import numpy as np
import torch
from torch.autograd import Variable


class Computations(object):
    def __init__(self,useCuda=False):
        self.useCuda=useCuda
            
    def preProcessImage(self,img):
        #return a torch variable containing the preProcessed image 
        
        #rescale between 0 and 1
        img=np.array(img,dtype= "float")
        maxValue=np.max(img)
        minValue=np.min(img)
        img=(img-minValue)/(maxValue-minValue)
        
        
        #ensure the right order of shapes
        #batch of 1
        if(len(img.shape)==3):
            img=np.array([img])
        
        a,b,c,d=img.shape
        if(b==c):
        #case to avoid : shape = (Batch, dimX, dimY, channels)
            img=np.swapaxes(img,1,3)
            img=np.swapaxes(img,2,3)
            #print("new ",img.shape)

        #set to pytorch Variable
        img=Variable(torch.from_numpy(img).float())
    
        if(self.useCuda):
            img=img.cuda()
        
    
        #reshape for a batch of 1
        return(img)
    
    def affineTransformation(self,x,gamma,beta):
        #assuming that the shapes are
        # x : B,C,H,W
        # gamma and beta : B,C
        # produces a tensor of shape : B,C,H,W and store it into x
        #print("gamma ", gamma.size())
        #print("x ", x.size())

        gamma = gamma.unsqueeze(2).unsqueeze(3).expand_as(x)
        beta = beta.unsqueeze(2).unsqueeze(3).expand_as(x)
    
    
        return(gamma*x+beta)


#test

#A=[1,2]
#B=np.array([3,4],dtype="float")
#
#A1=preProcessImage(A)
#print(type(A1))
#
#B1=preProcessImage(B)
#print(type(B1))

#img=np.ones((32,7,7,3))
#img[0,:,:,0]=0
#img[0,:,:,1]=0.5
#img[0,0,:,:]=1
#img[0,1,:,:]=0
#for i in range(3):
#    pl.imshow(img[0,:,:,i])
#    pl.show()
#    
#model=PreProcess()
#output=model.preProcessImage(img)
#output=output.data.numpy()
#
#for i in range(3):
#    pl.imshow(output[0,i,:,:])
#    pl.show()
#    
#new=np.swapaxes(img,0,2)
#new=np.swapaxes(new,1,2)
#for i in range(3):
#    pl.imshow(new[i,:,:])
#    pl.show()