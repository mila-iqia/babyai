# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:47:53 2017

@author: simon
"""

import numpy as np
import torch
from torch.autograd import Variable


def preProcessImage(img):
    #return a torch variable containing the preProcessed image coming as either
    #a numpy array or a list
    maxValue=255.0
    image=torch.mul(torch.from_numpy(np.array(img,dtype="double")),1/maxValue)



    #reshape for a batch of 1
    return(image.unsqueeze(0))

def affineTransformation(x,gamma,beta):
    #assuming that the shapes are
    # x : B,C,H,W
    # gamma and beta : B,C
    # produces a tensor of shape : B,C,H,W and store it into x

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
#
