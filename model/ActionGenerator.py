

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:13:14 2017

@author: sebbaghs
"""

import sys
import os
currentDirectory = os.getcwd()
if not currentDirectory in sys.path:
    print('adding local directory : ', currentDirectory)
    sys.path.insert(0,currentDirectory)

import torch

import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sentenceEmbedder as SE
import torch.nn.functional as F
import UsefulComputations 
import timeit
import torch.optim as optim

torch.manual_seed(1)
np.random.seed(1)

 
#for now CUDA does not work on my machine...
cp=UsefulComputations.Computations(useCuda=False)

#otherwise
#cp=UsefulComputations.Computations(useCuda=torch.cuda.is_available())

def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

class FilmBlock(nn.Module):
    def __init__(self, Nchannels):        
        super(FilmBlock, self).__init__()
        self.Conv_1=nn.Conv2d(Nchannels,Nchannels,3,stride=1,padding=1) 
        self.BN_1=torch.nn.BatchNorm2d(Nchannels)
        self.Conv_2=nn.Conv2d(Nchannels,Nchannels,3,stride=1,padding=1)
        self.BN_2=torch.nn.BatchNorm2d(Nchannels,affine=False)
        
    
    def forward(self,image,paramFromText):
        global cp
        image = self.Conv_1(image)
        image = self.BN_1(image)
        image = F.relu(image)
        
        x = self.Conv_2(image)
        x = self.BN_2(x)
        x = cp.affineTransformation(x,paramFromText[:,0,:],paramFromText[:,1,:])
        x = F.relu(x)
        
        x+=image
        x = F.relu(x)        
        
        return(x)


class ActionGenerator(nn.Module):
    def __init__(self,
        numberOfBlocks=4,
        numberOfFeaturesInBlock=64,
        numberOfActions=4,
        finalVectorDim=128):
        super(ActionGenerator, self).__init__()

        #Setting sentence encoder
        #should be implemented but not work on my machine for now
        self.useCuda=torch.cuda.is_available()     
        self.useCuda=False
        #should be implemented but not work on my machine for now
        #self.TextEncoder=SE.Sentence2Vec(useCuda=self.useCuda)        
        self.TextEncoder=SE.Sentence2Vec(useCuda=self.useCuda)
            
        self.dense1_SE=nn.Linear(4096,numberOfFeaturesInBlock*2*numberOfBlocks) 
        #self.dense2_SE=nn.Linear(2048,1024)

       
        #visual parameters
        self.numberOfBlocks=numberOfBlocks
        self.PreConv0=nn.Conv2d(3,8,3,stride=1,padding=1)
        self.PreBN0=torch.nn.BatchNorm2d(8)
        self.PreConv1=nn.Conv2d(8,16,3,stride=1,padding=1)
        self.PreBN1=torch.nn.BatchNorm2d(16)
        self.PreConv2=nn.Conv2d(16,32,3,stride=1,padding=1)
        self.PreBN2=torch.nn.BatchNorm2d(32)
        self.PreConv3=nn.Conv2d(32,64,3,stride=1,padding=1)
        self.PreBN3=torch.nn.BatchNorm2d(64)
        self.PreConv4=nn.Conv2d(64,64,3,stride=1,padding=1)
        self.PreBN4=torch.nn.BatchNorm2d(64)

        self.numberOfFeaturesInBlock=numberOfFeaturesInBlock

        #blocks for FILM
        self.dicOfBlocks={}

        #block0
        self.Block0=FilmBlock(self.numberOfFeaturesInBlock)
        self.dicOfBlocks["Block0"]=self.Block0
        #block1
        self.Block1=FilmBlock(self.numberOfFeaturesInBlock)
        self.dicOfBlocks["Block1"]=self.Block1
        #block2
        self.Block2=FilmBlock(self.numberOfFeaturesInBlock)
        self.dicOfBlocks["Block2"]=self.Block2
        #block3
        self.Block3=FilmBlock(self.numberOfFeaturesInBlock)
        self.dicOfBlocks["Block3"]=self.Block3
        
        
        #image2vector
        self.finalVectorDim=finalVectorDim
        self.Conv_I2V_1=nn.Conv2d(64,128,3,stride=1)
        self.Conv_I2V_2=nn.Conv2d(128,128,3,stride=1)
        self.Conv_I2V_3=nn.Conv2d(128,128,3,stride=1)
        self.Dense_I2V_1=nn.Linear(128,self.finalVectorDim) 
        self.Dense_I2V_2=nn.Linear(self.finalVectorDim,self.finalVectorDim) 


        #history LSTM
        self.hidden_dim = finalVectorDim
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim)
        #self.Conv_history_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        #self.Conv_history_1 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.hidden = self.init_hidden()
        
        
        
        #Selection network
        self.numberOfActions=numberOfActions
        self.conv1_S=nn.Conv2d(self.numberOfFeaturesInBlock,self.numberOfFeaturesInBlock,5,stride=1,padding=0) 
        self.conv2_S=nn.Conv2d(self.numberOfFeaturesInBlock,self.numberOfFeaturesInBlock,5,stride=1,padding=0) 
        self.dense1_S=nn.Linear(128*2*2,256) 
        self.dense2_S=nn.Linear(256,self.numberOfActions) 
        
        if (self.useCuda):
            self.cuda() 
            print("Using Cuda")



    def init_hidden(self,Nbatch=1):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.useCuda:
            return (Variable(torch.zeros(1, Nbatch, self.hidden_dim).cuda()),
                Variable(torch.zeros(1, Nbatch, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(1, Nbatch, self.hidden_dim)),
                Variable(torch.zeros(1, Nbatch, self.hidden_dim)))

    

    #in the future, this would be only a setence2glove embedding function
    #and the LSTM will be called in adaptText()
    def preProcessText(self,sentence):
        output=self.TextEncoder.encodeSent(sentence)
        if(self.useCuda):
            output=Variable(output.cuda())
        else:
            output=Variable(output)
#        print(output.type)
#        output=Variable(output)
#        print(output.type)
        return(output)
        
        
    def adaptText(self,sentence):
        shape=sentence.size()
        output=F.relu(self.dense1_SE(sentence))
        #output=F.relu(self.dense2_SE(output))
        return(output.view(shape[0],self.numberOfBlocks,2,self.numberOfFeaturesInBlock) )

    def visual(self,image):
        #first pre-process

        #PreConv
        x = self.PreConv0(image)
        x = self.PreBN0(x)
        x = F.relu(x)

        x = self.PreConv1(x)
        x = self.PreBN1(x)
        x = F.relu(x)

        x = self.PreConv2(x)
        x = self.PreBN2(x) 
        x = F.relu(x)


        x = self.PreConv3(x)
        x = self.PreBN3(x)
        x = F.relu(x)

        x = self.PreConv4(x)
        x = self.PreBN4(x)
        x = F.relu(x)

        return(x)


    def mixVisualAndText(self,image,paramFromText):
        # #blocks
        x=image
        for i in range(self.numberOfBlocks):
            x = self.dicOfBlocks["Block{}".format(i)](x,paramFromText[:,i,:,:])
            
        return(x)
        
    def flatten2vector(self,image):
        image=self.Conv_I2V_1(image)
        image=self.Conv_I2V_2(image)
        image=self.Conv_I2V_3(image)
        image=self.Dense_I2V_1(image.view(-1,self.finalVectorDim))
        image=self.Dense_I2V_2(image)

        return(image)
        
    
    def historicalLSTM(self,vector):
        lstm_out, tmpHidden = self.lstm(
            vector.view(1, -1, self.hidden_dim), self.hidden)
        #print('lstm_out', lstm_out.size())
        
        self.hidden=repackage_hidden(tmpHidden)
        return(lstm_out[-1,:,:])
        #return(lstm_out[:,:,:])
        
        
    def selectAction(self,x):

        x = self.conv1_S(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        x = F.relu(x)
        #print("size 1 :", x.size())

        x = self.conv2_S(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False)(x)
        x = F.relu(x)
        #print("size 2 :", x.size())


        x = x.view(-1, 512)
        #print("size 3 :", x.size())

        x = self.dense1_S(x)
        x = F.relu(x)
        #print("size 4 :", x.size())


        x = self.dense2_S(x)
        x = F.relu(x)
        return(x)






    def forward(self,image,sentence):
        if (image.size()[0]!=sentence.size()[0]):
            print('ERROR, check the batch size of Text and Image')
        image=self.visual(image)
        text=self.adaptText(sentence)
        #print("text ",text.size())
        #print("visual ",image.size())
        representation=self.mixVisualAndText(image,text)
        representation=self.flatten2vector(representation)
        #print("representation ", representation.size())        
        
        output=self.historicalLSTM(representation)
        return(output)

        #return(output)




#def adaptParameters(self,fromText):
 
