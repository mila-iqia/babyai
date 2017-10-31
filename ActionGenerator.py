

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:13:14 2017

@author: sebbaghs
"""

import torch
import matplotlib.pyplot as pl
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import WordEmbedding as we
import UsefulComputations as cp
import torch.nn.functional as F
import timeit

torch.manual_seed(1)


class ActionGenerator(nn.Module):
    def __init__(self,
        pathToWordEmbedding="glove50.txt",
        hiddenSize_GM=100,
        batch_GM=1,
        numLayers_GM=1,
        numDirections_GM=1,
        dropout_GM=0,
        batch_A=1,
        numLayers_A=1,
        numDirections_A=1,
        dropout_A=0,
        inputShape_V=(160,160),
        numberOfBlocks=4,
        numberOfFeaturesInBlock=128,
        numberOfActions=7):
        super(ActionGenerator, self).__init__()

        #dictionnary for the Word Embedding
        self.dico=we.Dictionary(pathToWordEmbedding)

        #general mission parameters
        self.hiddenSize_GM=hiddenSize_GM
        self.batch_GM=batch_GM
        self.numLayers_GM=numLayers_GM
        self.numDirections_GM=numDirections_GM
        self.inputDim_GM=len(self.dico.word2vec("test"))
        self.dropout_GM=dropout_GM
        self.rnn_GM = nn.LSTM(self.inputDim_GM, self.hiddenSize_GM, self.numLayers_GM,dropout=self.dropout_GM)

        #advice parameters
        self.hiddenSize_A=hiddenSize_GM
        self.batch_A=batch_A
        self.numLayers_A=numLayers_A
        self.numDirections_A=numDirections_A
        self.inputDim_A=self.inputDim_GM
        self.dropout_A=dropout_A
        self.rnn_A=nn.LSTM(self.inputDim_A, self.hiddenSize_A, self.numLayers_A,dropout=self.dropout_A)

        #linear parameters
        self.inputDim_L=self.hiddenSize_A
        self.numberOfFeaturesInBlock=numberOfFeaturesInBlock
        self.outputDim_L=numberOfBlocks*self.numberOfFeaturesInBlock*2
        self.linear_L=nn.Linear(self.inputDim_L,self.outputDim_L)

        #visual parameters
        self.inputShape_V=inputShape_V
        self.numberOfBlocks=numberOfBlocks
        self.PreConv0=nn.Conv2d(3,8,3,stride=1,padding=1).double()
        self.PreConv1=nn.Conv2d(8,16,3,stride=2,padding=1).double()
        self.PreConv2=nn.Conv2d(16,32,3,stride=2,padding=1).double()
        self.PreConv3=nn.Conv2d(32,64,3,stride=2,padding=1).double()
        self.PreConv4=nn.Conv2d(64,128,3,stride=2,padding=1).double()


        #blocks
        self.dicOfBlocks={}
        #block1
        self.conv1_B0=nn.Conv2d(self.numberOfFeaturesInBlock,self.numberOfFeaturesInBlock,3,stride=1,padding=1).double()
        self.dicOfBlocks["conv1_B0"]=self.conv1_B0
        self.conv2_B0=nn.Conv2d(self.numberOfFeaturesInBlock,self.numberOfFeaturesInBlock,3,stride=1,padding=1).double()
        self.dicOfBlocks["conv2_B0"]=self.conv2_B0
        self.conv3_B0=nn.Conv2d(self.numberOfFeaturesInBlock,self.numberOfFeaturesInBlock,3,stride=2,padding=1).double()
        self.dicOfBlocks["conv3_B0"]=self.conv3_B0

        #block2
        self.conv1_B1=nn.Conv2d(self.numberOfFeaturesInBlock,self.numberOfFeaturesInBlock,3,stride=1,padding=1).double()
        self.dicOfBlocks["conv1_B1"]=self.conv1_B1
        self.conv2_B1=nn.Conv2d(self.numberOfFeaturesInBlock,self.numberOfFeaturesInBlock,3,stride=1,padding=1).double()
        self.dicOfBlocks["conv2_B1"]=self.conv2_B1
        self.conv3_B1=nn.Conv2d(self.numberOfFeaturesInBlock,self.numberOfFeaturesInBlock,3,stride=2,padding=1).double()
        self.dicOfBlocks["conv3_B1"]=self.conv3_B1

        #block3
        self.conv1_B2=nn.Conv2d(self.numberOfFeaturesInBlock,self.numberOfFeaturesInBlock,3,stride=1,padding=1).double()
        self.dicOfBlocks["conv1_B2"]=self.conv1_B2
        self.conv2_B2=nn.Conv2d(self.numberOfFeaturesInBlock,self.numberOfFeaturesInBlock,3,stride=1,padding=1).double()
        self.dicOfBlocks["conv2_B2"]=self.conv2_B2
        self.conv3_B2=nn.Conv2d(self.numberOfFeaturesInBlock,self.numberOfFeaturesInBlock,3,stride=2,padding=1).double()
        self.dicOfBlocks["conv3_B2"]=self.conv3_B2

        #block4
        self.conv1_B3=nn.Conv2d(self.numberOfFeaturesInBlock,self.numberOfFeaturesInBlock,3,stride=1,padding=1).double()
        self.dicOfBlocks["conv1_B3"]=self.conv1_B3
        self.conv2_B3=nn.Conv2d(self.numberOfFeaturesInBlock,self.numberOfFeaturesInBlock,3,stride=1,padding=1).double()
        self.dicOfBlocks["conv2_B3"]=self.conv2_B3
        self.conv3_B3=nn.Conv2d(self.numberOfFeaturesInBlock,self.numberOfFeaturesInBlock,3,stride=2,padding=1).double()
        self.dicOfBlocks["conv3_B3"]=self.conv3_B3

        #Selection network
        self.numberOfActions=numberOfActions
        self.conv1_S=nn.Conv2d(self.numberOfFeaturesInBlock,self.numberOfFeaturesInBlock,5,stride=1,padding=0).double()
        self.conv2_S=nn.Conv2d(self.numberOfFeaturesInBlock,self.numberOfFeaturesInBlock,5,stride=1,padding=0).double()
        self.dense1_S=nn.Linear(128*2*2,256).double()
        self.dense2_S=nn.Linear(256,self.numberOfActions).double()


    def processGeneralMission(self, generalMission):
        #mat=we.seq2matrix(generalMission,self.dico)
        hn = Variable(torch.randn(self.numLayers_GM*self.numDirections_GM,self.batch_GM, self.hiddenSize_GM))
        cn = Variable(torch.randn(self.numLayers_GM*self.numDirections_GM,self.batch_GM, self.hiddenSize_GM))
        output, (hn,cn) = self.rnn_GM(generalMission,(hn, cn))
        #print("general",output.size(),"hidden",hn.size())
        return(output,(hn,cn))

    def processAdvice(self, advice,fromGM):
        #mat=we.seq2matrix(advice,self.dico)
        hn = fromGM[0]
        cn = fromGM[1]
        output, (hn,cn) = self.rnn_A(advice,(hn, cn))
        #print("advice",output.size(),"hidden",hn.size())
        return(output,(hn,cn))

    def processText(self,generalMission,advice):
        output_GM,fromGM=self.processGeneralMission(generalMission)
        output_A,from_A=self.processAdvice(advice,fromGM)
        output=self.linear_L(from_A[0])
        Nbatches,Channels,Dim=output.size()
        return(output.view(Nbatches,-1,2,self.numberOfFeaturesInBlock).double())

    def visual(self,image):
        #first pre-process


        #PreConv
        x = self.PreConv0(image)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False).double()(x)
        x = F.relu(x)

        x = self.PreConv1(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False).double()(x)
        x = F.relu(x)

        x = self.PreConv2(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False).double()(x)
        x = F.relu(x)


        x = self.PreConv3(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False).double()(x)
        x = F.relu(x)

        x = self.PreConv4(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False).double()(x)
        x = F.relu(x)

        return(x)


    def mixVisualAndText(self,image,paramFromText):
        # #blocks
        x=image
        for i in range(self.numberOfBlocks):
            x = self.dicOfBlocks["conv1_B{}".format(i)](x)
            y = F.relu(x)

            x =  self.dicOfBlocks["conv2_B{}".format(i)](y)
            x = torch.nn.BatchNorm2d(x.size()[1],affine=False).double().double()(x)
            x = cp.affineTransformation(x,paramFromText[:,i,0,:],paramFromText[:,i,1,:])
            x = F.relu(x)

            x=torch.add(y,x)
            #x= self.dicOfBlocks["conv3_B{}".format(i)](x)
        print("end")
        return(x)

    def selectAction(self,x):

        x = self.conv1_S(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False).double()(x)
        x = F.relu(x)
        #print("size 1 :", x.size())

        x = self.conv2_S(x)
        x = torch.nn.BatchNorm2d(x.size()[1],affine=False).double()(x)
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






    def forward(self,image,generalMission,advice):
        fromText=self.processText(generalMission,advice)
        fromVision=self.visual(image)
        mix=self.mixVisualAndText(fromVision,fromText)
        action=self.selectAction(mix)
        #output=torch.max(action, 1)
        #max is not differentiable, we need to make it inside the train.py
        return (action)









#def adaptParameters(self,fromText):
""""
gen=ActionGenerator()

sequence="this is my sequence haha"
sequence=Variable(gen.dico.seq2matrix(sequence))
advice="you should maybe go right or left"
advice=Variable(gen.dico.seq2matrix(advice))
inputsize=160
img=np.random.rand(3,inputsize,inputsize)
img=Variable(cp.preProcessImage(img))



start = timeit.timeit()
output=gen.processText(sequence,advice)
print ("output from process text", output.size())
print("value check ",output[0,0,0,0])
vi=gen.visual(img)
print("output from visual",vi.size())
mix=gen.mixVisualAndText(vi,output)
print("output from mix",mix.size())
out=gen.selectAction(mix)
print("output from select",out.size())


end = timeit.timeit()
print ("forward time",end - start)




sequence2="this is my sequence haha"
sequence2=Variable(gen.dico.seq2matrix(sequence2))

advice2="you should maybe go right or left"
advice2=Variable(gen.dico.seq2matrix(advice2))


inputsize=160
img2=np.random.rand(3,inputsize,inputsize)
img2=Variable(cp.preProcessImage(img2))

output2=gen(img2,sequence2,advice2)
output2=output2.data.numpy()
output2=torch.from_numpy(output2)
output2=Variable(output2)



criterion = nn.MSELoss()
optimizer = optim.SGD(gen.parameters(), lr=0.001, momentum=0.9)
optimizer.zero_grad()




start = timeit.timeit()
output1=gen(img,sequence,advice)
loss = criterion(output1, output2)
end = timeit.timeit()

print ("forward time",end - start)

start = timeit.timeit()
loss.backward()
end = timeit.timeit()
print ("backward time",end - start)

start = timeit.timeit()
optimizer.step()
end = timeit.timeit()
print ("optim time",end - start)

print("done")
"""