# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:27:57 2017

@author: simon
"""
import sys
import traceback
import nltk
import torch
from torch.autograd import Variable


import os
directory=os.getcwd()
if(not directory[-5:]=='model'):
    directory=directory+ '/model'
    sys.path.insert(0,directory)
    print("new path added to sys.path : ", directory)
    


class Sentence2Vec(object):
    def __init__(self,
                 glove_path=directory+"/InferSent/dataset/GloVe/glove.840B.300d.txt",
                 useCuda=False,
                 Nwords=10000,
                 pathToInferSentModel=directory+'/InferSent/infersent.allnli.pickle',
                 modelDirectory=directory+"/InferSent"):
        print ("Loading Glove Model")
        
        
        #adding directory to the InferSent module
        if (not modelDirectory in sys.path):
            print("adding local directory to load the model")
            sys.path.append(modelDirectory)
        else:
            print("directory already in the sys.path")
            
        
        nltk.download('punkt')        
        
        #loading model
        if (useCuda):
            print("you are on GPU (encoding ~1000 sentences/s, default)")
            self.infersent = torch.load(pathToInferSentModel)
        else: 
            print("you are on CPU (~40 sentences/s)")
            self.infersent = torch.load(pathToInferSentModel, map_location=lambda storage, loc: storage)
        
        
        
        self.infersent.set_glove_path(glove_path)
        
        print("loading the {} most common words".format(Nwords))
        try: 
            self.infersent.build_vocab_k_words(K=Nwords)
            print("vocab trained")
        except Exception as e:
            print("ERROR")    
            print(e)
            print("\nPOSSIBLE SOLUTION")
            print("if you have an encoding error, specify encoder='utf8' in the models.py file line 111 " )
        
        print("done")
    
    
    def encodeSent(self,sentence):
        if(type(sentence)==str):
            #print("processing one sentence")
            return(torch.from_numpy((self.infersent).encode([sentence],tokenize=True)))
        else:
            #print("processing {} sentences".format(len(sentence)))
            return(torch.from_numpy((self.infersent).encode(sentence,tokenize=True)))


#test code
#model=Sentence2Vec()
#sentence='Hello I am Simon'
#sentences=[sentence,'How are you ?']
#x=model.encodeSent(sentence)
#print(x.size())
#x=model.encodeSent(sentences)
#print(x.size())
#model.infersent.visualize(sentence)
#
#    
    
    
