# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import torch
import pandas as pd
import csv
import nltk

#path = "/u/sebbaghs/Projects/GloveData/glove.6B.50d.txt"
class Dictionary(object):
    def __init__(self,gloveFile):
        print ("Loading Glove Model")
        self.model = pd.read_table(gloveFile, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        print ("Done.",len(self.model)," words loaded!")

    def word2vec(self,word):
        return (torch.from_numpy(self.model.loc[word].as_matrix()))

    def seq2matrix (self,sequence):
        l=nltk.word_tokenize(sequence.lower())
        lenSeq=len(l)
        dim=len(self.word2vec(l[0]))
        output=torch.zeros((lenSeq,1,dim))
        for i in range(lenSeq):
            output[i,0,:]=self.word2vec(l[i])
        return(output)
