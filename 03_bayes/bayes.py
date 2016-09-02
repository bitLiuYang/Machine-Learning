# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 14:25:43 2016

@author: Jack
"""

def loadDataSet():
    text_set = [
        ['Dog', 'is', 'spam', 'OK'],
        ['Cat', 'is', 'very', 'good'],
        ['Human', 'are', 'very', 'smart'],
        ['Money', 'is', 'very', 'cheap']    
    ] 
    label = [0,0,0,1]
    return text_set,label
    
def createVocabList(dataSet):
    vocab = set([]) #使用set类型，保证没个单词只出现一次
    for document in dataSet:
        vocab = vocab | set(document)
    return list(vocab)
    
def setOfWords2Vec(vocabList, inputSet):
    ret_vec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            ret_vec[vocabList.index(word)] = 1
        else:
            print('the word : %s is not in vocabList' % word)
    return ret_vec
    
listText, Label = loadDataSet()
setText = createVocabList(listText)
print(setText)
list_in = setOfWords2Vec(setText, listText[3])
print(list_in)
print('new...'):wq

