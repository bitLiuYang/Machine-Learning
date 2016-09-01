# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 14:25:43 2016

@author: Jack
"""

from kNN import *
from os import listdir
import logging
logging.basicConfig(level = logging.DEBUG)

def num2vec(file_name):
    ret_vec = zeros((1, 1024))
    fr = open(file_name)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            ret_vec[0, 32*i + j] = int(line_str[j])
    return ret_vec

def get_num_data(path):
    num_label = []
    num_file = listdir(path)
    file_cnt = len(num_file)
    trainingSet = zeros((file_cnt, 1024))
    for i in range(file_cnt):
        file_name = num_file[i]
        num_label.append(file_name.split('_')[0])
        trainingSet[i,:] = num2vec(path+'/'+file_name)
    return trainingSet, num_label
    
def handWriting_rec():
    traingSet, traingLabel = get_num_data('digits/trainingDigits')
    validSet, validLabel = get_num_data('digits/testDigits')
    m = len(validLabel)
    ret_label = []
    error_cnt = 0
    for i in range(m):
        label = classifyKNN(validSet[i,:], traingSet, traingLabel, 3)
        ret_label.append(label)
        if label != validLabel[i]:
            error_cnt += 1
    print("error rate : %f /n" % float(error_cnt*1.0/m) )
    print("error count is %d; total  valid data is %d" % (error_cnt, m) )
    
handWriting_rec()