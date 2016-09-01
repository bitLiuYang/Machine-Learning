# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 11:11:36 2016

@author: Jack
"""


from numpy import *
import operator
import logging
logging.basicConfig(level = logging.WARNING)

#源程序中有问题呀...
def int_class(str_class):
    if str_class == 'largeDoses':
        return 1
    elif str_class == 'smallDoses':
        return 2
    else:
        return 3
        
#提取文件数据，返回数组
def file2Mat(file_name):
    fr = open(file_name)
    lines = fr.readlines()
    num_line = len(lines)
    ret_mat = zeros( (num_line, 3) )
    ret_label = []
    idx= 0
    for each_line in lines:
        each_line = each_line.strip()
        data_list = each_line.split('\t')
        ret_mat[idx, :] = data_list[0:3]
        #logging.info(data_list[-1])
        ret_label.append( int_class( data_list[-1] ) )
        #ret_label.append(['qwe'])       
        idx += 1
    return ret_mat, ret_label

def makeDataSet():
    dataSet = array([[1.0, 1.1], [0.9, 0.95], [0.0, 0.2], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return dataSet, labels
    
def classifyKNN(inX, dataSet, label, k):
    m = dataSet.shape[0]
    diffMat = tile(inX, (m,1)) - dataSet
    distances = diffMat**2
    sqDist = distances.sum(axis=1)
    sortDistInd = sqDist.argsort()
    classCount = {}    
    for i in range(k):
        voteIlabel = label[sortDistInd[i]]
        logging.info('voteIlabel is %s' % voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    logging.info(classCount)
    sorted_list = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse = True)
    logging.info(sorted_list)
    logging.info(sorted_list[0])
    return sorted_list[0][0]
    
def auto_norm(dataSet):
    _min = dataSet.min(0)
    _max = dataSet.max(0)
    _len = dataSet.shape[0]
    ret_data = (dataSet-tile(_min, (_len, 1)) )/tile(_max-_min, (_len,1)) 
    return ret_data, _min, _max
    



