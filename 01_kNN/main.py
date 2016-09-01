# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 15:27:14 2016

@author: Jack
"""
import matplotlib
import matplotlib.pyplot as plt
import logging
from kNN import *
from numpy import *
import operator

logging.basicConfig(level = logging.INFO)

'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(r[:,1], r[:,2], 15*array(l), 15*array(l) )
plt.show()
'''

data, _min, _max = auto_norm(r)
logging.info(data)
logging.info(_min)
logging.info(_max)

def test_kNN(hoRatio):
    dataSet_org, label= file2Mat('datingTestSet.txt')
    _len = int(hoRatio*dataSet_org.shape[0])
    dataSet, data_min, data_max = auto_norm(dataSet_org)
    err_count = 0
    for i in range(_len):
        ret_label = classifyKNN(dataSet[i,:], dataSet[_len:], label[_len:], 3)
        logging.info(ret_label)
        if ret_label == label[i]:
            print('predict right!')
        else:
            err_count += 1
            print('predict wrong!')
    return float(err_count)/_len
per = test_kNN(0.1)
print(per)
print('wrong rate is %f' % per)