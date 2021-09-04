# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 15:27:41 2021

@author: angel
"""

def save_dict(dic, filename):
    f = open(filename,'w')
    f.write(str(dic))
    f.close()


def load_dict(filename):
    f = open(filename,'r')
    data=f.read()
    data = data.replace('array', 'np.array')
    f.close()
    return eval(data)