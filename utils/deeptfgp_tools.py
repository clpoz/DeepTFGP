import pandas as pd
import numpy as np
import os
import csv
import h5py
import torch.nn as nn
import torch



def get_adj_matrix(adj_path:str)->list:

    adj_matrix = []
    with open(adj_path,"r",encoding="utf-8") as f:

        s = f.readlines()
        for st in s:
            row = [int(sp) for sp in st.strip("\n").split(" ")]
            adj_matrix.append(row)
            # print(row)

    return adj_matrix


def get_node_flow(data_path:str,time_size)->np.array:

    files =nodes #os.listdir(data_path)
    flow = np.empty((32,time_size))

    for file in files:
        data = np.genfromtxt(data_path+"/"+file+".csv",delimiter=',',dtype=str)
        data = data[1:251,1:97]
        data = data.flatten().astype(np.int32)
        flow[node_map[file]]=data

    return flow


def make_time_tfg(adj_matrix:list,node_flow:np.array,time_size=11520,interval=1,mark='15'):


    tfg = np.empty((time_size//interval,32,32))

    for t in range(time_size//interval):
        # 第t时刻的tfg
        fg = np.empty((32,32))
        for i in range(32):
            for j in range(32):
                if adj_matrix[i][j] ==-1:
                    fg[i][j]=0
                else:
                    fflow = 0
                    for k in range(interval):
                        fflow+=node_flow[adj_matrix[i][j]][t*interval+k]
                    fg[i][j]=fflow

        tfg[t]=fg

    data_path = "../data/TFG/TFG"+mark+'.h5'
    with h5py.File(data_path,"w") as f:
        f['tfg']=tfg


def make_dataset(tfg,interval,train_path,test_path,train_size=220):
    T = 96//interval
    size = train_size*T
    train_set = tfg[:train_size*T]
    test_set = tfg[(train_size-7)*T:]
    print('train:',train_set.shape,'test:',test_set.shape)
    with h5py.File(train_path,"w") as ftrian:
        ftrian['tfg']=train_set
    with h5py.File(test_path,"w") as ftest:
        ftest['tfg']=test_set


if __name__=="__main__":

    nodes = ['8304B', '8304A', '3868M', '8307B', '8308A', '8308J', '8312M', '3867M',
             '3865L', '3864M', '8310K', '8312L', '3869B', '3866B', '3863B', '3869A',
             '3864A', '8311B', '8311A', '8312J', '8311M', '3869K', '8313L', '8312K',
             '3867J', '3867K', '8314M', '8314J', '8314K', '3872B', '3873L', '3873A']
    node_map = {}
    for i in range(len(nodes)):
        node_map[nodes[i]] = i
    time_size = 250*96 #30*4*96 #11520
    adj_matrix = get_adj_matrix("../test/adj.txt")
    marks = ['15','30','45','60']
    for interval in range(1,5):
        data_path ="../data/TFG/TFG"+marks[interval-1]+'.h5'
        f = h5py.File(data_path,'r')
        tfg = f['tfg']
        tfg = np.array(tfg,dtype=np.float32)
        print(tfg.max(),tfg.min())

