import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import  numpy as np
from data.TFG.MaxMinNormalization import MinMaxNormalization

class TFGDataset(Dataset):
    def __init__(self,data_path="../data/TFG/",is_train=True,T=96,len_closeness=3,len_period=1,len_trend=1):
        '''
        T: number of time slip every day
        '''
        self.week_offset = T*7
        self.day_offset = T
        self.len_closeness = len_closeness
        self.len_period = len_period
        self.len_trend = len_trend
        self.mmn = MinMaxNormalization()
        file_path = data_path
        f = h5py.File(file_path,"r")
        self.tfg= np.array(f['tfg'],dtype=np.float32)

        self.tfg = self.mmn.transform(self.tfg)

        self.len = len(self.tfg)
        self.XC = []
        self.XP = []
        self.XT = []
        self.Y = []
        self.L = []

        for i in range(0+self.week_offset*self.len_trend,self.len):
            if i%200==199:
                print("数据准备: %d / %d"%(i,self.len))
            self.Y.append([self.tfg[i]])
            self.L.append([self.tfg[i-1]])
            self.XC.append([self.tfg[j] for j in range(i-len_closeness,i)])
            self.XP.append([self.tfg[i-self.day_offset*j] for j in range(1,len_period+1)])
            self.XT.append([self.tfg[i-self.week_offset*j] for j in range(1,len_trend+1)])

        self.XC = torch.Tensor(np.array(self.XC))
        self.XP = torch.Tensor(np.array(self.XP))
        self.XT = torch.Tensor(np.array(self.XT))
        self.Y = torch.Tensor(np.array(self.Y))
        self.L = torch.Tensor(np.array(self.L))
        self.len = self.len-self.week_offset*self.len_trend
        print(self.XC.shape,self.XP.shape,self.Y.shape)


    def __getitem__(self, item):
        return self.XC[item],self.XP[item],self.XT[item],self.L[item],self.Y[item]

    def __len__(self):
        return self.len


if __name__=="__main__":
    #tfgdata_train= TFGDataset(data_path="../data/TFG/",T=96,is_train=True,len_closeness=3,len_trend=1,len_period=1)
    tfgdata_train = TFGDataset(data_path="../data/TFG/train15.h5",T=96,is_train=True,len_closeness=3,len_period=1,len_trend=1)
    train_loader = DataLoader(dataset=tfgdata_train,batch_size=32,shuffle=False,num_workers=0)
    tfgdata_test = TFGDataset(data_path="../data/TFG/test15.h5",T=96,is_train=False,len_closeness=3,len_period=1,len_trend=1)


