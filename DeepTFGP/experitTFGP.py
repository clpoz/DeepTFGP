from models.DeepTFGPNet import DeepTFPNet
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset.Datas import TFGDataset
from data.TFG.MaxMinNormalization import  MinMaxNormalization

if __name__=="__main__":
    epoch = 150
    batch_size = 32
    T = 96
    len_closeness = 4
    len_period = 1
    len_trend =1
    node = 32
    lr = 0.00001
    nb_residual_unit = 4
    nodes = ['8304B', '8304A', '3868M', '8307B', '8308A', '8308J', '8312M', '3867M',
             '3865L', '3864M', '8310K', '8312L', '3869B', '3866B', '3863B', '3869A',
             '3864A', '8311B', '8311A', '8312J', '8311M', '3869K', '8313L', '8312K',
             '3867J', '3867K', '8314M', '8314J', '8314K', '3872B', '3873L', '3873A']
    node_map = {}
    for i in range(len(nodes)):
        node_map[nodes[i]] = i


    tfgdata_train= TFGDataset(data_path="../data/TFG/train15.h5",T=T,is_train=True,len_closeness=len_closeness,len_trend=len_trend,len_period=len_period)
    tfgdata_test = TFGDataset(data_path="../data/TFG/test15.h5",T=T,is_train=False,len_closeness=len_closeness,len_period=len_period,len_trend=len_trend)
    train_loader = DataLoader(dataset=tfgdata_train,batch_size=32,shuffle=True,num_workers=0)
    #test_loader = DataLoader(dataset=tfgdata_test,batch_size=32,shuffle=False,num_workers=0)
    x_test = [tfgdata_test.XC,tfgdata_test.XP,tfgdata_test.XT,tfgdata_test.L]
    y_test = tfgdata_test.Y
    model = DeepTFPNet(lr=lr,
                       epoches=epoch,
                       T=T,
                       batch_size=batch_size,
                       len_closeness=len_closeness,
                       len_trend=len_trend,
                       len_period=len_period,
                       node=node,
                       nb_residual_unit=nb_residual_unit)
    model = model.cuda()
    mmn = MinMaxNormalization()
    #model.load_model("best")
    model.train_model(train_loader,x_test,y_test)
    rmse,mae = model.evaluate(x_test,y_test)
    print('rmse %.4f  mae %.4f'%(rmse,mae))

    #model.test_model_tfg(mmn,x_test,y_test)
    model.test_model_node(nodes,mmn,X_test=x_test,Y_test=y_test)


