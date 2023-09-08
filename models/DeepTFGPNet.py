import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import torch.optim as optim
import time
from data.TFG.MaxMinNormalization import MinMaxNormalization

class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUnit, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        z = self.bn1(x)
        z = F.relu(z)
        z = self.conv1(z)
        z = self.bn2(z)
        z = F.relu(z)
        z = self.conv2(z)
        return z + x


class DeepTFPNet(nn.Module):
    def __init__(self,T=96,lr=0.001,epoches=50,batch_size=32,len_closeness=3,len_period=1,len_trend=1,node=32,nb_residual_unit=2):
        super(DeepTFPNet, self).__init__()
        self.epoches=epoches
        self.lr=lr
        self.batch_size=batch_size
        self.len_closeness = len_closeness
        self.len_period = len_period
        self.len_trend=len_trend
        self.node = node
        self.nb_residual_unit=nb_residual_unit
        self.T = T
        self.data_min=0.0
        self.data_max = 1092.0
        self.best_rmse = 99
        self.best_mae = 99
        self.save_path = "DeepTFGP15/L%d_C%d_P%d_T%d/" %(self.nb_residual_unit,self.len_closeness,self.len_period,self.len_trend)
        self._build_stresnet()
        print(self.save_path)

    def _build_stresnet(self,):
        self.c_net = nn.ModuleList([
            nn.Conv2d(self.len_closeness,64,kernel_size=3,stride=1,padding=1),
        ])
        for i in range(self.nb_residual_unit):
            self.c_net.append(ResUnit(64,64))
        self.c_net.append(nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1))

        self.p_net = nn.ModuleList([
            nn.Conv2d(self.len_period,64,kernel_size=3,stride=1,padding=1),
        ])
        for i in range(self.nb_residual_unit):
            self.p_net.append(ResUnit(64,64))
        self.p_net.append(nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1))

        self.t_net = nn.ModuleList([
            nn.Conv2d(self.len_trend,64,kernel_size=3,stride=1,padding=1),
        ])
        for i in range(self.nb_residual_unit):
            self.t_net.append(ResUnit(64,64))
        self.t_net.append(nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1))


        self.w_c = nn.Parameter(torch.rand(1,self.node,self.node),requires_grad=True)
        self.w_p = nn.Parameter(torch.rand(1,self.node,self.node),requires_grad=True)
        self.w_t = nn.Parameter(torch.rand(1,self.node,self.node),requires_grad=True)
        self.w_l = nn.Parameter(torch.rand(1,self.node,self.node),requires_grad=True)

    def forward_branch(self,branch,x_input):
        for layer in branch:
            x_input = layer(x_input)
        return x_input

    def forward(self,xc,xp,xt,xl):

        c_out = self.forward_branch(self.c_net,xc)
        p_out = self.forward_branch(self.p_net,xp)
        t_out = self.forward_branch(self.t_net,xt)

        con = self.w_c.unsqueeze(0)*c_out + self.w_p.unsqueeze(0)*p_out +\
              self.w_t * t_out + self.w_l * xl

        return torch.tanh(con)

    def train_model(self,train_loader,test_x,test_y):
        optimizer = optim.Adam(self.parameters(),lr = self.lr)
        epoch_loss = []
        for ep in range(self.epoches):
            self.train()
            for i,(xc,xp,xt,xl,y) in enumerate(train_loader):
                xc = xc.cuda()
                xp = xp.cuda()
                xt = xt.cuda()
                xl = xl.cuda()
                y = y.cuda()

                ypred = self.forward(xc,xp,xt,xl)

                loss = ((ypred-y)**2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())

                if i%50 == 0:
                    print("ep %d it %d, loss %.6f" % (ep, i, loss.item()))

            print("ep %d, loss %.6f"%(ep, np.mean(epoch_loss)))
            epoch_loss=[]
            test_rmse,test_mae = self.evaluate(test_x,test_y)
            print("ep %d test rmse %.4f, mae %.4f" % ( ep, test_rmse, test_mae))
            if test_rmse < self.best_rmse:
                self.save_model("best")
                self.best_rmse = test_rmse
                self.best_mae = test_mae


    def evaluate(self,X_test,Y_test):
        """
        X_test: a quadruplle: (xc, xp, xt, xl)
        y_test: a label
        mmn: minmax scaler, has attribute _min and _max
        """
        self.eval()
        for i in range(4):
            X_test[i] = X_test[i].cuda()
        Y_test = Y_test.cuda()
        with torch.no_grad():
            ypred = self.forward(X_test[0], X_test[1], X_test[2], X_test[3])
            rmse = ((ypred - Y_test) **2).mean().pow(1/2)
            mae = ((ypred - Y_test).abs()).mean()
            return rmse * (self.data_max - self.data_min) / 2, mae * (self.data_max - self.data_min) / 2

    def save_model(self,name):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        torch.save(self.state_dict(), self.save_path + name + ".pt")

    def load_model(self,name):
        if not name.endswith(".pt"):
            name += ".pt"
        self.load_state_dict(torch.load(self.save_path + name))

    def test_model_tfg(self,mmn:MinMaxNormalization,X_test,Y_test):
        """
        X_test: a quadruplle: (xc, xp, xt, xl)
        y_test: a label
        mmn: minmax scaler, has attribute _min and _max

        """
        self.eval()
        for i in range(4):
            X_test[i] = X_test[i].cuda()
        Y_test = Y_test.cuda()
        with torch.no_grad():
            ypred = self.forward(X_test[0], X_test[1], X_test[2], X_test[3])

            best_rmse = 999
            best_real=None
            best_pred=None
            best_tfg = None
            for i in range(len(Y_test)):
                yt= Y_test[i].cpu().detach().numpy()[0]
                yp = ypred[i].cpu().detach().numpy()[0]

                yt,yp = mmn.inverse_transform(yt),mmn.inverse_transform(yp)
                real_flow = np.zeros(self.node)
                predict = np.zeros(self.node)

                for j in range(self.node):
                    real_flow[j]=yt[j][j]
                    predict[j]=yp[j][j]

                rmse = pow(((real_flow-predict)**2).mean(),0.5)

                if rmse<best_rmse and real_flow.mean()>80:
                    best_rmse = rmse
                    best_real = real_flow
                    best_pred = predict
                    best_tfg = i

            print('best rmse %.4f:'%best_rmse)
            print('timestamp: %d'%best_tfg)
            print('       real  :  predict')
            for j in range(self.node):
                print('node %d:  %.2f %.2f' % (j, best_real[j], best_pred[j]))

    def test_model_node(self,nodes,mmn:MinMaxNormalization,X_test,Y_test,tt=96):
        """

        """
        self.eval()
        for i in range(4):
            X_test[i] = X_test[i].cuda()
        Y_test = Y_test.cuda()
        with torch.no_grad():
            ypred = self.forward(X_test[0], X_test[1], X_test[2], X_test[3])
            day_size = self.T
            # 30天是训练集天数
            nodep = np.empty((32,30,self.T))
            noder = np.empty((32,30,self.T))
            print(nodep.shape)
            for i in range(len(Y_test)):
                d = i // self.T #96
                st = i % self.T
                for j in range(self.node):
                    nodep[j][d][st] = ypred[i][0][j][j]
                    noder[j][d][st] = Y_test[i][0][j][j]

            nodep = mmn.inverse_transform(nodep)
            noder = mmn.inverse_transform(noder)
            node_rmse = pow(((nodep- noder) ** 2).mean(), 0.5)
            node_mae = (abs(nodep-noder)).mean()

            print('node_rmse %.4f  node_mae %.4f'%(node_rmse,node_mae))

            for j in range(self.node):
                best_rmse = 999
                best_node = None
                best_day = None
                best_mae = None
                for d in range(30):
                    predict = nodep[j][d]
                    real = noder[j][d]
                    rmse = pow(((real- predict) ** 2).mean(), 0.5)
                    mae = (abs((real-predict))).mean()
                    if rmse<best_rmse:
                        best_rmse=rmse
                        best_node=j
                        best_day=d
                        best_mae = mae

                print('best_rmse %.4f best_mae%.4f'%(best_rmse,best_mae))
                print('day :%d, node : %d %s'%(best_day,best_node,nodes[best_node]))



























