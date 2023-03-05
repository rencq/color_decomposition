import numpy as np
import torch
import torch.nn
import torch.nn.functional as F


class point_cloud(torch.nn.Module):
    def __init__(self,input,hidden_num1,hidden_num2,output):
        super(point_cloud,self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input,128,dtype=torch.float64),torch.nn.ReLU())
        self.linear_block1 = torch.nn.Sequential()
        self.linear_block2 = torch.nn.Sequential()

        for i in range(hidden_num1):
            self.linear_block1.add_module(f'linear{i}',torch.nn.Linear(128,128,dtype=torch.float64))
            self.linear_block1.add_module(f'relu1{i}',torch.nn.ReLU())

        self.layer2 = torch.nn.Sequential(torch.nn.Linear(128,256,dtype=torch.float64),torch.nn.ReLU())

        for i in range(hidden_num2):
            self.linear_block2.add_module(f'linear{i}', torch.nn.Linear(256, 256,dtype=torch.float64))
            self.linear_block2.add_module(f'relu2{i}', torch.nn.ReLU())

        self.layer3 = torch.nn.Linear(256,output,dtype=torch.float64)

    def forward(self,X):
        ans1 = self.linear_block1(self.layer1(X))
        ans2 = self.linear_block2(self.layer2(ans1))
        return self.layer3(ans2)


class point_cloud_classical(torch.nn.Module):
    def __init__(self,input,output):
        super(point_cloud_classical, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.ReLU(),torch.nn.Linear(input,256,dtype=torch.float64))
        self.layer2 = torch.nn.Sequential(torch.nn.ReLU(),torch.nn.Linear(256,output,dtype=torch.float64))

    def forward(self,X):
        return self.layer2(self.layer1(X))

class point_cloud_classical_num():
    def __init__(self,number,input,output):
        self.number = number
        self.point_cloud_map = {}
        for i in range(number):
            self.point_cloud_map[f'model{i}'] = point_cloud_classical(input[i],output[i])

    def get_model(self):
        return self.point_cloud_map

    def __getitem__(self, item):
        return self.get_model()[item]

    def __len__(self):
        return self.number
