import numpy as np
import torch
from train import  k_fold
from point_Model.point import point_cloud


input = np.loadtxt("/home/ubuntu/Rencq/nerf_data/point_cloud/fern/3_0.2/005_35/out_point_cloud0.txt")
input = torch.tensor(input)
maxlabel = max(input[...,3])
output = maxlabel+1
output = int(output)
X_train = input[...,0:3].clone()
# y_train = input[...,3]
y_train = input[...,3].clone()
y_train = torch.tensor(y_train,dtype=torch.long)
print("input data\n",input)
print("output class\n",output)
print("X train\n",X_train)
print("y train\n",y_train)
model = point_cloud(3,1,1,output)


num_epochs=1000
learning_rate=0.001
weight_decay=0
batch_size=4000



k_fold(model,4,X_train,y_train,num_epochs=num_epochs,learning_rate=learning_rate,weight_decay=weight_decay,batch_size=batch_size)

torch.save(model,'../logs/point_model/3_1_1.pt')