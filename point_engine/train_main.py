import numpy as np
import torch
from train import  k_fold
from point_Model.point import point_cloud,point_cloud_classical,point_cloud_classical_num
import os

edit = 4
indata = f"/home/ubuntu/Rencq/nerf_data/point_cloud/fern/opaque_3/eps007points40/out_point_cloud{edit}.txt"

output_path = '../logs'
input = np.loadtxt(indata)
input = torch.tensor(input,dtype=torch.float64)
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
model1 = point_cloud(3,1,1,256)
model2 = point_cloud_classical(256,output)

num_epochs=500
learning_rate=0.001
weight_decay=0.01
batch_size=4000



k_fold(model1,model2,4,X_train,y_train,num_epochs=num_epochs,learning_rate=learning_rate,weight_decay=weight_decay,batch_size=batch_size)

model_out_path =  os.path.join(output_path,'point_cloud')
if not os.path.exists(model_out_path):
    os.makedirs(model_out_path)

model1_out_filename = os.path.join(model_out_path,f'model1_{edit}.pth')
model2_out_filename = os.path.join(model_out_path,f'model2_{edit}.pth')
model1_out_param_filename = os.path.join(model_out_path,f'model1_param_{edit}.pth')
model2_out_param_filename = os.path.join(model_out_path,f'model2_param_{edit}.pth')

# torch.save(model1,model1_out_filename)
# torch.save(model2,model2_out_filename)
torch.save(model1.state_dict(),model1_out_param_filename)
torch.save(model2.state_dict(),model2_out_param_filename)