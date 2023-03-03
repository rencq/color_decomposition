import torch
import torch.utils.data
class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def try_all_gpus():  #@save
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


# K times
def get_k_fold_data(k,i,X,y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train,X_valid,y_valid = None, None,None,None
    for j in range(k):
        idx = slice(j*fold_size,(j+1) * fold_size)
        X_part,y_part = X[idx,:],y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train ,y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train,X_part],0)
            y_train = torch.cat([y_train,y_part],0)
    return X_train,y_train,X_valid,y_valid

#返回正确数量
def accuracy(y_hat,y):
    if len(y_hat.shape) >1 and y_hat.shape[1] >1:
        y_hat = torch.argmax(y_hat,dim=1)
    cmp = torch.tensor(y_hat,dtype=y.dtype) == y
    return float(torch.sum(cmp,dtype=y.dtype))

def train_batch(net,X,y,loss,trainer,devices):
    if isinstance(X,list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])

    #train
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred,y)
    l.sum().backward()
    trainer.step()
    train_acc_sum = accuracy(pred,y)
    return train_acc_sum

def train(net,train_features,train_labels,test_features,test_labels,num_epochs,
          learning_rate,weight_decay,batch_size,devices,is_train=True):
    dataset = torch.utils.data.TensorDataset(train_features,train_labels)
    train_iter = torch.utils.data.DataLoader(dataset,batch_size,shuffle=is_train,drop_last=True)
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=weight_decay)
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    net = torch.nn.DataParallel(net,device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        metric = Accumulator(2)
        for _,(X,y) in enumerate(train_iter):
            print("=========")
            acc = train_batch(net,X,y,loss,optimizer,devices)
            metric.add(acc,y.numel())
            print(f'epoch = {epoch} acc = {metric[0]/metric[1]}')


def k_fold(net,k,X_train,y_train,num_epochs,learning_rate,weight_decay,batch_size,devices=try_all_gpus()):
    # for i in range(k):
    #     data = get_k_fold_data(k,i,X_train,y_train)

    train(net,X_train,y_train,None,None,num_epochs=num_epochs,learning_rate=learning_rate,weight_decay=weight_decay,batch_size=batch_size,devices=devices)


