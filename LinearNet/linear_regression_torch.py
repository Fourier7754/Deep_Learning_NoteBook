import torch as pt
import numpy
import torch.utils.data as Data
from torch.nn import init
from linear_regression import *
import torch.optim as optim

class linear_network(pt.nn.Module):
    def __init__(self,data_dim):
        super(linear_network,self).__init__()
        self.linear = pt.nn.Linear(data_dim,1)

    def forward(self,x):
        y = self.linear(x)
        return y

class linear_regression_torch(linear_regression):
    def __init__(self,batch_size=None,learning_rate=None,epoch=None,data_dim=None):
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epoch = epoch
        self.net=linear_network(data_dim=data_dim)

    def generate_train_data(self,x,y):
        dataset = Data.TensorDataset(x,y)
        data_iter = Data.DataLoader(dataset,self.batch_size,shuffle=True)
        return data_iter

    def model_init(self):
        init.normal_(self.net.linear.weight,mean=1,std=0.01)
        init.constant_(self.net.linear.bias,val=0)
        self.loss_function=pt.nn.MSELoss()
        self.optimizer=optim.SGD(self.net.parameters(),lr=self.lr)

    def train_linear_model(self,x,y):
        for i in range(self.epoch):
            for X,Y in self.generate_train_data(x,y):
                loss=self.loss_function(self.net(X),Y.view(-1,1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print('loss=',loss.item())

if __name__ == '__main__':
    LinearNet=linear_regression_torch(batch_size=10,learning_rate=0.001,epoch=200,data_dim=3)
    x,y=LinearNet.generate_test_data(3)
    LinearNet.model_init()
    LinearNet.train_linear_model(x,y)
    print(LinearNet.net.linear.weight)

