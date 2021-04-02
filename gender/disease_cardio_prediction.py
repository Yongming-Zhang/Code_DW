import numpy as np 
import pandas as pd 
import torch
import torch.optim as optimizer
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

# Get data
df = pd.read_csv('./cardio_train.csv', delimiter=';')
df.drop('id', axis=1, inplace=True)
df.head()

X = df.drop('cardio', axis=1)
Y = df['cardio']

X = normalize(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        x = self.hidden(x)
        x = torch.sigmoid(self.predict(x))
        return x

net = Net(11, 11, 1)
#print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
loss_func = torch.nn.BCELoss()  
batch_size = 256

net.train()
iteration=100
for i in range(iteration):
    x_train = torch.as_tensor(x_train, dtype=torch.float32)
    y_train = torch.as_tensor(y_train, dtype=torch.float32)
    y_train = y_train.reshape(y_train.shape[0], -1)
   
    out = net(x_train)  
    loss = loss_func(out, y_train) 
    optimizer.zero_grad()    
    loss.backward()        
    optimizer.step()         
    print('Iteration=%d, Loss=%.4f' % (i, loss))

    if (i+1) % 10 == 0: 
        net.eval()
        #import ipdb; ipdb.set_trace()
        x_test = torch.as_tensor(x_test, dtype=torch.float32)
        y_test = torch.as_tensor(y_test, dtype=torch.float32)
        y_test = y_test.reshape(y_test.shape[0], -1)

        y_pred = net(x_test)
        y_pred = torch.where(y_pred>=0.5,torch.full_like(y_pred, 1),torch.full_like(y_pred, 0)) 
        accuracy = float((y_pred == y_test).sum()) / float(y_test.shape[0])
        print('Accuracy=%.4f' % accuracy)
