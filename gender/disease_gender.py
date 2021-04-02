import numpy as np 
import pandas as pd 
import torch
import torch.optim as optimizer
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from sklearn.preprocessing import StandardScaler

def standartization(x, s_list):
    x_std = x.copy(deep=True)
    for column in s_list:
        x_std[column] = (x_std[column]-x_std[column].mean())/x_std[column].std()
    return x_std 

def MF(X, Y):
    s_list = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
    X=standartization(X, s_list)

    #X = normalize(X)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    x_train = np.asarray(x_train)
    x_train = torch.tensor(x_train, dtype=torch.float32).cuda()
    x_test = np.asarray(x_test)
    x_test = torch.tensor(x_test, dtype=torch.float32).cuda()
    y_train = np.asarray(y_train)
    y_train = torch.tensor(y_train, dtype=torch.int64).cuda()
    y_test = np.asarray(y_test)
    y_test = torch.tensor(y_test, dtype=torch.int64).cuda()
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

    return x_train, y_train, x_test, y_test

def MR_FR(X, Y):
    s_list = ["age", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
    X=standartization(X, s_list)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.35, random_state=42) #test_size=0.65

    x_train = np.asarray(x_train)
    x_train = torch.tensor(x_train, dtype=torch.float32).cuda()
    y_train = np.asarray(y_train)
    y_train = torch.tensor(y_train, dtype=torch.int64).cuda()   
    x_test = np.asarray(x_test)
    x_test = torch.tensor(x_test, dtype=torch.float32).cuda()
    y_test = np.asarray(y_test)
    y_test = torch.tensor(y_test, dtype=torch.int64).cuda()
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

    return x_train, y_train, x_test, y_test

def M_F(X, Y):
    s_list = ["age", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
    X=standartization(X, s_list)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X = pd.concat([x_train, y_train], axis=1)
    print(X)
    x_train = X[X['gender']==2] #x_train = X[X['gender']==1]
    print(x_train)
    y_train = x_train['cardio']
    x_train = x_train.drop('cardio', axis=1)

    x_train = np.asarray(x_train)
    x_train = torch.tensor(x_train, dtype=torch.float32).cuda()
    y_train = np.asarray(y_train)
    y_train = torch.tensor(y_train, dtype=torch.int64).cuda()   
    x_test = np.asarray(x_test)
    x_test = torch.tensor(x_test, dtype=torch.float32).cuda()
    y_test = np.asarray(y_test)
    y_test = torch.tensor(y_test, dtype=torch.int64).cuda()
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

    return x_train, y_train, x_test, y_test

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.relu(x)
        x = self.predict(x)

        return x

def train(net, train_loader, optimizer, epoch):
    net.train()
    for i, (x_train, y_train) in enumerate(train_loader):
        #print(x_train.shape,y_train.shape)
        optimizer.zero_grad()
        out = net(x_train)  
        loss = loss_func(out, y_train) 
        loss.backward()        
        optimizer.step()    
        if (i+1) % 10 == 0:    
            print('Epoch:{}[{}/{}] ({:.0f}%), Loss={:.4f}'.format(epoch, (i+1)*len(x_train), len(train_loader.dataset), 100 * (i+1)/len(train_loader), loss.item()))

def test(net, test_loader):
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            y_pred = net(x_test)
            test_loss += loss_func(y_pred, y_test)
            y_pred = torch.argmax(y_pred, dim=1)
            #print('y_pred',y_pred)
            #y_pred = torch.where(y_pred>=0.5,torch.full_like(y_pred, 1),torch.full_like(y_pred, 0)) 
            correct += y_pred.eq(y_test).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct/len(test_loader.dataset)))

net = Net(11, 121, 2)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.CrossEntropyLoss()  
batch_size = 256

df = pd.read_csv('./cardio_train.csv', delimiter=';')
df.drop('id', axis=1, inplace=True)
#df.head()
'''
df[df['ap_lo'] >= df['ap_hi']]
df.drop(df[df["ap_lo"] > df["ap_hi"]].index, inplace=True)
df.drop(df[df["ap_lo"] <= 30].index, inplace=True)
df.drop(df[df["ap_hi"] <= 40].index, inplace=True)
df.drop(df[df["ap_lo"] >= 200].index, inplace=True)
df.drop(df[df["ap_hi"] >= 250].index, inplace=True)
'''
Y = df['cardio']
X = df.drop('cardio', axis=1)

x_train, y_train, x_test, y_test = MR_FR(X, Y)
#import ipdb; ipdb.set_trace()
train_data = TensorDataset(x_train, y_train)
#train = torchvision.datasets.MNIST('./mnist',train=True,transform = torchvision.transforms.ToTensor(),download=False)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = TensorDataset(x_test, y_test)
#test = torchvision.datasets.MNIST('./mnist',train=False,transform = torchvision.transforms.ToTensor(),download=False)
test_loader = DataLoader(dataset=test_data, batch_size=1024)

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
net = net.to(device)
epochs = 50

for epoch in range(epochs): 
    train(net, train_loader, optimizer, epoch)
    test(net, test_loader)
    
    