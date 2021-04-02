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

# Get data
df = pd.read_csv('./cardio_train.csv', delimiter=';')
df.drop('id', axis=1, inplace=True)
#df.head()

X = df.drop('cardio', axis=1)
Y = df['cardio']

#X = normalize(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
x_train = np.asarray(x_train)
x_train = torch.tensor(x_train, dtype=torch.float32).cuda()
x_test = np.asarray(x_test)
x_test = torch.tensor(x_test, dtype=torch.float32).cuda()
y_train = np.asarray(y_train)
y_train = torch.tensor(y_train, dtype=torch.int64).cuda()
y_test = np.asarray(y_test)
y_test = torch.tensor(y_test, dtype=torch.int64).cuda()

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

net = Net(28*28, 294, 10)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_func = torch.nn.CrossEntropyLoss()  
batch_size = 256

#train = TensorDataset(x_train, y_train)
train = torchvision.datasets.MNIST('./mnist',train=True,transform = torchvision.transforms.ToTensor(),download=False)
#train = torchvision.datasets.ImageFolder('/mnt/users/code/CellData/chest_xray/train',transform = torchvision.transforms.ToTensor())
train_loader = DataLoader(dataset=train,batch_size=batch_size, shuffle=True)

#test = TensorDataset(x_test, y_test)
test = torchvision.datasets.MNIST('./mnist',train=False,transform = torchvision.transforms.ToTensor(),download=False)
#test = torchvision.datasets.ImageFolder('/mnt/users/code/CellData/chest_xray/test',transform = torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test,batch_size=1024)

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
net = net.to(device)
epoch=50

for e in range(epoch): 
    net.train()
    for i, (x_train, y_train) in enumerate(train_loader):
        optimizer.zero_grad()
        #print(x_train.size())
        x_train = x_train.view(-1, 28*28)
        #print(x_train)
        #print(y_train)
        #y_train = y_train.reshape(y_train.shape[0], -1)
        
        out = net(x_train.cuda())  
        #out = out.reshape(out.shape[0], -1)
        #print(out)
        #print(y_train)
        loss = loss_func(out, y_train.cuda()) 
        loss.backward()        
        optimizer.step()    
        if (i+1) % 10 == 0:    
            print('Epoch:{}[{}/{}] ({:.0f}%), Loss={:.4f}'.format(e, (i+1)*len(x_train), len(train_loader.dataset), 100 * (i+1)/len(train_loader), loss.item()))

    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.view(-1, 28*28)
            y_pred = net(x_test.cuda())
            test_loss += loss_func(y_pred, y_test.cuda())
            y_pred = torch.argmax(y_pred, dim=1)
            #y_pred = torch.where(y_pred>=0.5,torch.full_like(y_pred, 1),torch.full_like(y_pred, 0)) 
            correct += y_pred.eq(y_test.cuda()).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct/len(test_loader.dataset)))