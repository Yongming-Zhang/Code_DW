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
df = pd.read_csv('./indian_liver_patient.csv', delimiter=',')
df['Gender'] = df['Gender'].apply(lambda x:1 if x=='Male' else 0)
#print(df.head())

s_list = ["Age", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase", 
"Alamine_Aminotransferase","Aspartate_Aminotransferase","Total_Protiens","Albumin","Albumin_and_Globulin_Ratio"]
def standartization(x):
    x_std = x.copy(deep=True)
    for column in s_list:
        x_std[column] = (x_std[column]-x_std[column].mean())/x_std[column].std()
    return x_std 
df=standartization(df)
print(df.head())

#print(np.isnan(df).any())
df.dropna(inplace=True)
X = df.drop('Dataset', axis=1)
df['Dataset'] = df['Dataset'].apply(lambda x:0 if x==1 else 1)
Y = df['Dataset']

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

net = Net(10, 100, 2)

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
loss_func = torch.nn.CrossEntropyLoss()  
batch_size = 25

train = TensorDataset(x_train, y_train)
#print('x_train',x_train.shape,x_train)
#print('y_train',y_train.shape,y_train)
#train = torchvision.datasets.MNIST('./mnist',train=True,transform = torchvision.transforms.ToTensor(),download=False)
train_loader = DataLoader(dataset=train,batch_size=batch_size, shuffle=True)

test = TensorDataset(x_test, y_test)
#test = torchvision.datasets.MNIST('./mnist',train=False,transform = torchvision.transforms.ToTensor(),download=False)
test_loader = DataLoader(dataset=test,batch_size=25, shuffle=True)

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
net = net.to(device)
epoch=50

for e in range(epoch): 
    net.train()
    for i, (x_train, y_train) in enumerate(train_loader):
        #print('x_train_la',x_train.shape,x_train)
        #print('y_train_la',y_train.shape,y_train)
        optimizer.zero_grad()
        #print('y_train',y_train.shape,y_train)
        out = net(x_train)  
        #print('out',out.shape,out)
        loss = loss_func(out, y_train) 
        loss.backward()        
        optimizer.step()    
        if (i+1) % 2 == 0:    
            print('Epoch:{}[{}/{}] ({:.0f}%), Loss={:.4f}'.format(e, (i+1)*len(x_train), len(train_loader.dataset), 100 * (i+1)/len(train_loader), loss.item()))

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
            correct += y_pred.eq(y_test.view_as(y_pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct/len(test_loader.dataset)))