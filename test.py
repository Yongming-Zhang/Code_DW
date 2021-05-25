import torch
import numpy as np
from scipy.interpolate import bisplrep, bisplev
from scipy.interpolate import interpn

w = torch.Tensor([4., 8., 16., 32., 64.])
w_r = torch.Tensor([2., 1., 0.5])
s = torch.Tensor([8., 16., 32., 64., 128.])
a =  w * w_r[:,None]
print(w_r[:,None])
print(a)
a = s[None,:] * w_r[:,None]
print(s) 
print(a)
h = w * w_r[:,None] * s[None,:]
h_ = h.view(-1)
print(h)
print(h_)
x = torch.Tensor([1,2,3,4])
y = torch.Tensor([1,2,3,4])
xx = x.repeat(len(y))
print(xx)
yy = y.view(-1,1).repeat(1,len(x)).view(-1)
print(y.view(-1,1).repeat(1,len(x)))
print(xx,yy)
a = torch.Tensor([1,2])
b = torch.Tensor([[2,4],[2,3]])
v =a[:,None].expand(2,b.size(0)).transpose(0,1).contiguous().is_contiguous()
print(v)
m = torch.Tensor([[1], [2], [3], [4]])
print(m)
x = torch.Tensor([[4., 8., 16., 32.]])
s = x.mm(m)
print(s)
x = torch.arange(1,11).reshape(2,5)
print("number elements of x is ",x,x.numel())
y = torch.randn(3,10,5)
print("number elements of y is ",y.numel())
print(x.new_full((2,5),-1))
z = torch.arange(10).reshape(2,5)
v = z == 2
print(z==2)
print(x[v])
a = torch.rand(3,4)
b = torch.randn(1,4)
c = torch.cat([a],dim=0)
print('a',a,a.detach())
print(c)
x = c.sort()
print('x',c.sort()[0])
c = b.permute(1,0)
print(c,c[0].item())
i = torch.rand(2,3,4)
print(i,i.flatten(1))
def function(arg,*args,**kwargs):
    print(arg,args,kwargs)
a = dict(p='a')
b = dict()
b[1] = 10
function(5,6,7,8,9,'a',1, a, b, b=2, c=3)
index = dict()
for i in range(80):
    index[i] = i
a = [[1,2,3],[1,2,3],[3,2,1],[2,3,4]]
b = []
for i in a :
    if i not in b:
        b.append(i)
print(b)

def softmax(X):
    """
    softmax函数实现
    
    参数：
    x --- 一个二维矩阵, m * n,其中m表示向量个数，n表示向量维度
    
    返回：
    softmax计算结果
    """
    assert(len(X.shape) == 2)
    row_max = np.max(X, axis=1).reshape(-1, 1)
    X -= row_max
    X_exp = 1-np.exp(X)+1e-4
    s = X_exp / np.sum(X_exp, axis=1, keepdims=True)

    return s

a = [[1,2,3],[-1,-2,-3]]
b = [[1,2,3,4]]
a = np.array(a)
b = np.array(b)
print(softmax(b))

print('-----------------')
plaque_cta = np.zeros((512,512,512), dtype='uint8')
plaque_cta[0,0,0] = 1
plaque_cta[0,0,1] = 1
plaque_cta[0,1,0] = 1
plaque_cta[0,1,1] = 1
plaque_cta[1,0,0] = 1
plaque_cta[1,0,1] = 1
plaque_cta[1,0,1] = 1
plaque_cta[1,1,0] = 1
plaque_cta[1,1,1] = 1
plaque_cta[2,2,2] = 1
plaque_cta[2,2,3] = 1
plaque_cta[2,3,2] = 1
plaque_cta[2,3,3] = 1
plaque_cta[3,2,2] = 1
plaque_cta[3,2,3] = 1
plaque_cta[3,3,2] = 1
plaque_cta[3,3,3] = 1
plaque_cta[0,213,56] = 1
plaque_cta[0,213,57] = 1
plaque_cta[0,214,56] = 1
plaque_cta[0,214,57] = 1
plaque_cta[1,213,56] = 1
plaque_cta[1,214,56] = 1
plaque_cta[1,213,57] = 1
plaque_cta[1,214,57] = 1
plaque_cta[256,256,256] = 1
plaque_cta[11,11,11] = 1
plaque_cta[10,10,10] = 1
plaque_cta = plaque_cta.reshape(1, 1, plaque_cta.shape[0], plaque_cta.shape[1], plaque_cta.shape[2]).astype(np.float32)
plaque_cta = torch.as_tensor(plaque_cta).cuda()
cpr_coord = np.array([[0,0,0],[2.5,2.5,2.5],[2,2,2],[1.5,1.5,1.5],[0.5,0.5,0.5],[0.9,1,1],[2,0.5,0.5],[0,0,0],[1,1.1,2],[-0.5,-1,-1],[-100,-100,-100],[0.7,0.7,0.7],[3,3,3],[-1,0,0],[0.1,213.7,56.3],[256,256,256],[11,11,11],[10,10,10]]).reshape(3,6,3)
print('cpr_coord', cpr_coord, cpr_coord.shape)
tmp = cpr_coord.reshape([-1, 3])
print('cprmaxmin',tmp.max(axis=0),tmp.min(axis=0))
cpr_coord = (cpr_coord/np.asarray(plaque_cta.shape[2:])[None,None])*2.0 - 1.0
print(cpr_coord)
print(cpr_coord.max(),cpr_coord.min())
cpr_coord = cpr_coord.reshape(1, 1, cpr_coord.shape[0], cpr_coord.shape[1], cpr_coord.shape[2]).astype(np.float32)
cpr_coord = torch.as_tensor(cpr_coord).cuda()
cpr_data = torch.nn.functional.grid_sample(plaque_cta, cpr_coord, mode='bilinear', padding_mode='border', align_corners=True)
cpr_data = torch.squeeze(cpr_data)
cpr_data = torch.squeeze(cpr_data)
cpr_data = torch.squeeze(cpr_data)
print(cpr_data.shape)
cpr_data = cpr_data.cpu().numpy()
print(cpr_data)
#print(np.unique(cpr_data))

a = np.array([1,4,2,3])
b = np.argmin(a, axis=0)
print(b)

import re
_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

list1 = ["1", "100A", "342B", "2C", "132", "36", "302F"]
list1.sort(key=natural_sort_key)
print(list1)

list2 = ['coro001_0', 'coro001_108', 'coro001_117', 'coro001_126', 'coro001_135', 'coro001_144', 'coro001_153', 'coro001_162', 'coro001_171', 'coro001_18', 'coro001_27', 'coro001_36', 'coro001_45', 'coro001_54', 'coro001_63', 'coro001_72', 'coro001_81', 'coro001_9', 'coro001_90', 'coro001_99']
list2 = sorted(list2, key=natural_sort_key)
print(list2)
b = np.zeros((2,2))
def res(a):
    a[0,0] = 11
    return a
res(b)
print(b)

plaque_cta = np.zeros((512,512,512), dtype='uint8')
plaque_cta[0,0,0] = 1
plaque_cta[0,0,1] = 1
plaque_cta[0,1,0] = 1
plaque_cta[0,1,1] = 1
plaque_cta[1,0,0] = 1
plaque_cta[1,0,1] = 1
plaque_cta[1,0,1] = 1
plaque_cta[1,1,0] = 1
plaque_cta[1,1,1] = 1
plaque_cta[2,2,2] = 1
plaque_cta[2,2,3] = 1
plaque_cta[2,3,2] = 1
plaque_cta[2,3,3] = 1
plaque_cta[3,2,2] = 1
plaque_cta[3,2,3] = 1
plaque_cta[3,3,2] = 1
plaque_cta[3,3,3] = 1
x = plaque_cta[:,:,0]
y = plaque_cta[:,:,1]
z = plaque_cta[:,:,2]
print(x)
points = (x, y, z)
print(points)
print('-----')
#point = np.array([2.21, 3.12, 1.15])
#print(interpn(points, plaque_cta, point))
