'''
import matplotlib
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D

example_filename = '/mnt/coronarygroup/mask/0657669_0000.nii.gz'
img = nib.load(example_filename)
print(img)
print(img.header['db_name'])  # 输出头信息

#shape有四个参数 patient001_4d.nii.gz
#shape有三个参数 patient001_frame01.nii.gz   patient001_frame12.nii.gz
#shape有三个参数  patient001_frame01_gt.nii.gz   patient001_frame12_gt.nii.gz
width, height, queue = img.dataobj.shape
OrthoSlicer3D(img.dataobj).show()

num = 1
for i in range(1, queue, 10):
    img_arr = img.dataobj[:, :, 1]
    plt.subplot(5, 4, num)
    plt.imshow(img_arr, cmap='gray')


plt.show()
'''
'''
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
example_filename = '/mnt/coronarygroup/mask/0657669_0000.nii.gz'
img = nib.load(example_filename)
OrthoSlicer3D(img.dataobj).show()

import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
 
lena = mpimg.imread('/mnt/users/code/1.jpg') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)
 
plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()
'''
'''
import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
plt.imshow(np.random.randint(0, 2, (2, 2)))
plt.show()
'''

import tkinter as Tk
import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2 

img = Image.open('1.jpg')
print(img.size,img.format)
print(img)
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.figure('picture')
plt.imshow(img)
plt.show()
#plt.savefig('2.jpg')
img = cv2.imread('2.jpg')
cv2.imshow("img", img)
cv2.waitKey(0)    
cv2.destroyAllWindows() 

