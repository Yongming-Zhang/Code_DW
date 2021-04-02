import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D

#data = nib.load('0657735_0002.nii.gz').get_data()
#plt.imshow(data)
data = '0657735_0002.nii.gz'
img = nib.load(data)
print('img',img)
print('header',img.header['db_name'])

width, height, queue = img.dataobj.shape
OrthoSlicer3D(img.dataobj).show()

num = 1
for i in range(0, queue, 10):
    im = img.dataobj[:, :, i]
    plt.subplot(5, 4, num)
    plt.imshow(im, cmap='gray')
    num += 1

plt.show()