import SimpleITK as sitk 
import cv2

filename = '/mnt/users/1036295/1036295_vessel.nii.gz'
ds = sitk.ReadImage(filename)
print(ds.GetSize())
img = sitk.GetArrayFromImage(ds)
frame_num, width, height = img.shape
print(img.shape)
'''
outpath = '../test/image'
index = -1
for i in img:
    index = index + 1
    cv2.imwrite('%s/%d.png'%(outpath, index), i)
print('Done!')
'''
