import SimpleITK as sitk
import skimage.io as io
from itkwidgets import view
import itk
from scipy import ndimage
import numpy as np
import os
from tqdm import tqdm

def distance_transform(path, transformed_path):
    path_list = os.listdir(path)
    for pa in tqdm(path_list):
        img = sitk.ReadImage(path + pa)
        data = sitk.GetArrayFromImage(img)
        #nonzero_data = np.nonzero(data)
        #print('data', data[nonzero_data])
        transformed_data = ndimage.morphology.distance_transform_edt(data)
        #print('transformed_data', transformed_data)
        #nonzero_transformed_data = np.nonzero(transformed_data)
        #print('transformed_data', transformed_data[nonzero_transformed_data])
        out = sitk.GetImageFromArray(transformed_data)
        # out.SetSpacing(transformed_data.GetSpacing())
        # out.SetOrigin(transformed_data.GetOrigin())
        sitk.WriteImage(out, transformed_path + pa)

def show_data(path):
    path_list = os.listdir(path)
    print('start')
    for pa in tqdm(path_list):
        img = sitk.ReadImage(path + pa)
        data = sitk.GetArrayFromImage(img)
        data = np.unique(data)
        if 255 in data or len(data) != 3:
            print(data,img)
    print('end')

def show_da(path):
    print(path)
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    print(data.shape)
    data = np.unique(data)
    #if 255 in data or len(data) != 3:
    print(data)
        
path = '/mnt/BrainDataNFS/dataset/ccta/db_update/others/crop_cardiac_zoom/vessel/'
transformed_path = '/mnt/BrainDataNFS/dataset/ccta/db_update/others/crop_cardiac_zoom/transformed_vessel_20201110/0987885_0002.nii.gz'
#img = itk.imread(path)
#view(img)
#distance_transform(path, transformed_path)
show_da(transformed_path)
#img = itk.imread(transformed_path)
#view(img)
#data = read_img(path)
#show_imgs(data)