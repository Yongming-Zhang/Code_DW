#-*-coding:utf-8-*-
import pydicom
import json
import matplotlib.pyplot as plt
from PIL import Image
import SimpleITK as sitk
import skimage.io as io
from mitok.utils.mdicom import SERIES
import nibabel as nib
import numpy as np
import os
import glob
import shutil
from scipy.interpolate import griddata as gd
#校对标注前，要对dicom进行判断，来确定nii.gz是否要翻转
plaque_mask_newmap_path = '/mnt/users/ffr_datasets/ffr_cpr_mask_newmap' #'/mnt/users/ffr_datasets/ffr_cpr_mask_newmap/'
new_plaque_path = '/mnt/users/ffr_plaque_mask/'
transformed_plaque_path = '/mnt/users/ffr_datasets/ffr_mask_transformed' #'/mnt/users/ffr_datasets/ffr_mask_transformed/'
name_list = ['1036604']#['1036623', '1073309', '1073332', '1073330', '1073318', '1036617']#['1036604']#['1036631', '1036632', '1036610', '1036619', '1036630', '1036615', '1036625', '1073332', '1036607', '1036624']
plaque_list = os.listdir(plaque_mask_newmap_path)
for plaque in plaque_list:
    if plaque not in name_list:
        continue
    print(plaque)
    series_list = os.listdir(os.path.join(plaque_mask_newmap_path, plaque))
    for series in series_list:
        #if series != '6566EAC0_CTA':
        #    continue
        print(series)
        mask_dir = os.path.join(plaque_mask_newmap_path, plaque, series, 'mask_plaque_centerline.nii.gz') #mask_plaque_newmap.nii.gz
        shutil.copy(mask_dir, os.path.join(new_plaque_path, plaque, series.split('_')[0]))

        mask_dir = os.path.join(plaque_mask_newmap_path, plaque, series, 'mask_plaque_direct.nii.gz') 
        shutil.copy(mask_dir, os.path.join(new_plaque_path, plaque, series.split('_')[0]))
        '''
        for i in range(17):
            mask_dir = os.path.join(plaque_mask_newmap_path, plaque, series, 'mask_plaque_cpr_regionint3%s.nii.gz'%i)
            shutil.copy(mask_dir, os.path.join(new_plaque_path, plaque, series.split('_')[0]))
        '''
        '''
        img = sitk.ReadImage(mask_dir)
        data = sitk.GetArrayFromImage(img)
        print(data.shape)
        affine_arr = np.eye(4)
        data = data[::-1,:,:]
        data = np.transpose(data, (2, 1, 0))
        data = data.astype('float32')
        plaque_nii = nib.Nifti1Image(data, affine_arr)
        if not os.path.exists(os.path.join(transformed_plaque_path, plaque, series)):
            os.makedirs(os.path.join(transformed_plaque_path, plaque, series))
        nib.save(plaque_nii, os.path.join(transformed_plaque_path, plaque, series, 'mask_plaque_centerline_transformed.nii.gz')) #mask_plaque_newmap_transformed.nii.gz 
        '''
        '''
        img = sitk.ReadImage(os.path.join(new_plaque_path, plaque, series.split('_')[0], 'mask_plaque.nii.gz'))
        data = sitk.GetArrayFromImage(img)
        print(data.shape)
        affine_arr = np.eye(4)
        data = data[::-1,:,:]
        data = np.transpose(data, (2, 1, 0))
        data = data.astype('float32')
        plaque_nii = nib.Nifti1Image(data, affine_arr)
        if not os.path.exists(os.path.join(transformed_plaque_path, plaque, series)):
            os.makedirs(os.path.join(transformed_plaque_path, plaque, series))
        nib.save(plaque_nii, os.path.join(transformed_plaque_path, plaque, series, 'mask_plaque_transformed.nii.gz'))
        '''