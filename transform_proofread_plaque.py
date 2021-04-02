import numpy as np
import os
import glob
import SimpleITK as sitk
from skimage import measure
import nibabel as nib
import itk
from itkwidgets import view
import seaborn as sns
from mitok.utils.mdicom import SERIES
import glob

#对校对后的mask_plaque.nii.gz的z轴进行翻转，保存下翻转后的mask_plaque.nii.gz
plaque_path = '/mnt/users/transformed_z_plaque' 
proofread_plaque_path = '/mnt/users/bad_case' #'/mnt/users/transformed_z_plaque_proofread' 
transformed_plaque_path = '/mnt/users/transformed_plaque' 
transformed_proofread_plaque_path = '/mnt/users/transformed_bad_case' #'/mnt/users/transformed_plaque_proofread' 
bad_plaque_list = ['1073332', '1073305', '1073318'] 
bad_series_list = ['5F654861', '5F653FE9', 'CD1B554E', 'AF7B8171', 'AF7B89E9']
plaque_list = os.listdir(plaque_path)
for plaque in plaque_list:
    if plaque.split('/')[-1] not in bad_plaque_list:
        continue
    plaque_dir = os.path.join(plaque_path, plaque)
    series_list = os.listdir(plaque_dir)
    for series in series_list:
        if series.split('/')[-1] not in bad_series_list:
            continue
        series_dir = os.path.join(plaque_dir, series, 'mask_plaque.nii.gz')
        print(series_dir)
        data = sitk.ReadImage(series_dir)
        data = sitk.GetArrayFromImage(data)
        data = data[::-1,:,:]
        data = np.transpose(data, (2, 1, 0))
        data = data.astype('float32')
        affine_arr = np.eye(4)
        data_nii = nib.Nifti1Image(data, affine_arr)
        if not os.path.exists(os.path.join(transformed_plaque_path, plaque, series)):
            os.makedirs(os.path.join(transformed_plaque_path, plaque, series))
        nib.save(data_nii, os.path.join(transformed_plaque_path, plaque, series, 'mask_plaque.nii.gz')) 

        proofread_series_dir = os.path.join(proofread_plaque_path, plaque, series, 'mask_plaque.nii.gz')
        print(proofread_series_dir)
        new_data = sitk.ReadImage(proofread_series_dir)
        new_data = sitk.GetArrayFromImage(new_data)
        new_data = new_data[::-1,:,:]
        new_data = np.transpose(new_data, (2, 1, 0))
        new_data = new_data.astype('float32')
        affine_arr = np.eye(4)
        new_data_nii = nib.Nifti1Image(new_data, affine_arr)
        if not os.path.exists(os.path.join(transformed_proofread_plaque_path, plaque, series)):
            os.makedirs(os.path.join(transformed_proofread_plaque_path, plaque, series))
        nib.save(new_data_nii, os.path.join(transformed_proofread_plaque_path, plaque, series, 'mask_plaque.nii.gz')) 

