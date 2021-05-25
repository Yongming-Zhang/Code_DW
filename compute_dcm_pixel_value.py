#-*-coding:utf-8-*-
import os
import numpy as np
import glob
import cv2
from PIL import Image
import SimpleITK as sitk
import re
import pydicom
from PIL import Image

_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

plaque_mask_gen_path = '/mnt/users/ffr_datasets/ffr_cpr_mask_dealt/' 
mapping_gen_path = '/mnt/DrwiseDataNFS/drwise_runtime_env/data1/inputdata' 
name_list = ['1036604']
dicom_list = []
'''
for patient in sorted(os.listdir(plaque_mask_gen_path)):
    if patient not in name_list: 
        continue
    print(patient)
    for series in os.listdir(os.path.join(plaque_mask_gen_path, patient)):
        print(series)
        cpr_paths = glob.glob(os.path.join(mapping_gen_path, patient, '*', series, 'cprCoronary'))[0]
        cpr_paths = sorted(glob.glob(os.path.join(cpr_paths, '*.dcm')), key=natural_sort_key)
        cpr_names = [cpr_path.split('/')[-1].split('.')[0] for cpr_path in cpr_paths]
        print(cpr_names)

        cpr_paths_copy = glob.glob(os.path.join(mapping_gen_path, patient+'_60_0416', '*', series, 'cprCoronary'))[0]
        cpr_paths_copy = sorted(glob.glob(os.path.join(cpr_paths_copy, '*.dcm')), key=natural_sort_key)
        cpr_names_copy = [cpr_path.split('/')[-1].split('.')[0] for cpr_path in cpr_paths_copy]
        print(cpr_names_copy)

        for cpr in cpr_paths:
            cpr_name = cpr.split('/')[-1].split('.')[0]
            if cpr_name in cpr_names_copy:
                print(cpr_name, cpr)
                ds_60 = pydicom.read_file(cpr)
                ds_20 = pydicom.read_file(cpr.split(patient)[0]+patient+'_60_0416'+cpr.split(patient)[1])
                value = ds_60.pixel_array-ds_20.pixel_array
                print(value, np.unique(value))
                if np.unique(value) != 0:
                    print(cpr_name)
                    dicom_list.append(cpr)
print(dicom_list)
'''
for patient in sorted(os.listdir(plaque_mask_gen_path)):
    if patient not in name_list: 
        continue
    print(patient)
    for series in os.listdir(os.path.join(plaque_mask_gen_path, patient)):
        print(series)
        cpr_paths = glob.glob(os.path.join(mapping_gen_path, patient, '*', series, 'diagnose_debug/segmentation'))[0]
        cpr_paths = sorted(glob.glob(os.path.join(cpr_paths, '*.png')), key=natural_sort_key)
        cpr_names = [cpr_path.split('/')[-1].split('.')[0] for cpr_path in cpr_paths]
        print(cpr_names)

        cpr_paths_copy = glob.glob(os.path.join(mapping_gen_path, patient+'_60_0416', '*', series, 'diagnose_debug/segmentation'))[0]
        cpr_paths_copy = sorted(glob.glob(os.path.join(cpr_paths_copy, '*.png')), key=natural_sort_key)
        cpr_names_copy = [cpr_path.split('/')[-1].split('.')[0] for cpr_path in cpr_paths_copy]
        print(cpr_names_copy)

        for cpr in cpr_paths:
            cpr_name = cpr.split('/')[-1].split('.')[0]
            if cpr_name in cpr_names_copy:
                print(cpr_name, cpr)
                ds_60 = Image.open(cpr)
                ds_60 = np.array(ds_60)
                ds_20 = Image.open(cpr.split(patient)[0]+patient+'_60_0416'+cpr.split(patient)[1])
                ds_20 = np.array(ds_20)
                value = ds_60-ds_20
                print(value, np.unique(value))
                if len(np.unique(value)) != 1:
                    print(cpr_name)
                    dicom_list.append(cpr)
print(dicom_list, len(dicom_list))