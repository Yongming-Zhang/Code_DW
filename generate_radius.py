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
        transformed_data = ndimage.morphology.distance_transform_edt(data)
        out = sitk.GetImageFromArray(transformed_data)
        if not os.path.exists(transformed_path):
            os.makedirs(transformed_path)
        sitk.WriteImage(out, transformed_path + pa)
        
path = '/mnt/BrainDataNFS/dataset/ccta/db_update/others/crop_cardiac_zoom/vessel/'
transformed_path = '/mnt/BrainDataNFS/dataset/ccta/db_update/others/crop_cardiac_zoom/transformed_vessel_20201126/'
distance_transform(path, transformed_path)

