#-*-coding:utf-8-*-
import numpy as np
import os
import glob
import SimpleITK as sitk
from skimage import measure
import nibabel as nib
from mitok.utils.mdicom import SERIES
import cv2
from scipy import ndimage
from skimage import morphology
import skimage
import pandas as pd
import json
from skimage.draw import line
from numpy.linalg import norm
import skimage.draw
import math
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import torch
from mitok.image.cv_gpu import edt
from scipy.ndimage import gaussian_filter
import shutil
import math
import numbers
from torch import nn
from skimage.morphology import ball
from torch.nn import functional as F

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, kernel_size, sigma, dim=2, channels=1,
                 padmode='constant', padvalue=0):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.pad = []
        for i in range(len(kernel_size)-1, -1, -1):
            self.pad.append(kernel_size[i]//2)
            self.pad.append(kernel_size[i]//2)
        self.pad = tuple(self.pad)
        self.padmode = padmode
        self.padvalue = padvalue

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, x):
        """
        Apply gaussian filter to x.
        Arguments:
            x (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """

        x = torch.unsqueeze(x, dim=1)
        x = F.pad(x, self.pad, mode=self.padmode, value=self.padvalue)
        x = self.conv(x, weight=self.weight.to(x.device), groups=self.groups)
        return torch.squeeze(x, dim=1)

def save_nii(plaque, data, nii_name):
    dealt_plaque_dir = plaque.split('mask_pipeline_vessel_plaque_middlepoint.nii.gz')[0]
    #dcm_folder = plaque.split('_CTA')[0]
    #series = SERIES(series_path=dcm_folder, strict_check_series=True)
    affine_arr = np.eye(4)
    #affine_arr[0][0] = series.pixel_spacing[0]
    #affine_arr[1][1] = series.pixel_spacing[1]
    #affine_arr[2][2] = series.slice_spacing
    #print(data[np.nonzero(data)])
    #data[np.nonzero(data)] = 1
    data = np.transpose(data, (2, 1, 0))
    data = data.astype('float32')

    #对血管进行高斯平滑
    gauss_data = data.copy()
    gauss_data[gauss_data==2] = 0
    #gauss_data = gaussian_filter(gauss_data, sigma=1, order=0, output=None, mode="reflect", cval=0.0, truncate=4.0)
    gauss_data = torch.as_tensor(gauss_data).cuda()
    gauss_data = torch.unsqueeze(gauss_data, dim=0)
    gaussiansmoothing = GaussianSmoothing(5, 1, dim=3)
    gauss_data = gaussiansmoothing.forward(gauss_data)
    gauss_data = torch.squeeze(gauss_data, dim=0)
    gauss_data = gauss_data.cpu().numpy()
    print(np.unique(gauss_data))
    gauss_data = (gauss_data>=0.5).astype(np.float32)
    gauss_data[np.logical_and(data==2, np.logical_not(gauss_data))] = 2
    print(np.unique(gauss_data))

    plaque_nii = nib.Nifti1Image(gauss_data, affine_arr)
    nib.save(plaque_nii, os.path.join(dealt_plaque_dir, nii_name)) 

#plaque_dir = '/mnt/users/ffr_plaque_mask/'
plaque_dir = '/mnt/public/zhangyongming/ffr_cpr_mask_newmap_dealt'
vessel_dir = '/mnt/DrwiseDataNFS/drwise_runtime_env/data1/inputdata' 
#plaque_list = glob.glob(os.path.join(plaque_dir, '*', '*', 'mask_plaque.nii.gz'))
plaque_list = glob.glob(os.path.join(plaque_dir, '*', '*', 'mask_pipeline_vessel_plaque_middlepoint.nii.gz'))
#vessel_list = glob.glob(os.path.join(vessel_dir, plaque.split('/')[4], '*', plaque.split('/')[5]+'_CTA', 'mask_source/mask_vessel.nii.gz'))
plaques = []
#broken_vessels = [1073332, 1073332, 1036627, 1036604, 1073308, 1036623, 1073309, 1036609, 1073297, 1073318, 1073318, 1036617, 1036617, 1073300, 1073298, 1022836]
broken_vessels = ['1036623']#['1036604_60_0416', '1073309', '1036617']#['1036604_60_0416', '1036623', '1073309', '1073330', '1073318', '1036617', '1073332']#['1036623', '1073309', '1073332', '1073330', '1073318', '1036617'] #[1036604_60_0416]
for plaque in plaque_list:
    #print(plaque.split('/')[4])
    if plaque.split('/')[5] in broken_vessels: #int(plaque.split('/')[4]) > 0: 
        #continue
        #plaque = '/mnt/users/ffr_plaque_mask/1073318/AF7B89E9/mask_plaque.nii.gz'
        print(plaque)
        plaques.append(plaque)
        plaque_data = sitk.ReadImage(plaque)
        plaque_data = sitk.GetArrayFromImage(plaque_data)
        save_nii(plaque, plaque_data, 'mask_pipeline_vessel_plaque_gaussian.nii.gz')
        