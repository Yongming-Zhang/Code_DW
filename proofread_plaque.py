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

def connected_domain(image, mask=True):
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    _input = sitk.GetImageFromArray(image.astype(np.uint8))
    output_ex = cca.Execute(_input)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(output_ex)
    num_label = cca.GetObjectCount()
    num_list = [i for i in range(1, num_label+1)]
    area_list = []
    for l in range(1, num_label +1):
        area_list.append(stats.GetNumberOfPixels(l))
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    largest_area = area_list[num_list_sorted[0] - 1]
    final_label_list = [num_list_sorted[0]]

    for idx, i in enumerate(num_list_sorted[1:]):
        if area_list[i-1] >= (largest_area//10):
            final_label_list.append(i)
        else:
            break
    output = sitk.GetArrayFromImage(output_ex)

    for one_label in num_list:
        if  one_label in final_label_list:
            continue
        x, y, z, w, h, d = stats.GetBoundingBox(one_label)
        one_mask = (output[z: z + d, y: y + h, x: x + w] != one_label)
        output[z: z + d, y: y + h, x: x + w] *= one_mask

    if mask:
        output = (output > 0).astype(np.uint8)
    else:
        output = ((output > 0)*255.).astype(np.uint8)
    return output

def connected_component(image):
    # 标记输入的3D图像
    label, num = measure.label(image, connectivity=2, return_num=True)
    if num < 1:
        return [], image
        
    # 获取对应的region对象
    region = measure.regionprops(label)
    # 获取每一块区域面积并排序
    num_list = [i for i in range(1, num+1)]
    area_list = [region[i-1].area for i in num_list]
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    # 去除面积较小的连通域
    '''
    if len(num_list_sorted) > 3:
        # for i in range(3, len(num_list_sorted)):
        for i in num_list_sorted[3:]:
            # label[label==i] = 0
            label[region[i-1].slice][region[i-1].image] = 0
        num_list_sorted = num_list_sorted[:3]
    '''
    return num_list_sorted, label
   
def np_count(nparray, x):
    i = 0
    for n in nparray:
        if n == x:
            i += 1
    return i

def remove_mix_region(num, volume, out_data):
    after_num = []
    after_volume = []
    for i in range(len(num)):
        if volume[i] > 0:
            after_num.append(num[i])
            after_volume.append(volume[i])
        else:
            out_data[out_data==num[i]] = 0
    return after_num, after_volume, out_data


def show_data(path):
    print(path)
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    print(data.shape)
    data = np.unique(data)
    print(data)

#先执行transform_proofread_plaque.py对校对后的mask_plaque.nii.gz的z轴进行翻转，再对翻转后的mask_plaque.nii.gz进行分析 
plaque_path = '/mnt/users/transformed_plaque' 
proofread_plaque_path = '/mnt/users/transformed_bad_case' #'/mnt/users/transformed_plaque_proofread' 
vessel_plaque_path = '/mnt/DrwiseDataNFS/drwise_runtime_env/data1/inputdata'
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
        num, out_data = connected_component(data)
        if len(num) != 0:
            volume = []
            for i in range(len(num)):
                volume.append(np_count(out_data[np.nonzero(out_data)], num[i]))
            print('原斑块为', num, volume)
        else:
            print('原来无斑块')
            
        proofread_series_dir = os.path.join(proofread_plaque_path, plaque, series, 'mask_plaque.nii.gz')
        print(proofread_series_dir)
        new_data = sitk.ReadImage(proofread_series_dir)
        new_data = sitk.GetArrayFromImage(new_data)
        num, out_data = connected_component(new_data)
        if len(num) != 0:
            volume = []
            for i in range(len(num)):
                volume.append(np_count(out_data[np.nonzero(out_data)], num[i]))
            print('校对后的斑块为', num, volume)
        else:
            print('校对后无斑块')

        vessel_plaque_dir = glob.glob(os.path.join(vessel_plaque_path, plaque, '*', series+'_CTA', 'mask_source', 'mask_vessel.nii.gz'))[0]
        vessel_plaque_data = sitk.ReadImage(vessel_plaque_dir)
        vessel_plaque_data = sitk.GetArrayFromImage(vessel_plaque_data)
        vessel_plaque_data[np.nonzero(new_data)] = 2

        affine_arr = np.eye(4)
        vessel_plaque_data = np.transpose(vessel_plaque_data, (2, 1, 0))
        vessel_plaque_data = vessel_plaque_data.astype('float32')
        vessel_plaque_nii = nib.Nifti1Image(vessel_plaque_data, affine_arr)
        nib.save(vessel_plaque_nii, os.path.join(proofread_plaque_path, plaque, series, 'mask_vessel_plaque_proofread.nii.gz')) 

