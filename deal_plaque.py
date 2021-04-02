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
    '''
    print(num_list_sorted ,output[np.nonzero(output)])
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
    '''
    print(num_list_sorted, output[np.nonzero(output)])
    return num_list_sorted, output

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

def save_nii(plaque, data):
    dealt_plaque_dir = plaque.split('mask_source/')[0]
    dcm_folder = plaque.split('_CTA')[0]
    series = SERIES(series_path=dcm_folder, strict_check_series=True)
    affine_arr = np.eye(4)
    affine_arr[0][0] = series.pixel_spacing[0]
    affine_arr[1][1] = series.pixel_spacing[1]
    affine_arr[2][2] = series.slice_spacing
    #print(data[np.nonzero(data)])
    data[np.nonzero(data)] = 1
    data = np.transpose(data, (2, 1, 0))
    data = data.astype('float32')
    plaque_nii = nib.Nifti1Image(data, affine_arr)
    nib.save(plaque_nii, os.path.join(dealt_plaque_dir, 'mask_source/mask_plaque_dealt.nii.gz'))            
    
def get_brightness(after_num, after_data, img_tensor):
    brightness = []
    for i in after_num:
        brightness.append(round(img_tensor[after_data==i].mean()))
        #print(img_tensor[after_data==i])
    return brightness

def remove_mix_plaque(after_data, after_num, after_volume, brightness, plaque_volumes_mean, plaque_brightness_mean):
    dealt_num = []
    dealt_data = after_data.copy()
    for i in range(len(after_num)):
        if after_volume[i] < plaque_volumes_mean or brightness[i] < plaque_brightness_mean:
            dealt_data[dealt_data==after_num[i]] = 0
            dealt_num.append(after_num[i])
    #print('dealt_data',dealt_data[np.nonzero(dealt_data)])
    #print('after_data',after_data[np.nonzero(after_data)])
    return dealt_data, dealt_num

def plaque_erode(dealt_data):
    dealt_data[np.nonzero(dealt_data)] = 1
    print(dealt_data[np.nonzero(dealt_data)])

    struct = ndimage.generate_binary_structure(3, 1)
    #kernel = skimage.morphology.ball(1, dtype=np.int)
    #dealt_data = ndimage.binary_dilation(dealt_data, struct, 1).astype(dealt_data.dtype)
    # Scipy erosion
    erosion_data = ndimage.binary_erosion(dealt_data).astype(dealt_data.dtype)
    #erosion_data = morphology.erosion(dealt_data, morphology.ball(3, dtype=np.int))
    # Scikit-image erosion
    #erosion_data = morphology.binary_erosion(dealt_data.astype(uint), struct)

    #设置卷积核5*5
    #kernel = np.ones((5,5), np.uint8)
    #dealt_data = cv2.cvtColor(dealt_data, cv2.COLOR_BGR2GRAY)
    #图像的腐蚀，默认迭代次数
    #erosion_data = cv2.erode(dealt_data, kernel)
    #图像的膨胀
    #dilate_data = cv2.dilate(erosion, kernel)
    return erosion_data

def gengerate_plaque(after_data, dealt_num, erosion_data):
    for i in range(len(dealt_num)):
        erosion_data[after_data==dealt_num[i]] = 1
    return erosion_data

plaque_dir = '/mnt/DrwiseDataNFS/drwise_runtime_env/data1/inputdata' 
plaque_list = glob.glob(os.path.join(plaque_dir, '*', '*', '*'+'_CTA', 'mask_source/mask_plaque.nii.gz'))
counts = 0
without_plaque_list = []
all_plaque_volumes = []
all_plaque_brightness = []
for plaque in plaque_list:
    #print(plaque.split('/')[6])
    if int(plaque.split('/')[6]) < 1020000:
        continue
    if counts == 16:
        break
    print(plaque)
    data = sitk.ReadImage(plaque)
    data = sitk.GetArrayFromImage(data)
    #print(data)
    #num, out_data = connected_domain(data)
    num, out_data = connected_component(data)
    if len(num) != 0:
        volume = []
        for i in range(len(num)):
            volume.append(np_count(out_data[np.nonzero(out_data)], num[i]))
        counts += 1
    else:
        without_plaque_list.append(plaque.split('/')[6]+'/'+plaque.split('/')[7]+'/'+plaque.split('/')[8])
        continue
    print(num, volume)
    after_num, after_volume, after_data = remove_mix_region(num, volume, out_data)
    all_plaque_volumes.extend(after_volume)
    #print(all_plaque_volumes)
    #print(after_num, after_volume)
    #print(after_data.shape)
    #save_nii(plaque, after_data)
    dcm_folder = plaque.split('_CTA')[0]
    series = SERIES(series_path=dcm_folder, strict_check_series=True)
    img_tensor_int16 = series.get_image_tensor_int16()
    #print(img_tensor_int16)
    brightness = get_brightness(after_num, after_data, img_tensor_int16)
    print(num, brightness)
    all_plaque_brightness.extend(brightness)
    #print(all_plaque_brightness)

volume_csv = pd.DataFrame(data=all_plaque_volumes)
volume_csv.to_csv('/mnt/users/code/test/volumes.csv', encoding='utf-8')
brightness_csv = pd.DataFrame(data=all_plaque_brightness)
brightness_csv.to_csv('/mnt/users/code/test/brightness.csv', encoding='utf-8')
plaque_volumes_mean = np.mean(all_plaque_volumes)
plaque_brightness_mean = np.mean(all_plaque_brightness)
plaque_volumes_std = np.std(all_plaque_volumes, ddof=1)
plaque_brightness_std = np.std(all_plaque_brightness, ddof=1)
print(plaque_volumes_mean, plaque_volumes_std, plaque_brightness_mean, plaque_brightness_std)

count = 0
for plaque in plaque_list:
    #print(plaque.split('/')[6])
    if int(plaque.split('/')[6]) < 1020000:
        continue
    if count == 166:
        break
    print(plaque)
    data = sitk.ReadImage(plaque)
    data = sitk.GetArrayFromImage(data)
    #print(data)
    #num, out_data = connected_domain(data)
    num, out_data = connected_component(data)
    if len(num) != 0:
        volume = []
        for i in range(len(num)):
            volume.append(np_count(out_data[np.nonzero(out_data)], num[i]))
        count += 1
    else:
        continue
    print(num, volume)
    after_num, after_volume, after_data = remove_mix_region(num, volume, out_data)
    all_data = after_data
    #save_nii(plaque, after_data)
    dcm_folder = plaque.split('_CTA')[0]
    series = SERIES(series_path=dcm_folder, strict_check_series=True)
    img_tensor_int16 = series.get_image_tensor_int16()
    #print(img_tensor_int16)
    brightness = get_brightness(after_num, after_data, img_tensor_int16)
    print(num, brightness)
    dealt_data, dealt_num = remove_mix_plaque(after_data, after_num, after_volume, brightness, plaque_volumes_mean+plaque_volumes_std, plaque_brightness_mean+plaque_brightness_std)
    erosion_data = plaque_erode(dealt_data)
    print(erosion_data[np.nonzero(erosion_data)])
    generated_data = gengerate_plaque(all_data, dealt_num, erosion_data)
    #print(generated_data[np.nonzero(generated_data)])
    save_nii(plaque, generated_data)
