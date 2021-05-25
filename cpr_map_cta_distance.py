#-*-coding:utf-8-*-
import os
import struct
import numpy as np
import shutil
import glob
import cv2
from PIL import Image
import math
from scipy.interpolate import griddata
import nibabel as nib
from mitok.utils.mdicom import SERIES
import torch
from skimage.measure import label
import copy

def ConnectComponent(bw_img):
    '''
    compute largest Connect component of a binary image
    
    Parameters:
    ---

    bw_img: ndarray
        binary image
	
	Returns:
	---

	lcc: ndarray
		largest connect component.

    Example:
    ---
        >>> lcc = largestConnectComponent(bw_img)

    '''

    labeled_img, num = label(bw_img, neighbors=4, background=0, return_num=True)    
    #print(labeled_img[np.nonzero(labeled_img)], num)
    '''
    max_label = 0
    max_num = 0
    for i in range(1, num+1): # 这里从1开始，防止将背景设置为最大连通域
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)
    '''
    all_index = []
    for i in range(1, num+1):
        index = np.argwhere(labeled_img==i)
        all_index.append(index)
        print(len(index))

    return all_index

def load_cpr_coord_map(file_path):
    f = open(file_path, mode='rb')
    a = f.read()
    w, h, c, t = struct.unpack('iiii', a[:16])
#    x, y, z, c = struct.unpack('iiii', a[:16])
#    print(x,y,z,c)
    assert c == 3 and t == 10, 'The third and fourth items of cpr coor map should be 3 and 10'
#    assert c == 2 and t == 10
    maps = struct.unpack('f' * w * h * c, a[16:])
    maps = np.float32(maps).reshape(h, w, c)

#   maps = struct.unpack('f' * x * y * z * c, a[20:])
#    maps = numpy.float32(maps).reshape(z, y, x, c)

  #  print(maps)
    return maps

def trilinear_interpolation(x_cta, y_cta, z_cta, coord_list, cpr_mask_values):
    print(cpr_mask_values)
    x, y, z = x_cta, y_cta, z_cta
    x0, y0, z0 = coord_list[0]
    x1, y1, z1 = coord_list[1]

    tx = (x - x0)/(x1 - x0)
    ty = (y - y0)/(y1 - y0)
    tz = (z - z0)/(z1 - z0)

    c = np.arange(8).reshape(2,2,2)
    cn = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                c[i][j][k] = cpr_mask_values[cn]
                cn += 1

    accum = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                accum += (i*tx+(1-i)*(1-tx))*(j*ty+(1-j)*(1-ty))*(k*tz+(1-k)*(1-tz))*c[i][j][k]
    if accum != 0:
        print(accum)
    return accum

def save_nii(plaque_dir, flip, data, nii_name):
    data[data>=0.5] = 1
    data[data<0.5] = 0
    if not flip: 
        data = np.transpose(data, (2, 1, 0))
        data = data.astype('float32')
    else:
        data = data.astype('float32')
    affine_arr = np.eye(4)
    plaque_nii = nib.Nifti1Image(data, affine_arr)
    nib.save(plaque_nii, os.path.join(plaque_dir, nii_name))  

def pairwise_distances(x, device, y=None, to_numpy=True, do_sqrt=True):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    if do_sqrt is True, return dist**0.5
    '''
    if not isinstance(x, torch.Tensor):
        x = np.asarray(x)
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        x = torch.as_tensor(x).to(device)
    x = x.to(torch.float32)
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        if not isinstance(y, torch.Tensor):
            y = np.asarray(y)
            if len(y.shape) == 1:
                y = y[np.newaxis, :]
            y = torch.as_tensor(y).to(device)
        y = y.to(torch.float32)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    dist = torch.clamp(dist, 0.0, np.inf)
    if do_sqrt:
        dist = dist**0.5
    if to_numpy:
        dist = dist.cpu().numpy()
    return dist

def getListMaxNumIndex(num_list,topk=3):
    '''
    获取列表中最大的前n个数值的位置索引
    '''
    tmp_list=copy.deepcopy(num_list)
    max_num=sum([abs(O) for O in num_list])
    min_num=-1*max_num
    max_num_index,min_num_index=[],[]
    for i in range(topk):
        one_max_index=num_list.index(max(num_list))
        max_num_index.append(one_max_index)
        num_list[one_max_index]=min_num
    for i in range(topk):
        one_min_index=tmp_list.index(min(tmp_list))
        min_num_index.append(one_min_index)
        tmp_list[one_min_index]=max_num
    #print ('max_num_index:',max_num_index)
    #print ('min_num_index:',min_num_index)

    return min_num_index

def cpr2cta_map(cpr_coords, cpr_mask_data, cta_shape, cpr_map_index):
    cta_data = np.zeros(cta_shape[::-1], dtype='uint8')
    counts = 0 #用来计数cpr角度
    all_cta_coord = [] #用来存cta_coord
    coord_dic = {} #用来存所有点值
    for cpr_coord, cpr_mask in zip([cpr_coords[i] for i in cpr_map_index], [cpr_mask_data[j] for j in cpr_map_index]):
        print('cpr_coord', cpr_coord.shape, cpr_mask.shape)
        if not cpr_mask.any() > 0:
            continue
        area_index = ConnectComponent(cpr_mask)
        #print(area_index)
        
        if counts == 0:
            for r in range(len(area_index)):
                #print(len(area_index[r]))
                cta_coord = [] #用来存cta_coord
                for i in area_index[r]:
                    if [cpr_coord[i[0], i[1]][0], cpr_coord[i[0], i[1]][1], cpr_coord[i[0], i[1]][2]] not in cta_coord:
                        cta_coord.append([cpr_coord[i[0], i[1]][0], cpr_coord[i[0], i[1]][1], cpr_coord[i[0], i[1]][2]])
                        coord_dic[cpr_coord[i[0], i[1]][0], cpr_coord[i[0], i[1]][1], cpr_coord[i[0], i[1]][2]] = 1
                all_cta_coord.append(cta_coord)

        if counts != 0:
            for r in range(len(area_index)):
                cta_coord = [] #用来存cta_coord
                for i in area_index[r]:
                    #cta_coord.append(cpr_coord[i[0], i[1]])
                    if [cpr_coord[i[0], i[1]][0], cpr_coord[i[0], i[1]][1], cpr_coord[i[0], i[1]][2]] not in cta_coord:
                        cta_coord.append([cpr_coord[i[0], i[1]][0], cpr_coord[i[0], i[1]][1], cpr_coord[i[0], i[1]][2]])
                        coord_dic[cpr_coord[i[0], i[1]][0], cpr_coord[i[0], i[1]][1], cpr_coord[i[0], i[1]][2]] = 1
                #为了区分同一个cpr上的不同斑块
                for d in range(len(all_cta_coord)):
                    #print(d)
                    distances = pairwise_distances(np.array(cta_coord), device=torch.device("cuda"), y=np.array(all_cta_coord[d]))
                    #print(np.unique(distances))
                    if (distances<3**0.5).any():
                        all_cta_coord[d].extend(cta_coord)
                        flag = 0
                        break
                    else:
                        flag = 1
                #添加新增的连通域
                if flag and (d == len(all_cta_coord)-1):  
                    all_cta_coord.append(cta_coord)

        counts += 1
        if counts == 20:
            print(len(all_cta_coord))
            for co_index in range(len(all_cta_coord)):
                #if co_index == 0:
                #    continue
                cta_coord = all_cta_coord[co_index]
                print(len(cta_coord))
                xmin_3d, xmax_3d = np.min(np.array(cta_coord)[:, 0]), np.max(np.array(cta_coord)[:, 0])
                ymin_3d, ymax_3d = np.min(np.array(cta_coord)[:, 1]), np.max(np.array(cta_coord)[:, 1])
                zmin_3d, zmax_3d = np.min(np.array(cta_coord)[:, 2]), np.max(np.array(cta_coord)[:, 2])
                print(xmin_3d, xmax_3d, ymin_3d, ymax_3d, zmin_3d, zmax_3d)
                
                cta_coord_data = np.zeros(cta_shape[::-1], dtype='uint8')
                cta_coord_data[int(xmin_3d)-1:int(xmax_3d)+2, int(ymin_3d)-1:int(ymax_3d)+2, int(zmin_3d)-1:int(zmax_3d)+2] = 1
                cta_coord_data[int(xmin_3d):int(xmax_3d)+1, int(ymin_3d):int(ymax_3d)+1, int(zmin_3d):int(zmax_3d)+1] = 10
                cta_coord_newadd = np.argwhere(cta_coord_data==1)
                print(cta_coord_newadd.shape)
                for i in cta_coord_newadd:
                    #print(type(i),i)
                    if [i[0], i[1], i[2]] not in cta_coord:
                        cta_coord.append([i[0],i[1],i[2]])
                        coord_dic[i[0],i[1],i[2]] = 0
                
                xmin_3d, xmax_3d = np.min(np.array(cta_coord)[:, 0]), np.max(np.array(cta_coord)[:, 0])
                ymin_3d, ymax_3d = np.min(np.array(cta_coord)[:, 1]), np.max(np.array(cta_coord)[:, 1])
                zmin_3d, zmax_3d = np.min(np.array(cta_coord)[:, 2]), np.max(np.array(cta_coord)[:, 2])
                print(xmin_3d, xmax_3d, ymin_3d, ymax_3d, zmin_3d, zmax_3d)

                target_point = []
                for x in range(int(xmin_3d)-1, int(xmax_3d)+2):
                    for y in range(int(ymin_3d)-1, int(ymax_3d)+2):
                        for z in range(int(zmin_3d)-1, int(zmax_3d)+2):
                            target_point.append([x, y, z])
            
                #target_point = np.array(target_point)
                print(np.array(target_point).shape)

                cta_coord = np.array(cta_coord).reshape(-1,3)
                print(cta_coord.shape)
                distances = pairwise_distances(np.array(target_point), device=torch.device("cuda"), y=cta_coord)
                #求所有整数点分别到其他点的距离，并求距离总和，进行排序，找距离和最小的整数已知点和其最近的8个点计算三线性插值
                for k in range(len(target_point)):
                    print(np.array(target_point).shape)
                    if k != 0:
                        #print(target_point[k:])
                        newadd = np.sqrt(np.sum((target_point-np.array(cta_coord[-1]))**2, axis=1))
                        print(distances[k].shape)
                        distances = np.insert(distances, len(distances[k-1]), values=newadd, axis=1)
                    print(distances[k].shape)
                    print(np.array(cta_coord).shape)
                    print(target_point[k])
                    dis_list = list(distances[k])
                    #print(dis_list)
                    dist = getListMaxNumIndex(dis_list,topk=8)
                    print(dist)
                    dist_sum = 0
                    for ix in dist:
                        dist_sum += distances[k][ix]
                    values = 0
                    for i in dist:
                        print('cta_coord', cta_coord[i])
                        print(coord_dic[cta_coord[i][0],cta_coord[i][1],cta_coord[i][2]])
                        cpr_mask_value = coord_dic[cta_coord[i][0],cta_coord[i][1],cta_coord[i][2]]
                        values += distances[k][i]/dist_sum * cpr_mask_value
                    print('values', values)
                    cta_data[target_point[k]] = values
                    cta_coord = list(cta_coord)
                    cta_coord.append(target_point[k])
                    coord_dic[target_point[k][0], target_point[k][1], target_point[k][2]] = values

            all_cta_coord = [] #用来存cta_coord
            coord_dic = {} #用来存所有点值
            counts = 0

    return cta_data      

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
plaque_mask_newmap_path = '/mnt/users/ffr_10datasets/ffr_cpr_mask_newmap/'
plaque_mask_gen_path = '/mnt/users/ffr_10datasets/ffr_cpr_mask_dealt/'
cta_data_gen_path = '/mnt/users/ffr_10datasets/ffr_plaque/'
mapping_gen_path = '/mnt/users/ffr_10datasets/ffr_cpr_dat/'
#dat_path = '/data1/zhangyongming/ffr_10datasets/ffr_cpr_dat/1036602/10791347/EE599B93_CTA/cprCoronary/'
#cpr_mask_dir = '/data1/zhangyongming/ffr_10datasets/ffr_cpr_mask_dealt/1036602/EE599B93_CTA/'
#cta_path = '/data1/zhangyongming/ffr_10datasets/ffr_plaque/1036602/10791347/EE599B93_CTA/'

for patient in sorted(os.listdir(plaque_mask_gen_path)):
    if patient != '1036602':
        continue
    print(patient)
    for series in os.listdir(os.path.join(plaque_mask_gen_path, patient)):
        print(series)
        mask_paths = sorted(glob.glob(os.path.join(plaque_mask_gen_path, patient, series, '*.png')))
        mask_names = [mask_path.split('/')[-1].split('.')[0] for mask_path in mask_paths]
        mask_files = [np.array(Image.open(mask_path)) for mask_path in mask_paths]

        cpr_paths = glob.glob(os.path.join(mapping_gen_path, patient, '*', series, 'cprCoronary'))[0]
        cpr_paths = sorted(glob.glob(os.path.join(cpr_paths, '*.dcm')))
        cpr_names = [cpr_path.split('/')[-1].split('.')[0] for cpr_path in cpr_paths]

        vessel_mask_paths = glob.glob(os.path.join(mapping_gen_path, patient, '*', series, 'cprCoronary'))[0]
        vessel_mask_paths = sorted(glob.glob(os.path.join(vessel_mask_paths, '*.bmp')))
        vessel_mask_names = [vessel_mask_path.split('/')[-1].split('.')[0] for vessel_mask_path in vessel_mask_paths]
        vessel_mask_files = [np.array(Image.open(vessel_mask_path)) for vessel_mask_path in vessel_mask_paths]       

        mapping_paths = glob.glob(os.path.join(mapping_gen_path, patient, '*', series, 'cprCoronary'))[0]
        mapping_paths = sorted(glob.glob(os.path.join(mapping_paths, '*.dat')))
        mapping_names = [mapping_path.split('/')[-1].split('.')[0] for mapping_path in mapping_paths]
        mapping_files = [load_cpr_coord_map(mapping_path) for mapping_path in mapping_paths]

        cta_path = glob.glob(os.path.join(cta_data_gen_path, patient, '*', series))[0]
        cta_paths = glob.glob(os.path.join(cta_path, '*.dcm'))
        cta_paths.sort(key=lambda ele: int(ele.split('/')[-1].split('.')[0]))
        #cta_paths = os.listdir(cta_path) 
        #print(len(cta_paths))
        cta_shape = [len(cta_paths), 512, 512]                                               
        
        roi_index_list = []
        for i in range(len(mask_names)):
            if mask_names[i] in mapping_names:        
                roi_index = mapping_names.index(mask_names[i])
                roi_index_list.append(roi_index)
        #print(mask_names)
        #print(mapping_names)
        cta_data = cpr2cta_map(mapping_files, mask_files, cta_shape, roi_index_list)

        if not os.path.exists(os.path.join(plaque_mask_newmap_path, patient, series)):
            os.makedirs(os.path.join(plaque_mask_newmap_path, patient, series))
        cta_series = SERIES(series_path=cta_path, strict_check_series=True)
        save_nii(os.path.join(plaque_mask_newmap_path, patient, series), cta_series.flip, cta_data, 'mask_plaque_newmap_dist30.5.nii.gz')
