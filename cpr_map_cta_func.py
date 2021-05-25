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
import time
from scipy import ndimage
import torch
from scipy.interpolate import interpn

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

    labeled_img, num = label(bw_img, connectivity=1, background=0, return_num=True)    
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
    #if not flip: 
    #    data = data[:,:,::-1]
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

def dedupe(items,key=None):
    seen = set()
    for item in items:
        #不可哈希值转可哈希值
        val = item if key is None else key(item)
        if val not in seen:
            yield item
            seen.add(val)

def compute_dice(pred, gt): 
    pred[pred>0] = 1
    gt[gt>0] = 1
    pred, gt = pred.flatten(), gt.flatten()
    num = np.sum(pred * gt)
    den1 = np.sum(pred)
    den2 = np.sum(gt)
    epsilon = 1e-4
    dice = (2 * num + epsilon) / (den1 + den2 + epsilon)
    return dice

def cta2cpr(cpr_coords, plaque_cta, cpr_map_index, patient, series, mask_names, cpr_counts):
    print('plaque')
    print(plaque_cta.shape)
    plaque_cta = (plaque_cta>=0.5).astype(np.float32)
    print(np.unique(plaque_cta))
    #cta_w, cta_h, cta_d = plaque_cta.shape
    plaque_cta = plaque_cta.reshape(1, 1, plaque_cta.shape[0], plaque_cta.shape[1], plaque_cta.shape[2])
    plaque_cta = torch.as_tensor(plaque_cta).cuda()

    for p in range(cpr_counts*20, cpr_counts*20+20):
        cpr_coord = cpr_coords[p]
        print('cpr_coord', cpr_coord.shape)
        '''
        cpr_h, cpr_w = cpr_coord.shape[:2]
        cpr_data = np.ones([cpr_h, cpr_w]) 
        for i in range(cpr_h):
            for j in range(cpr_w):
                x, y, z = cpr_coord[i, j]
                x, y, z = int(round(x)), int(round(y)), int(round(z))
                if 0 <= x < cta_w and 0 <= y < cta_h and 0 <= z < cta_d:
                    cpr_data[i, j] = plaque_cta[x, y, z]
        '''
        tmp = cpr_coord.reshape([-1, 3])
        print('cprminmax',tmp.max(axis=0),tmp.min(axis=0))
        cpr_coord = (cpr_coord/np.asarray(plaque_cta.shape[2:])[None,None])*2.0 - 1.0
        print(cpr_coord.max(),cpr_coord.min())
        cpr_coord = cpr_coord.reshape(1, 1, cpr_coord.shape[0], cpr_coord.shape[1], cpr_coord.shape[2]).astype(np.float32)
        cpr_coord = torch.as_tensor(cpr_coord).cuda()
        cpr_data = torch.nn.functional.grid_sample(plaque_cta, cpr_coord, mode='bilinear', padding_mode='border')
        cpr_data = torch.squeeze(cpr_data)
        cpr_data = torch.squeeze(cpr_data)
        cpr_data = torch.squeeze(cpr_data)
        print(cpr_data.shape)
        cpr_data = cpr_data.cpu().numpy()
        print(np.unique(cpr_data))
        cpr_data[cpr_data>0.5] = 255


        '''
        xs, ys, zs = cpr_coord[:, :, 0], cpr_coord[:, :, 1], cpr_coord[:, :, 2]
        cpr_data = ndimage.map_coordinates(plaque_cta, [xs, ys, zs], order=1, cval=0).astype(plaque_cta.dtype)
        '''
        '''
        cpr_h, cpr_w = cpr_coord.shape[:2]
        cpr_list = []
        for i in range(cpr_h):
            for j in range(cpr_w):
                x, y, z = cpr_coord[i, j]
                #x, y, z = int(round(x)), int(round(y)), int(round(z))
                cpr_list.append([x, y, z])

        cpr_list = np.array(cpr_list)        
        xyz_min = np.min(cpr_list, axis=0)
        xyz_max = np.max(cpr_list, axis=0)  
        print(xyz_min, xyz_max)
        values = []
        points = []
        for i in range(max(0, int(xyz_min[0])), min(int(xyz_max[0])+1, plaque_cta.shape[0])):
            for j in range(max(0, int(xyz_min[1])), min(int(xyz_max[1])+1, plaque_cta.shape[1])):
                for k in range(max(0, int(xyz_min[2])), min(int(xyz_max[2])+1, plaque_cta.shape[2])):
                    points.append([i,j,k])
                    values.append([plaque_cta[i,j,k]])
        print(len(points))
        
        points = np.array(points)
        grid_x, grid_y, grid_z = cpr_list[:,0], cpr_list[:,1], cpr_list[:,2]
        grid = GridData3D(points[:,0], points[:,1], points[:,2], values)
        cpr_data = grid(grid_x, grid_y, grid_z)
        '''


        cpr_data[cpr_data>=0.5] = 255
        cpr_data[cpr_data<0.5] = 0
        #print(cpr_data)
        if not os.path.exists(os.path.join(plaque_newcpr_path, patient, series)):
       	    os.makedirs(os.path.join(plaque_newcpr_path, patient, series))
        #print(mask_names[p])
        cv2.imwrite(os.path.join(plaque_newcpr_path, patient, series, mask_names[p]+'.png'), cpr_data)

        
        origin_cpr_data = cv2.imread(os.path.join(plaque_mask_gen_path, patient, series, mask_names[p]+'.png'), 0)
        dice = compute_dice(cpr_data, origin_cpr_data)
        print('dice', dice)
        
def cpr2cta_map(cpr_coords, cpr_mask_data, cta_shape, cpr_map_index, patient, series, mask_names):
    cta_data = np.zeros(cta_shape[::-1], dtype='uint8')
    x_cta, y_cta, z_cta = cta_shape[::-1]
    counts = 0 #用来计数cpr角度
    all_cta_coord = [] #用来存cta_coord
    coord_dic = {} #用来存所有点值
    coord_point = {} #用来存所有点坐标
    cpr_counts = 0
    time1 = time.time()
    for cpr_coord, cpr_mask in zip([cpr_coords[ii] for ii in cpr_map_index], [cpr_mask_data[jj] for jj in cpr_map_index]):
        print('cpr_coord', cpr_coord.shape, cpr_mask.shape)
        cpr_h, cpr_w = cpr_coord.shape[:2]
        #if not cpr_mask.any() > 0:
        #    continue
        area_index = ConnectComponent(cpr_mask)
        #print(area_index)
        if counts == 0:
            for r in range(len(area_index)):
                #print(len(area_index[r]))
                xy_max = np.max(area_index[r], axis=0)
                xy_min = np.min(area_index[r], axis=0)
                #print((xy_min[0],xy_max[0]),(xy_min[1],xy_max[1]))
                cta_coord = []
                for i in range(max(0,xy_min[0]-1), min(xy_max[0]+1,cpr_h)):
                    for j in range(max(0,xy_min[1]-1), min(xy_max[1]+1,cpr_w)):
                        x, y, z = cpr_coord[i, j]
                        cta_coord.append([x, y, z])
                        coord_dic[x,y,z] = cpr_mask[i, j]/255.0
                        if (int(round(x)),int(round(y)),int(round(z))) not in coord_point:
                            coord_point[int(round(x)),int(round(y)),int(round(z))] = [[x, y, z]]
                        else:
                            coord_point[int(round(x)),int(round(y)),int(round(z))].append([x, y, z])
                all_cta_coord.append(cta_coord)

        if counts != 0:
            for r in range(len(area_index)):
                xy_max = np.max(area_index[r], axis=0)
                xy_min = np.min(area_index[r], axis=0)
                #print((xy_min[0],xy_max[0]),(xy_min[1],xy_max[1]))
                cta_coord = []
                for i in range(max(0,xy_min[0]-1), min(xy_max[0]+1,cpr_h)):
                    for j in range(max(0,xy_min[1]-1), min(xy_max[1]+1,cpr_w)):
                        x, y, z = cpr_coord[i, j]
                        cta_coord.append([x, y, z])
                        coord_dic[x,y,z] = cpr_mask[i, j]/255.0
                        if (int(round(x)),int(round(y)),int(round(z))) not in coord_point:
                            coord_point[int(round(x)),int(round(y)),int(round(z))] = [[x, y, z]]
                        else:
                            coord_point[int(round(x)),int(round(y)),int(round(z))].append([x, y, z])
                
                #为了区分同一个cpr上的不同斑块
                for d in range(len(all_cta_coord)):
                    distances = pairwise_distances(np.array(cta_coord), device=torch.device("cuda"), y=np.array(all_cta_coord[d]))
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
            cta_cpr_data = np.zeros(cta_shape[::-1], dtype='uint8')
            print('all_cta_coord', len(all_cta_coord))
            for co_index in range(len(all_cta_coord)):
                #if co_index == 0:
                #    continue
                cta_coord = all_cta_coord[co_index]
                print(len(cta_coord))
                #cta_coord = list(dedupe(cta_coord, key=lambda x:tuple(x)))

                target_point = {} 
                for i in range(len(cta_coord)):
                    if (int(round(cta_coord[i][0])), int(round(cta_coord[i][1])), int(round(cta_coord[i][2]))) not in target_point:
                        if (int(round(cta_coord[i][0])), int(round(cta_coord[i][1])), int(round(cta_coord[i][2]))) in coord_point:
                            target_point[int(round(cta_coord[i][0])), int(round(cta_coord[i][1])), int(round(cta_coord[i][2]))] = len(coord_point[int(round(cta_coord[i][0])), int(round(cta_coord[i][1])), int(round(cta_coord[i][2]))])
 
                target_point = sorted(target_point.items(),key = lambda x: x[1],reverse = True)
                print(len(target_point))
                
                for target_k in target_point:
                    target_k = target_k[0]
                    print(target_k)
                    print('coord_point', len(coord_point[target_k]))
                    if not (target_k[0] >=0 and target_k[0] <= x_cta-1 and target_k[1] >=0 and target_k[1] <= y_cta-1 and target_k[2] >=0 and target_k[2] <= z_cta-1):
                        continue
                    if len(coord_point[target_k]) >= 8:
                        distances = pairwise_distances(target_k, device=torch.device("cuda"), y=coord_point[target_k])
                        #print(distances)
                        dist = getListMaxNumIndex(list(distances[0]), topk=8)
                        #print(dist)
                        dist_sum = 0
                        for ix in dist:
                            dist_sum += coord_dic[coord_point[target_k][ix][0], coord_point[target_k][ix][1], coord_point[target_k][ix][2]]
                        values = dist_sum/8
                        coord_dic[target_k] = values
                        print('coordvalues', values)
                    else:
                        target_expand_x, target_expand_y, target_expand_z = target_k[0], target_k[1], target_k[2]
                        
                        target_expand_list = []
                        for x in range(max(0,target_expand_x-1), min(target_expand_x+1, cta_shape[::-1][0])):
                            for y in range(max(0,target_expand_y-1), min(target_expand_y+1,cta_shape[::-1][1])):
                                for z in range(max(0,target_expand_z-1), min(target_expand_z+1,cta_shape[::-1][2])):
                                    if (x, y, z) in coord_point:
                                        target_expand_list.extend(coord_point[x,y,z])

                        print('<8', len(target_expand_list))
                        #print(target_expand_list)
                        distances = pairwise_distances(target_k, device=torch.device("cuda"), y=target_expand_list)
                        #print(distances)
                        dist = getListMaxNumIndex(list(distances[0]), topk=8)
                        #print(dist)
                        dist_sum = 0
                        for ix in dist:
                            #print(target_expand_list[ix])
                            dist_sum += coord_dic[target_expand_list[ix][0], target_expand_list[ix][1], target_expand_list[ix][2]]
                        values = dist_sum/8
                        coord_dic[target_k] = values
                        print('<8values', values)

                    cta_data[target_k] = values
                    cta_cpr_data[target_k] = values
            
            cta2cpr(cpr_coords, cta_cpr_data, cpr_map_index, patient, series, mask_names, cpr_counts)
            cpr_counts += 1
            all_cta_coord = [] #用来存cta_coord
            coord_dic = {} #用来存所有点值
            coord_point = {} #用来存所有点坐标
            counts = 0

    time2 = time.time()
    print(time2-time1)
    return cta_data      

plaque_newcpr_path = '/mnt/users/ffr_datasets/ffr_newcpr/'
plaque_mask_newmap_path = '/mnt/users/ffr_datasets/ffr_cpr_mask_newmap/' #'/mnt/users/ffr_10datasets/ffr_cpr_mask_newmap/'
plaque_mask_gen_path = '/mnt/users/ffr_datasets/ffr_cpr_mask_dealt/' #'/mnt/users/ffr_10datasets/ffr_cpr_mask_dealt/'
cta_data_gen_path = '/mnt/DrwiseDataNFS/drwise_runtime_env/data1/inputdata' #'/mnt/users/ffr_10datasets/ffr_plaque/'
mapping_gen_path = '/mnt/DrwiseDataNFS/drwise_runtime_env/data1/inputdata' #'/mnt/users/ffr_10datasets/ffr_cpr_dat/'
#dat_path = '/data1/zhangyongming/ffr_10datasets/ffr_cpr_dat/1036602/10791347/EE599B93_CTA/cprCoronary/'
#cpr_mask_dir = '/data1/zhangyongming/ffr_10datasets/ffr_cpr_mask_dealt/1036602/EE599B93_CTA/'
#cta_path = '/data1/zhangyongming/ffr_10datasets/ffr_plaque/1036602/10791347/EE599B93_CTA/'

for patient in sorted(os.listdir(plaque_mask_gen_path)):
    if patient != '1036602': #1036602
        continue
    print(patient)
    for series in os.listdir(os.path.join(plaque_mask_gen_path, patient)):
        #if series != '5F654861_CTA':
        #    continue
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

        cta_path = glob.glob(os.path.join(cta_data_gen_path, patient, '*', series.split('_')[0]))[0]
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
        cta_data = cpr2cta_map(mapping_files, mask_files, cta_shape, roi_index_list, patient, series, mask_names)

        if not os.path.exists(os.path.join(plaque_mask_newmap_path, patient, series)):
            os.makedirs(os.path.join(plaque_mask_newmap_path, patient, series))
        cta_series = SERIES(series_path=cta_path, strict_check_series=True)
        save_nii(os.path.join(plaque_mask_newmap_path, patient, series), cta_series.flip, cta_data, 'mask_plaque_func.nii.gz')
