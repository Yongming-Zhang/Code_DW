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
from scipy.interpolate import bisplrep, bisplev
from mitok.image.cv_gpu import fill_holes
import json

class GridData3D():
    """ Interpolation of data on a three-dimensional irregular grid.
    
        Essentially, a wrapper function around scipy's interpolate.griddata.

        https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.interpolate.griddata.html

        An alternative to griddata could be Rbf, as discussed here:

        https://stackoverflow.com/questions/37872171/how-can-i-perform-two-dimensional-interpolation-using-scipy

        Attributes: 
            u: 1d numpy array
                data points 1st coordinate
            v: 1d numpy array
                data points 2nd coordinate
            w: 1d numpy array
                data points 3rd coordinate
            r: 1d numpy array
                data values
            method : {‘linear’, ‘nearest’}, optional
    """
    def __init__(self, u, v, w, r, method='linear'):
        self.uvw = np.array([u,v,w]).T
        self.r = r
        self.method = method


    def __call__(self, theta, phi, z, grid=False):
        """ Interpolate data

            theta, phi, z can be floats or arrays.

            If grid is set to False, the interpolation will be evaluated at 
            the coordinates (theta_i, phi_i, z_i), where theta=(theta_1,...,theta_N), 
            phi=(phi_1,...,phi_N) and z=(z_,...,z_K). Note that in this case, theta, phi, 
            z must all have the same length.

            If grid is set to True, the interpolation will be evaluated at 
            all combinations (theta_i, theta_j, z_k), where theta=(theta_1,...,theta_N), 
            phi=(phi_1,...,phi_M) and z=(z_1,...,z_K). Note that in this case, the lengths 
            of theta, phi, z do not have to be the same.

            Args: 
                theta: float or array
                   1st coordinate of the points where the interpolation is to be evaluated
                phi: float or array
                   2nd coordinate of the points where the interpolation is to be evaluated
                z: float or array
                   3rd coordinate of the points where the interpolation is to be evaluated
                grid: bool
                   Specify how to combine elements of theta and phi.

            Returns:
                ri: Interpolated values
        """        
        if grid: theta, phi, z, M, N, K = self._meshgrid(theta, phi, z)

        pts = np.array([theta,phi,z]).T
        ri = griddata(self.uvw, self.r, pts, method=self.method, fill_value=0)

        if grid: ri = np.reshape(ri, newshape=(M,N,K))

        return ri

    def _meshgrid(self, theta, phi, z, grid=False):
        """ Create grid

            Args: 
                theta: 1d numpy array
                   1st coordinate of the points where the interpolation is to be evaluated
                phi: 1d numpy array
                   2nd coordinate of the points where the interpolation is to be evaluated
                z: float or array
                   3rd coordinate of the points where the interpolation is to be evaluated

            Returns:
                theta, phi, z: 3d numpy array
                    Grid coordinates
                M, N, K: int
                    Number of grid values
        """        
        M = 1
        N = 1
        K = 1
        if np.ndim(theta) == 1: M = len(theta)
        if np.ndim(phi) == 1: N = len(phi)
        if np.ndim(z) == 1: K = len(z)
        theta, phi, z = np.meshgrid(theta, phi, z)
        theta = theta.flatten()
        phi = phi.flatten()
        z = z.flatten()        
        return theta,phi,z,M,N,K

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
        print('index', len(index))

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
    print(np.unique(data))
    #data[data>=0.5] = 1
    #data[data<0.5] = 0
    data = (data>=128).astype(np.float32)
    #if not flip: 
    #    data = data[:,:,::-1]
    print(np.unique(data))
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
    index = np.argwhere(plaque_cta==1)
    print(index)
    print(plaque_cta[0,213,56])
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
        print('cprmaxmin',tmp.max(axis=0),tmp.min(axis=0))
        cpr_coord = (cpr_coord/np.asarray(plaque_cta.shape[2:])[None,None])*2.0 - 1.0
        print(cpr_coord.max(),cpr_coord.min())
        cpr_coord = cpr_coord.reshape(1, 1, cpr_coord.shape[0], cpr_coord.shape[1], cpr_coord.shape[2]).astype(np.float32)
        cpr_coord = torch.as_tensor(cpr_coord).cuda()
        print(cpr_coord)
        print(torch.unique(plaque_cta))
        cpr_data = torch.nn.functional.grid_sample(plaque_cta, cpr_coord, mode='bilinear', padding_mode='border', align_corners=True)
        cpr_data = torch.squeeze(cpr_data)
        cpr_data = torch.squeeze(cpr_data)
        cpr_data = torch.squeeze(cpr_data)
        print(cpr_data.shape)
        cpr_data = cpr_data.cpu().numpy()
        print(cpr_data)
        print(np.unique(cpr_data))
        cpr_data[cpr_data>=128] = 255
        cpr_data[cpr_data<128] = 0


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
        print('dice',dice)
        

def cpr2cta_map(cpr_coords, cpr_mask_data, cta_shape, cpr_map_index, patient, series, mask_names, contour_files, cpr_centerlines):
    time1 = time.time()
    cta_data = np.zeros(cta_shape[::-1], dtype='uint8')
    print(len(cpr_centerlines))
    cpr_coords = [cpr_coords[ii] for ii in cpr_map_index]
    cpr_masks = [cpr_mask_data[jj] for jj in cpr_map_index]
    contour_files = [contour_files[kk] for kk in cpr_map_index]
    cpr_counts = 0
    all_cpr_coord = [] #用来存cpr坐标
    all_cpr_contours = [] #用来存轮廓坐标
    all_cta_coord = [] #用来存轴位坐标
    all_centerlines = [] #用来存中心线坐标
    all_ids = [] #  用来存cpr的id

    for cout in range(len(cpr_centerlines)):
        c = 0
        for i in range(cpr_counts*20, (cpr_counts+1)*20):
            cpr_mask = cpr_masks[i]
            cpr_contour = contour_files[i]
            cpr_axis_coord = cpr_coords[i]
            print('cpr_contour', cpr_contour.shape)
            print('cpr_mask', cpr_mask.shape)
            area_index = ConnectComponent(cpr_mask)

            cpr_centerline = cpr_centerlines[cpr_counts]['vessel_group']
            for cen in range(len(cpr_centerline)):
                if cpr_centerline[cen]['name'] == mask_names[i]:
                    cpr_centerline_points = cpr_centerline[cen]['points']
                    break
            print(cpr_centerline[cen]['name'])
            if i%20 == 0:
                for r in range(len(area_index)):
                    mask_index = area_index[r]
                    #print(len(area_index[r]))
                    c += len(area_index[r])
                    cpr_coord = []
                    cpr_maskcontour = []
                    cta_coord = []
                    for cpr in mask_index:
                        cpr_coord.append([cpr[0],cpr[1]])
                        x, y, z = cpr_axis_coord[cpr[0],cpr[1]]
                        cta_coord.append([x, y, z])
                        if cpr_contour[cpr[0],cpr[1]] == 255:
                            cpr_maskcontour.append([cpr[0],cpr[1]])
                    all_cpr_coord.append([cpr_coord])
                    all_cpr_contours.append([cpr_maskcontour])
                    all_cta_coord.append(cta_coord)
                    all_centerlines.append([cpr_centerline_points])
                    all_ids.append([i%20])

            if i%20 != 0:
                for r in range(len(area_index)):
                    mask_index = area_index[r]
                    #print(len(area_index[r]))
                    c += len(area_index[r])
                    cpr_coord = []
                    cpr_maskcontour = []
                    cta_coord = []
                    for cpr in mask_index:
                        cpr_coord.append([cpr[0],cpr[1]])
                        x, y, z = cpr_axis_coord[cpr[0],cpr[1]]
                        cta_coord.append([x, y, z])
                        if cpr_contour[cpr[0],cpr[1]] == 255:
                            cpr_maskcontour.append([cpr[0],cpr[1]])
                    
                    #为了区分同一个cpr上的不同斑块
                    for d in range(len(all_cta_coord)):
                        #print('cpr_coord',cpr_coord)
                        #print('all_cpr_coord_copy',all_cpr_coord_copy[d])
                        distances = pairwise_distances(np.array(cta_coord), device=torch.device("cuda"), y=np.array(all_cta_coord[d]))
                        if (distances<3**0.5).any():
                            all_cpr_coord[d].append(cpr_coord)
                            all_cpr_contours[d].append(cpr_maskcontour)
                            all_cta_coord[d].extend(cta_coord)
                            all_centerlines[d].append(cpr_centerline_points)
                            all_ids[d].append(i%20)
                            flag = 0
                            break
                        else:
                            flag = 1
                    #添加新增的连通域
                    if flag and (d == len(all_cta_coord)-1):  
                        all_cpr_coord.append([cpr_coord])
                        all_cpr_contours.append([cpr_maskcontour])
                        all_cta_coord.append(cta_coord)
                        all_centerlines.append([cpr_centerline_points])
                        all_ids.append([i%20])

        print('c1', c)
        c = 0
        print('all_cpr_coord', len(all_cpr_coord))
        for i in range(len(all_cpr_coord)):
            print('all_cpr_coord[i]', len(all_cpr_coord[i]))
            all_axis_coord_left = [] #存左边轮廓点
            all_axis_coord_right = [] #存右边轮廓点
            id_left = [] #存左边id
            id_right = [] #存右边id
            centerline_left = [] #存左边中心线
            centerline_right = [] #存右边中心线
            for j in range(len(all_cpr_coord[i])):
                cpr_index = all_cpr_coord[i][j]
                ids = all_ids[i][j]
                cpr_axis_coord = cpr_coords[cpr_counts*20+ids]
                cpr_contour_index = all_cpr_contours[i][j]
                print('cpr_index', len(cpr_index))
                c += len(cpr_index)
                cpr_index_array = np.asarray(cpr_index) 
                xy_max = np.max(cpr_index_array, axis=0)
                xy_min = np.min(cpr_index_array, axis=0)
                #print((xy_max[0], xy_max[1]))
                cpr_data = np.zeros((xy_max[0]+1, xy_max[1]+1))
                print(cpr_data.shape)
                for index in cpr_index:
                    cpr_data[index[0], index[1]] = 1
                cenlines = []
                #print(all_centerlines[i][j])
                all_axis_centerline = []
                for centerline in all_centerlines[i][j]: 
                    if xy_min[0] <= int(round(centerline[0])) <= xy_max[0] and xy_min[1] <= int(round(centerline[1])) <= xy_max[1]:
                        cpr_data[int(round(centerline[0])), int(round(centerline[1]))] = 0
                        cenlines.append(cpr_axis_coord[int(round(centerline[0])), int(round(centerline[1]))])
                all_axis_centerline.append(cenlines)
                print('all_axis_centerline',len(all_axis_centerline))
                area_cpr_data_index = ConnectComponent(cpr_data)

                #用来保存mask的轮廓点
                allcontours = []
                #用来保存中心线
                allcontours_centerline = []
                for k in range(len(area_cpr_data_index)):
                    area_cpr = area_cpr_data_index[k]
                    contours = []
                    for index in area_cpr:
                        if [index[0], index[1]] in cpr_contour_index:
                            x, y, z = cpr_axis_coord[index[0], index[1]]
                            contours.append([x, y, z])
                            #contours.append([index[0], index[1]])
                    allcontours.append(contours)
                    allcontours_centerline.append(all_axis_centerline)

                if j == 0 or all_axis_coord_left == []: 
                    print('len(allcontours)0', len(allcontours))
                    if len(allcontours) == 2:
                        print('中心线使该区域分为两个连通域1')
                        all_axis_coord_left.append(allcontours[0])
                        id_left.append(all_ids[i][j])
                        centerline_left.append(allcontours_centerline[0])
                        all_axis_coord_right.append(allcontours[1])
                        id_right.append(all_ids[i][j])
                        centerline_right.append(allcontours_centerline[1])

                    elif len(allcontours) == 1:
                        print('没有使该区域分为两个连通域1', len(allcontours))
                        all_axis_coord_left.append(allcontours[0])
                        id_left.append(all_ids[i][j])
                        centerline_left.append(allcontours_centerline[0])

                else:
                    print('len(allcontours)', len(allcontours))
                    if len(allcontours) == 2:
                        print('中心线使该区域分为两个连通域2')
                        for k in range(len(allcontours)):
                            distances_left = pairwise_distances(np.array(allcontours[k]), device=torch.device("cuda"), y=np.array(all_axis_coord_left[0]))
                            if len(all_axis_coord_right) != 0:
                                distances_right = pairwise_distances(np.array(allcontours[k]), device=torch.device("cuda"), y=np.array(all_axis_coord_right[0]))
                                if np.min(distances_left) < np.min(distances_right):
                                    all_axis_coord_left.append(allcontours[k])
                                    id_left.append(all_ids[i][j])
                                    centerline_left.append(allcontours_centerline[k])
                                else:
                                    all_axis_coord_right.append(allcontours[k])
                                    id_right.append(all_ids[i][j])
                                    centerline_right.append(allcontours_centerline[k])
                            else:
                                distances = pairwise_distances(np.array(allcontours[k]), device=torch.device("cuda"), y=np.array(all_axis_coord_left[0]))
                                if (distances<3**0.5).any():
                                    all_axis_coord_left.append(allcontours[k])
                                    id_left.append(all_ids[i][j])
                                    centerline_left.append(allcontours_centerline[k])
                                else:
                                    all_axis_coord_right.append(allcontours[k])
                                    id_right.append(all_ids[i][j])
                                    centerline_right.append(allcontours_centerline[k])
                    elif len(allcontours) == 1:
                        print('没有使该区域分为两个连通域2', len(allcontours))
                        distances = pairwise_distances(np.array(allcontours[0]), device=torch.device("cuda"), y=np.array(all_axis_coord_left[0]))
                        if (distances<3**0.5).any():
                            all_axis_coord_left.append(allcontours[0])
                            id_left.append(all_ids[i][j])
                            centerline_left.append(allcontours_centerline[0])
                        else:
                            all_axis_coord_right.append(allcontours[0])
                            id_right.append(all_ids[i][j])
                            centerline_right.append(allcontours_centerline[0])
            print('id_left',id_left)
            print('all_axis_coord_left', len(all_axis_coord_left))
            print('centerline_left', len(centerline_left))
            print('id_right',id_right)
            print('all_axis_coord_right', len(all_axis_coord_right))
            print('centerline_right', len(centerline_right))
            print(id_left)
            id_left_copy = list(set(id_left))
            left_counts = {}
            for le in id_left_copy:
                left_counts[le] = id_left.count(le)
            print(left_counts)

            left_i = 0
            while left_i < len(id_left):
                if left_i < len(id_left)-left_counts[id_left[-1]]:  
                    #判断如果有两个相同编号的斑块连通域，则判断这两个的下一个，若有下一个挨着的编号，则分别与下一个连接，
                    #若没有，则要判断该区域是否被中心线分割，若有，则与中心线连接，若没有，则不连接，保存这些点。
                    left_next = left_i + left_counts[id_left[left_i]]
                    print(left_i, left_next)
                    if id_left[left_i]+1 == id_left[left_next]:
                        left_next_next = left_next + left_counts[id_left[left_next]]
                        for left_in in range(left_i, left_next):
                            dist = []
                            for left_jn in range(left_next, left_next_next):
                                distances = pairwise_distances(np.array(all_axis_coord_left[left_in]), device=torch.device("cuda"), y=np.array(all_axis_coord_left[left_jn]))
                                dist.append(np.min(distances))
                            ind = np.argmin(dist, axis=0)
                            generate_point_lines(all_axis_coord_left[left_in], all_axis_coord_left[left_next+ind], cta_data)
                            
                        for left_in in range(left_next, left_next_next):
                            dist = []
                            for left_jn in range(left_i, left_next):
                                distances = pairwise_distances(np.array(all_axis_coord_left[left_in]), device=torch.device("cuda"), y=np.array(all_axis_coord_left[left_jn]))
                                dist.append(np.min(distances))
                            ind = np.argmin(dist, axis=0)
                            generate_point_lines(all_axis_coord_left[left_in], all_axis_coord_left[left_i+ind], cta_data)

                    elif id_left[left_i]+1 != id_left[left_next]:
                        while left_i < left_next:
                            if centerline_left[left_i][0] == []:
                                #将这些值赋值到cta_data
                                generate_data(all_axis_coord_left[left_i], cta_data)
                            elif centerline_left[left_i][0] != []:
                                print('centerline_left[left_i][0]', centerline_left[left_i][0])
                                generate_point_lines(all_axis_coord_left[left_i], centerline_left[left_i][0], cta_data)
                            left_i += 1
                    left_i = left_next
                else:
                    left_i += 1
       
            #判断当前所连接连通域的编号是否连续，若连续，并且最后的编号是19，开始编号是0，则19连接0
            if len(id_left) >= 20:
                left_list = list(set(id_left))
                if left_list == 20:
                    left_ind = len(id_left) - left_counts[id_left[-1]]
                    left_ind_next = len(id_left)
                    left_0 = 0
                    left_0_next = left_counts[id_left[0]]
                    for left_in in range(left_ind, left_ind_next):
                        dist = []
                        for left_jn in range(left_0, left_0_next):
                            distances = pairwise_distances(np.array(all_axis_coord_left[left_in]), device=torch.device("cuda"), y=np.array(all_axis_coord_left[left_jn]))
                            dist.append(np.min(distances))
                        ind = np.argmin(dist, axis=0)
                        generate_point_lines(all_axis_coord_left[left_in], all_axis_coord_left[left_0+ind], cta_data)
                            
                    for left_in in range(left_0, left_0_next):
                        dist = []
                        for left_jn in range(left_ind, left_ind_next):
                            distances = pairwise_distances(np.array(all_axis_coord_left[left_in]), device=torch.device("cuda"), y=np.array(all_axis_coord_left[left_jn]))
                            dist.append(np.min(distances))
                        ind = np.argmin(dist, axis=0)
                        generate_point_lines(all_axis_coord_left[left_in], all_axis_coord_left[left_ind+ind], cta_data)

            print(id_right)
            id_right_copy = list(set(id_right))
            right_counts = {}
            for le in id_right_copy:
                right_counts[le] = id_right.count(le)
            print(right_counts)

            right_i = 0
            while right_i < len(id_right):
                if right_i < len(id_right)-right_counts[id_right[-1]]:  
                    #判断如果有两个相同编号的斑块连通域，则判断这两个的下一个，若有下一个挨着的编号，则分别与下一个连接，
                    #若没有，则要判断该区域是否被中心线分割，若有，则与中心线连接，若没有，则不连接，保存这些点。
                    right_next = right_i + right_counts[id_right[right_i]]
                    print(right_i, right_next)
                    if id_right[right_i]+1 == id_right[right_next]:
                        right_next_next = right_next + right_counts[id_right[right_next]]
                        for right_in in range(right_i, right_next):
                            dist = []
                            for right_jn in range(right_next, right_next_next):
                                distances = pairwise_distances(np.array(all_axis_coord_right[right_in]), device=torch.device("cuda"), y=np.array(all_axis_coord_right[right_jn]))
                                dist.append(np.min(distances))
                            ind = np.argmin(dist, axis=0)
                            generate_point_lines(all_axis_coord_right[right_in], all_axis_coord_right[right_next+ind], cta_data)
                            
                        for right_in in range(right_next, right_next_next):
                            dist = []
                            for right_jn in range(right_i, right_next):
                                distances = pairwise_distances(np.array(all_axis_coord_right[right_in]), device=torch.device("cuda"), y=np.array(all_axis_coord_right[right_jn]))
                                dist.append(np.min(distances))
                            ind = np.argmin(dist, axis=0)
                            generate_point_lines(all_axis_coord_right[right_in], all_axis_coord_right[right_i+ind], cta_data)

                    elif id_right[right_i]+1 != id_right[right_next]:
                        while right_i < right_next:
                            if centerline_right[right_i][0] == []:
                                #将这些值赋值到cta_data
                                generate_data(all_axis_coord_right[right_i], cta_data)
                            elif centerline_right[right_i][0] != []:
                                print('centerline_right[right_i][0]', centerline_right[right_i][0])
                                generate_point_lines(all_axis_coord_right[right_i], centerline_right[right_i][0], cta_data)
                            right_i += 1
                    right_i = right_next
                else:
                    right_i += 1
      
            #判断当前所连接连通域的编号是否连续，若连续，并且最后的编号是19，开始编号是0，则19连接0
            if len(id_right) >= 20:
                right_list = list(set(id_right))
                if right_list == 20:
                    right_ind = len(id_right)-right_counts[id_right[-1]]
                    right_ind_next = len(id_right)
                    right_0 = 0
                    right_0_next = right_counts[id_right[0]]
                    for right_in in range(right_ind, right_ind_next):
                        dist = []
                        for right_jn in range(right_0, right_0_next):
                            distances = pairwise_distances(np.array(all_axis_coord_right[right_in]), device=torch.device("cuda"), y=np.array(all_axis_coord_right[right_jn]))
                            dist.append(np.min(distances))
                        ind = np.argmin(dist, axis=0)
                        generate_point_lines(all_axis_coord_right[right_in], all_axis_coord_right[right_0+ind], cta_data)
                            
                    for right_in in range(right_0, right_0_next):
                        dist = []
                        for right_jn in range(right_ind, right_ind_next):
                            distances = pairwise_distances(np.array(all_axis_coord_right[right_in]), device=torch.device("cuda"), y=np.array(all_axis_coord_right[right_jn]))
                            dist.append(np.min(distances))
                        ind = np.argmin(dist, axis=0)
                        generate_point_lines(all_axis_coord_right[right_in], all_axis_coord_right[right_ind+ind], cta_data)

        print('c2', c)
        cpr_counts += 1
        all_cpr_coord = []
        all_cpr_contours = []
        all_cta_coord = []
        all_centerlines = []
        all_ids = []
 
    cta_data = fill_holes(cta_data, 0, to_numpy=True)

    time2 = time.time()
    print(time2-time1)
    return cta_data      

def cpr2cta_map_direct(cpr_coords, cpr_mask_data, cta_shape, cpr_map_index, patient, series, mask_names, contour_files, cpr_centerlines):
    time1 = time.time()
    cta_data = np.zeros(cta_shape[::-1], dtype='uint8')
    print(len(cpr_centerlines))
    cpr_coords = [cpr_coords[ii] for ii in cpr_map_index]
    cpr_masks = [cpr_mask_data[jj] for jj in cpr_map_index]
    contour_files = [contour_files[kk] for kk in cpr_map_index]
    cpr_counts = 0
    all_cpr_coord = [] #用来存cpr坐标
    all_cpr_contours = [] #用来存轮廓坐标
    all_cta_coord = [] #用来存轴位坐标
    all_centerlines = [] #用来存中心线坐标
    all_ids = [] #  用来存cpr的id

    for cout in range(len(cpr_centerlines)):
        c = 0
        for i in range(cpr_counts*20, (cpr_counts+1)*20):
            cpr_mask = cpr_masks[i]
            cpr_contour = contour_files[i]
            cpr_axis_coord = cpr_coords[i]
            print('cpr_contour', cpr_contour.shape)
            print('cpr_mask', cpr_mask.shape)
            area_index = ConnectComponent(cpr_mask)

            cpr_centerline = cpr_centerlines[cpr_counts]['vessel_group']
            for cen in range(len(cpr_centerline)):
                if cpr_centerline[cen]['name'] == mask_names[i]:
                    cpr_centerline_points = cpr_centerline[cen]['points']
                    break
            print(cpr_centerline[cen]['name'])
            if i%20 == 0:
                for r in range(len(area_index)):
                    mask_index = area_index[r]
                    #print(len(area_index[r]))
                    c += len(area_index[r])
                    cpr_coord = []
                    cpr_maskcontour = []
                    cta_coord = []
                    for cpr in mask_index:
                        cpr_coord.append([cpr[0],cpr[1]])
                        x, y, z = cpr_axis_coord[cpr[0],cpr[1]]
                        cta_data[int(round(x)), int(round(y)), int(round(z))] = 255 
                        cta_coord.append([x, y, z])
                        if cpr_contour[cpr[0],cpr[1]] == 255:
                            cpr_maskcontour.append([cpr[0],cpr[1]])
                    all_cpr_coord.append([cpr_coord])
                    all_cpr_contours.append([cpr_maskcontour])
                    all_cta_coord.append(cta_coord)
                    all_centerlines.append([cpr_centerline_points])
                    all_ids.append([i%20])

            if i%20 != 0:
                for r in range(len(area_index)):
                    mask_index = area_index[r]
                    #print(len(area_index[r]))
                    c += len(area_index[r])
                    cpr_coord = []
                    cpr_maskcontour = []
                    cta_coord = []
                    for cpr in mask_index:
                        cpr_coord.append([cpr[0],cpr[1]])
                        x, y, z = cpr_axis_coord[cpr[0],cpr[1]]
                        cta_data[int(round(x)), int(round(y)), int(round(z))] = 255
                        cta_coord.append([x, y, z])
                        if cpr_contour[cpr[0],cpr[1]] == 255:
                            cpr_maskcontour.append([cpr[0],cpr[1]])
                    
                    #为了区分同一个cpr上的不同斑块
                    for d in range(len(all_cta_coord)):
                        #print('cpr_coord',cpr_coord)
                        #print('all_cpr_coord_copy',all_cpr_coord_copy[d])
                        distances = pairwise_distances(np.array(cta_coord), device=torch.device("cuda"), y=np.array(all_cta_coord[d]))
                        if (distances<3**0.5).any():
                            all_cpr_coord[d].append(cpr_coord)
                            all_cpr_contours[d].append(cpr_maskcontour)
                            all_cta_coord[d].extend(cta_coord)
                            all_centerlines[d].append(cpr_centerline_points)
                            all_ids[d].append(i%20)
                            flag = 0
                            break
                        else:
                            flag = 1
                    #添加新增的连通域
                    if flag and (d == len(all_cta_coord)-1):  
                        all_cpr_coord.append([cpr_coord])
                        all_cpr_contours.append([cpr_maskcontour])
                        all_cta_coord.append(cta_coord)
                        all_centerlines.append([cpr_centerline_points])
                        all_ids.append([i%20])

        print('c1', c)
        cpr_counts += 1
        all_cpr_coord = []
        all_cpr_contours = []
        all_cta_coord = []
        all_centerlines = []
        all_ids = []
 

    time2 = time.time()
    print(time2-time1)
    return cta_data      


def generate_data(a, cta_data):
    for i in range(len(a)):
        fz, fy, fx = int(round(a[i][0])), int(round(a[i][1])), int(round(a[i][2]))
        cta_data[fz, fy, fx] = 1
    
    return cta_data

def generate_point_lines(a, b, cta_data):
    if len(a) < len(b):
        a, b = b, a
    a_len, b_len = len(a), len(b)
    map_step = b_len/a_len
    for i in range(len(a)):
        #print('i', i, int(i*map_step))
        generate_lines(a[i], b[int(i*map_step)], cta_data)
    
    return cta_data

def generate_lines(a, b, cta_data):
    fz, fy, fx, sz, sy, sx = int(round(a[0])), int(round(a[1])), int(round(a[2])), int(round(b[0])), int(round(b[1])), int(round(b[2]))
    point_line_length = math.ceil(math.sqrt(abs(fz-sz)**2+abs(fy-sy)**2+abs(fx-sx)**2))
    #print('point_line_length', point_line_length)

    z_dist = abs(fz-sz)
    y_dist = abs(fy-sy)
    x_dist = abs(fx-sx)
    points = []

    for i in range(0, point_line_length+1):
        point = []
        if z_dist != 0:
            if fz-sz < 0:
                point.append(fz + int(round(i/(point_line_length/z_dist))))
                #point.append(fz + int((i//(point_line_length/z_dist))))
            elif fz-sz > 0:
                point.append(fz - int(round(i/(point_line_length/z_dist))))
                #point.append(fz - int((i//(point_line_length/z_dist))))
        else:
            point.append(fz)
        if y_dist != 0:
            if fy-sy < 0:
                point.append(fy + int(round(i/(point_line_length/y_dist))))
                #point.append(fy + int((i//(point_line_length/y_dist))))
            elif fy-sy > 0:
                point.append(fy - int(round(i/(point_line_length/y_dist))))
                #point.append(fy - int((i//(point_line_length/y_dist))))
        else:
            point.append(fy)
        if x_dist != 0:
            if fx-sx < 0:
                point.append(fx + int(round(i/(point_line_length/x_dist))))
                #point.append(fx + int((i//(point_line_length/x_dist))))
            elif fx-sx > 0:
                point.append(fx - int(round(i/(point_line_length/x_dist))))
                #point.append(fx - int((i//(point_line_length/x_dist))))
        else:
            point.append(fx)
        points.append(point)

    #print('points', points)
    #final_points = []
    #用来判断各个点是否连接上，若不连接增加连接点
    '''
    for i in range(1, len(points)):
        if points[i][0] != points[i-1][0] and points[i][1] != points[i-1][1] and points[i][2] != points[i-1][2]:
            #final_points.append([points[i-1][0], points[i][1], points[i][2]])
            points.insert(i, [points[i-1][0], points[i][1], points[i][2]])
    '''
    #final_points.extend(points)
    #print('final_points', points)
    
    for data in points:
        cta_data[data[0]][data[1]][data[2]] = 1

    return cta_data

def read_centerline(cpr_centerline_paths):
    #print(cpr_centerline_paths)
    with open(cpr_centerline_paths, mode='r') as f:
        centerline = json.load(f)

    return centerline


plaque_newcpr_path = '/mnt/users/ffr_datasets/ffr_newcpr/'
plaque_mask_newmap_path = '/mnt/users/ffr_datasets/ffr_cpr_mask_newmap/' #'/mnt/users/ffr_10datasets/ffr_cpr_mask_newmap/'
plaque_mask_gen_path = '/mnt/users/ffr_datasets/ffr_cpr_mask_dealt/' #'/mnt/users/ffr_10datasets/ffr_cpr_mask_dealt/'
plaque_contour_gen_path = '/mnt/users/ffr_datasets/ffr_cpr_mask_contours/'
cta_data_gen_path = '/mnt/DrwiseDataNFS/drwise_runtime_env/data1/inputdata' #'/mnt/users/ffr_10datasets/ffr_plaque/'
mapping_gen_path = '/mnt/DrwiseDataNFS/drwise_runtime_env/data1/inputdata' #'/mnt/users/ffr_10datasets/ffr_cpr_dat/'
plaque_centerline_path = '/mnt/DrwiseDataNFS/drwise_runtime_env/data1/inputdata'
#dat_path = '/data1/zhangyongming/ffr_10datasets/ffr_cpr_dat/1036602/10791347/EE599B93_CTA/cprCoronary/'
#cpr_mask_dir = '/data1/zhangyongming/ffr_10datasets/ffr_cpr_mask_dealt/1036602/EE599B93_CTA/'
#cta_path = '/data1/zhangyongming/ffr_10datasets/ffr_plaque/1036602/10791347/EE599B93_CTA/'

for patient in sorted(os.listdir(plaque_mask_gen_path)):
    if patient != '1036604': #1036602
        continue
    print(patient)
    for series in os.listdir(os.path.join(plaque_mask_gen_path, patient)):
        #if series != '5F654861_CTA':
        #    continue
        print(series)
        mask_paths = sorted(glob.glob(os.path.join(plaque_mask_gen_path, patient, series, '*.png')))
        mask_names = [mask_path.split('/')[-1].split('.')[0] for mask_path in mask_paths]
        mask_files = [np.array(Image.open(mask_path)) for mask_path in mask_paths]
        #print(mask_names)
        contour_paths = sorted(glob.glob(os.path.join(plaque_contour_gen_path, patient, series, '*.png')))
        contour_names = [contour_path.split('/')[-1].split('.')[0] for contour_path in contour_paths]
        contour_files = [np.array(cv2.imread(contour_path, 0)) for contour_path in contour_paths]

        cpr_centerline_paths = glob.glob(os.path.join(plaque_centerline_path, patient, '*', series, 'centerline'))[0]
        cpr_centerline_paths = sorted(glob.glob(os.path.join(cpr_centerline_paths, '*.2d_cpr')))
        cpr_centerline_names = [cpr_centerline.split('/')[-1].split('.')[0] for cpr_centerline in cpr_centerline_paths]
        cpr_centerlines = [read_centerline(cpr_centerline_path) for cpr_centerline_path in cpr_centerline_paths]
        
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
        cta_data = cpr2cta_map(mapping_files, mask_files, cta_shape, roi_index_list, patient, series, mask_names, contour_files, cpr_centerlines)

        if not os.path.exists(os.path.join(plaque_mask_newmap_path, patient, series)):
            os.makedirs(os.path.join(plaque_mask_newmap_path, patient, series))
        cta_series = SERIES(series_path=cta_path, strict_check_series=True)
        save_nii(os.path.join(plaque_mask_newmap_path, patient, series), cta_series.flip, cta_data, 'mask_plaque_centerline.nii.gz')

        cta_data = cpr2cta_map_direct(mapping_files, mask_files, cta_shape, roi_index_list, patient, series, mask_names, contour_files, cpr_centerlines)
        save_nii(os.path.join(plaque_mask_newmap_path, patient, series), cta_series.flip, cta_data, 'mask_plaque_direct.nii.gz')