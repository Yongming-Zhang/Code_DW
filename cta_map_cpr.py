#-*-coding:utf-8-*-
import os
import struct
import numpy as np
import glob
import cv2
import SimpleITK as sitk
from PIL import Image
from scipy.interpolate import griddata
from skimage import measure
from scipy import ndimage
from scipy import misc
import torch  

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

def compute_dice(pred, gt): 
    pred[pred>0] = 1
    gt[gt>0] = 1
    print('pred', np.unique(pred))
    print('gt', np.unique(gt))
    pred, gt = pred.flatten(), gt.flatten()
    num = np.sum(pred * gt)
    den1 = np.sum(pred)
    den2 = np.sum(gt)
    epsilon = 1e-4
    dice = (2 * num + epsilon) / (den1 + den2 + epsilon)
    return dice

def cta2cpr(cpr_coords, plaque_cta, cpr_map_index, patient, series, mask_names):
    print(plaque_cta.shape)
    dice_value = []
    cta_w, cta_h, cta_d = plaque_cta.shape
    #plaque_cta = (plaque_cta>=0.5).astype(np.float32)
    #plaque_cta = plaque_cta.reshape(1, 1, plaque_cta.shape[0], plaque_cta.shape[1], plaque_cta.shape[2])
    #plaque_cta = torch.as_tensor(plaque_cta).cuda()
    for p in range(len(cpr_coords)):
        cpr_coord = cpr_coords[p]
        print('cpr_coord', cpr_coord.shape)
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
        #cpr_data[cpr_data>=128] = 255
        #cpr_data[cpr_data<128] = 0
        '''
        xs, ys, zs = cpr_coord[:, :, 0], cpr_coord[:, :, 1], cpr_coord[:, :, 2]
        cpr_data = ndimage.map_coordinates(plaque_cta, [zs, ys, xs], order=1, cval=0).astype(plaque_cta.dtype)
        '''
        cpr_h, cpr_w = cpr_coord.shape[:2]
        cpr_data = np.zeros([cpr_h, cpr_w]) 
        for i in range(cpr_h):
            for j in range(cpr_w):
                x, y, z = cpr_coord[i, j]
                x, y, z = int(round(x)), int(round(y)), int(round(z))
                if 0 <= x < cta_w and 0 <= y < cta_h and 0 <= z < cta_d:
                    cpr_data[i, j] = plaque_cta[x, y, z]
        '''
        '''
        cpr_h, cpr_w = cpr_coord.shape[:2]
        cpr_list = []
        for i in range(cpr_h):
            for j in range(cpr_w):
                x, y, z = cpr_coord[i, j]
                x, y, z = int(round(x)), int(round(y)), int(round(z))
                cpr_list.append([z, y, x])
        '''
        '''
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
                    values.append(plaque_cta[i,j,k])
        print(len(points))
        
        points = np.array(points)
        print(points[:,0])
        grid_x, grid_y, grid_z = cpr_coord[:, :, 0], cpr_coord[:, :, 1], cpr_coord[:, :, 2]
        grid = GridData3D(points[:,0], points[:,1], points[:,2], values)
        cpr_data = grid(grid_x, grid_y, grid_z)
        '''

        cpr_data[cpr_data>=0.5] = 1
        cpr_data[cpr_data<0.5] = 0
        print('cpr_data', np.unique(cpr_data))
        if not os.path.exists(os.path.join(plaque_newcpr_path, patient, series)):
       	    os.makedirs(os.path.join(plaque_newcpr_path, patient, series))
        cv2.imwrite(os.path.join(plaque_newcpr_path, patient, series, mask_names[p]+'.png'), cpr_data)
        origin_cpr_data = cv2.imread(os.path.join(plaque_mask_gen_path, patient, series, mask_names[p]+'.png'), 0)
        dice = compute_dice(cpr_data, origin_cpr_data)
        dice_value.append(dice)
        print('dice', dice)
    print('meandice', np.mean(dice_value)) 

plaque_newcpr_path = '/mnt/users/ffr_datasets/ffr_newcpr/'
plaque_mask_newmap_path = '/mnt/users/ffr_datasets/ffr_cpr_mask_newmap/'
plaque_mask_gen_path = '/mnt/users/ffr_datasets/ffr_cpr_mask_dealt/'
cta_data_gen_path = '/mnt/DrwiseDataNFS/drwise_runtime_env/data1/inputdata'
mapping_gen_path = '/mnt/DrwiseDataNFS/drwise_runtime_env/data1/inputdata'
for patient in sorted(os.listdir(plaque_mask_newmap_path)):
    if patient != '1036604':
        continue
    print(patient)
    for series in os.listdir(os.path.join(plaque_mask_newmap_path, patient)):
        print(series)
        plaque = os.path.join(plaque_mask_newmap_path, patient, series, 'mask_plaque_newwsw.nii.gz')
        plaque_cta_mask = sitk.ReadImage(plaque)
        plaque_cta = sitk.GetArrayFromImage(plaque_cta_mask)

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

        #cta_path = glob.glob(os.path.join(cta_data_gen_path, patient, '*', series.split('_')[0]))[0]
        #cta_paths = glob.glob(os.path.join(cta_path, '*.dcm'))
        #cta_paths.sort(key=lambda ele: int(ele.split('/')[-1].split('.')[0]))

        #cta_shape = [len(cta_paths), 512, 512]                                               
        
        roi_index_list = []
        for i in range(len(mask_names)):
            if mask_names[i] in mapping_names:        
                roi_index = mapping_names.index(mask_names[i])
                roi_index_list.append(roi_index)
        #print(mask_names)
        #print(mapping_names)
        cta2cpr(mapping_files, plaque_cta, roi_index_list, patient, series, mask_names)

        
