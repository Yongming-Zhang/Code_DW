import numpy as np
from scipy.interpolate import griddata

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
        print(pts)
        ri = griddata(self.uvw, self.r, pts, method=self.method)

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
        print(theta)
        print(phi)
        print(z)
        theta = theta.flatten()
        phi = phi.flatten()
        z = z.flatten()        
        print(theta)
        print(phi)
        print(z)
        print(M,N,K)
        return theta,phi,z,M,N,K
    
grid = GridData3D([0,0,0,0,1,1,1,1,1,10],[0,0,1,1,0,1,1,1,1,8],[0,1,0,1,0,0,0,1,2,2],[1,1,0,0,1,1,0,0,0.9,2])
print(grid([0.5,0.2,1.1,1,6,10],[0.5,0.2,1,1,5,8],[0.5,0.2,1.5,1.5,1,1]))

import torch
a = torch.rand((1, 1, 5, 5))
print(a)

# x1 = 2.5, x2 = 4.5, y1 = 0.5, y2 = 3.5
# out_w = 2, out_h = 3
size = torch.Size((1, 1, 3, 2))
print(size)

# theta
theta_np = np.array([[0.5, 0, 0.75], [0, 0.75, 0]]).reshape(1, 2, 3)
theta = torch.from_numpy(theta_np)
print('theta:')
print(theta)

flowfield = torch.nn.functional.affine_grid(theta, size, align_corners=True)
print(flowfield, flowfield.size())
sampled_a = torch.nn.functional.grid_sample(a, flowfield.to(torch.float32), align_corners=True)
sampled_a = sampled_a.numpy().squeeze()
print('sampled_a:')
print(sampled_a)

# compute bilinear at (0.5, 2.5), using (0, 3), (0, 4), (1, 3), (1, 4)
# quickly compute(https://blog.csdn.net/lxlclzy1130/article/details/50922867)
print()
coeff = np.array([[0.5, 0.5]])
A = a[0, 0, 0:2, 2:2+2]
print('torch sampled at (0.5, 3.5): %.4f' % sampled_a[0,0])
print('numpy compute: %.4f' % np.dot(np.dot(coeff, A), coeff.T).squeeze())
