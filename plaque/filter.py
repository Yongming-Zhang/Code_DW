import math
import numbers
import torch
import numpy as np
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

def apply_mean_filter(x, kernel_size, use_2d_smooth, use_ball=True):
    if kernel_size % 2 != 1:
        raise RuntimeError("kernel_size should be odd value.")

    src_dtype = x.dtype
    x = x.to(torch.float16)
    if len(x.shape) > 2 and use_2d_smooth:
        x = torch.unsqueeze(x, dim=1)
        s_dim = [1]
    else:
        x = torch.unsqueeze(x, dim=0)
        x = torch.unsqueeze(x, dim=0)
        s_dim = [0, 0]

    if use_ball:
        kernel = ball(kernel_size//2)
    else:
        kernel = np.ones([kernel_size, kernel_size, kernel_size])
    kernel = torch.as_tensor(kernel, dtype=torch.float16).to(x.device)
    kernel = torch.unsqueeze(kernel, dim=0)
    kernel = torch.unsqueeze(kernel, dim=0)

    if use_2d_smooth:
        kernel = kernel[:,:,kernel_size//2,:,:]
        kernel = kernel / kernel.sum()
        x = F.pad(x, [kernel_size//2]*4, mode='replicate')
        x = F.conv2d(x, kernel)
    else:
        kernel = kernel / kernel.sum()
        x = F.pad(x, [kernel_size//2]*6, mode='replicate')
        x = F.conv3d(x, kernel)

    for i in s_dim:
        x = torch.squeeze(x, dim=i)
    if src_dtype == torch.bool:
        x = x >= 0.5
    else:
        x = x.to(src_dtype)
    return x
