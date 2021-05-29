# Based on the reference implementation https://github.com/NVIDIA/partialconv

import torch
import torch.nn.functional as F


class PartialConv2d(torch.nn.Module):
    """
    2d Partial convolution layer implementation
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True):
        assert (type(kernel_size) is tuple and len(
            kernel_size) == 2), 'PartialConv2d argument kernel_size must be a tuple of length 2'
        assert (type(stride) is tuple and len(stride) == 2), 'PartialConv2d argument stride must be a tuple of length 2'
        assert (type(padding) is tuple and len(
            padding) == 2), 'PartialConv2d argument padding must be a tuple of length 2'
        assert (type(bias) is bool), 'PartialConv2d bias argument must be of type bool'
        assert (in_channels % groups == 0), 'PartialConv2d `in_channels` must be a multiple of `groups`'

        super(PartialConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.register_buffer('kernel_ones',
                             torch.ones((out_channels, in_channels//groups, kernel_size[0], kernel_size[1])),
                             persistent=False)

        self.kernel_volume = (self.kernel_ones.shape[1]) * self.kernel_ones.shape[2] * self.kernel_ones.shape[3]

        self.weight = torch.nn.Parameter(torch.zeros((out_channels, in_channels // groups, *kernel_size)))
        # following the _ConvNd implementation here...
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, in_image: torch.Tensor, in_mask: torch.Tensor):
        with torch.no_grad():
            out_mask = F.conv2d(in_mask, self.kernel_ones, bias=None, stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=self.groups)

            out_mask[out_mask == 0] = -1
            scaling = torch.div(self.kernel_volume, out_mask)
            out_mask = torch.clamp(out_mask, 0, 1)
            scaling = torch.mul(scaling, out_mask)  # clear masked pixels from self.scaling
            
            # the above scaling is correct only if the convolution kernel weights are all equal, in general
            # the numerically correct scaling depends on the kernel weights:

            # compute the group-wise kernel weight sum, and divide by the output pixel-wise masked kernel sum
            # !!! unstable...
            #self.scaling = torch.div(
            #    torch.sum(self.weight, dim=(1, 2, 3)).view(1, self.out_channels, 1, 1),
            #    F.conv2d(in_mask, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation,
            #             groups=self.groups) + self.eps
            #)
            #self.out_mask = torch.clamp(self.out_mask, 0, 1)
            #self.scaling = torch.mul(self.scaling, self.out_mask)  # clear masked pixels from self.scaling (not needed)

        x = F.conv2d(torch.mul(in_image, in_mask), self.weight, self.bias, stride=self.stride, padding=self.padding,
                     dilation=self.dilation, groups=self.groups)

        if self.bias is not None:
            b = self.bias.view(1, self.out_channels, 1, 1)
            out_image = torch.mul(x - b, scaling) + b  # do not scale the bias term
        else:
            out_image = torch.mul(x, scaling)

        return out_image, out_mask
