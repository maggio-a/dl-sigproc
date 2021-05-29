# NOTE: The model architecture starts from the 3rd layer rather than from the first
# since it is trained on images of resolution 128x128.

import torch
import torch.nn.functional as F
from . import layers

import torchvision.models


class UNet(torch.nn.Module):

    class EncoderBlock(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int, ks: int, pad: int, bn: bool):
            super(UNet.EncoderBlock, self).__init__()

            self.conv = layers.PartialConv2d(in_channels, out_channels, kernel_size=(ks, ks), stride=(2, 2),
                                             padding=(pad, pad))
            self.bn = torch.nn.BatchNorm2d(out_channels) if bn else None
            self.relu = torch.nn.ReLU(inplace=True)

            self.reset_weights()

        @torch.no_grad()
        def reset_weights(self):
            torch.nn.init.normal_(self.conv.weight, 0.0, 0.02)
            if self.conv.bias is not None:
                torch.nn.init.constant_(self.conv.bias, 0.0)
            if self.bn is not None:
                torch.nn.init.normal_(self.bn.weight, 1.0, 0.02)
                torch.nn.init.constant_(self.bn.bias, 0.0)

        def forward(self, img_in, mask_in):
            img_out, mask_out = self.conv(img_in, mask_in)
            if self.bn is not None:
                img_out = self.bn(img_out)
            img_out = self.relu(img_out)
            return img_out, mask_out

    class DecoderBlock(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int, ks: int, pad: int, bn: bool, leaky_relu: bool):
            super(UNet.DecoderBlock, self).__init__()

            self.upsampler = torch.nn.UpsamplingNearest2d(scale_factor=2)
            self.conv = layers.PartialConv2d(in_channels, out_channels, kernel_size=(ks, ks), stride=(1, 1),
                                             padding=(pad, pad))
            self.bn = torch.nn.BatchNorm2d(out_channels) if bn else None
            self.leaky_relu = torch.nn.LeakyReLU(0.2, inplace=True) if leaky_relu else None

            self.reset_weights()

        @torch.no_grad()
        def reset_weights(self):
            torch.nn.init.normal_(self.conv.weight, 0.0, 0.02)
            if self.conv.bias is not None:
                torch.nn.init.constant_(self.conv.bias, 0.0)
            if self.bn is not None:
                torch.nn.init.normal_(self.bn.weight, 1.0, 0.02)
                torch.nn.init.constant_(self.bn.bias, 0.0)

        def forward(self, img_in, mask_in, img_skip, mask_skip):
            # upscale data from lower layer
            img_up = self.upsampler(img_in)
            mask_up = self.upsampler(mask_in)

            # concatenate with skip-link data
            img_out = torch.cat((img_skip, img_up), dim=1)
            mask_out = torch.cat((mask_skip, mask_up), dim=1)

            # convolve
            img_out, mask_out = self.conv(img_out, mask_out)
            if self.bn is not None:
                img_out = self.bn(img_out)
            if self.leaky_relu is not None:
                img_out = self.leaky_relu(img_out)
            return img_out, mask_out

    def __init__(self):
        super(UNet, self).__init__()

        self.e1 = UNet.EncoderBlock(3, 64, 7, 3, False)
        self.e2 = UNet.EncoderBlock(64, 128, 5, 2, True)
        self.e3 = UNet.EncoderBlock(128, 256, 5, 2, True)
        self.e4 = UNet.EncoderBlock(256, 512, 3, 1, True)
        self.e5 = UNet.EncoderBlock(512, 512, 3, 1, True)
        self.e6 = UNet.EncoderBlock(512, 512, 3, 1, True)
        self.e7 = UNet.EncoderBlock(512, 512, 3, 1, True)
        self.e8 = UNet.EncoderBlock(512, 512, 3, 1, True)

        self.d9 = UNet.DecoderBlock(512+512, 512, 3, 1, True, True)
        self.d10 = UNet.DecoderBlock(512+512, 512, 3, 1, True, True)
        self.d11 = UNet.DecoderBlock(512+512, 512, 3, 1, True, True)
        self.d12 = UNet.DecoderBlock(512+512, 512, 3, 1, True, True)
        self.d13 = UNet.DecoderBlock(512+256, 256, 3, 1, True, True)
        self.d14 = UNet.DecoderBlock(256+128, 128, 3, 1, True, True)
        self.d15 = UNet.DecoderBlock(128+64, 64, 3, 1, True, True)
        self.d16 = UNet.DecoderBlock(64+3, 3, 3, 1, False, False)

        self.activation = torch.nn.Sigmoid()

    def forward(self, img_in, mask_in):
        i1, m1 = self.e1.forward(img_in, mask_in)
        i2, m2 = self.e2.forward(i1, m1)
        i3, m3 = self.e3.forward(i2, m2)
        i4, m4 = self.e4.forward(i3, m3)
        i5, m5 = self.e5.forward(i4, m4)
        i6, m6 = self.e6.forward(i5, m5)
        i7, m7 = self.e7.forward(i6, m6)
        i8, m8 = self.e8.forward(i7, m7)

        i9, m9 = self.d9.forward(i8, m8, i7, m7)
        i10, m10 = self.d10.forward(i9, m9, i6, m6)
        i11, m11 = self.d11.forward(i10, m10, i5, m5)
        i12, m12 = self.d12.forward(i11, m11, i4, m4)
        i13, m13 = self.d13.forward(i12, m12, i3, m3)
        i14, m14 = self.d14.forward(i13, m13, i2, m2)
        i15, m15 = self.d15.forward(i14, m14, i1, m1)
        i16, m16 = self.d16.forward(i15, m15, img_in, mask_in)

        ifinal = self.activation(i16)

        return ifinal, m16


# Implementation of the loss function
class IrregularHolesLoss(torch.nn.Module):

    @staticmethod
    def normalize(x: torch.Tensor):
        device = x.device
        return torch.div(x - torch.tensor([[[0.485]], [[0.456]], [[0.406]]], device=device),
                         torch.tensor([[[0.229]], [[0.224]], [[0.225]]], device=device))

    vgg16_conv_layers: torch.nn.ModuleList
    l1: torch.nn.Module

    def __init__(self):
        super(IrregularHolesLoss, self).__init__()
        vgg_full = torchvision.models.vgg16(pretrained=True)

        # We need the first 3 max_pool layers, which are at indices 4, 9 and 16
        self.vgg16_conv_layers = torch.nn.ModuleList(vgg_full.features[:17])

        # Freeze the parameters of the vgg16 layers to save some resources
        for param in self.vgg16_conv_layers.parameters():
            param.requires_grad = False

        self.l1 = torch.nn.L1Loss()

    def _extract_vgg_features(self, x: torch.Tensor):
        """ Extracts vectorized VGG features. """
        vgg_features = []
        vgg_shapes = []
        for i, layer in enumerate(self.vgg16_conv_layers):
            x = layer(x)
            if i in [4, 9, 16]:
                batch_size, c, h, w = x.shape
                vgg_features.append(x.reshape(batch_size, c, h * w))
                vgg_shapes.append((batch_size, c, h * w))
        return vgg_features, vgg_shapes

    def forward(self, img_gt, mask_in, img_out) -> torch.Tensor:
        img_comp = torch.mul(img_gt, mask_in) + torch.mul(img_out, (1 - mask_in))

        hole_loss = ((1 - mask_in) * torch.abs(img_gt - img_out)).mean()
        valid_loss = (mask_in * torch.abs(img_gt - img_out)).mean()

        img_gt_normalized = self.normalize(img_gt)
        img_out_normalized = self.normalize(img_out)
        img_comp_normalized = self.normalize(img_comp)

        gt_features, gt_shapes = self._extract_vgg_features(img_gt_normalized)
        out_features, out_shapes = self._extract_vgg_features(img_out_normalized)
        comp_features, comp_shapes = self._extract_vgg_features(img_comp_normalized)

        assert gt_shapes == out_shapes and gt_shapes == comp_shapes

        for i in range(len(gt_features)):
            gt_features[i].requires_grad = False

        perceptual_loss = sum(self.l1(out_features[i], gt_features[i]) + self.l1(comp_features[i], gt_features[i])
                              for i in range(3))

        K = [(1 / (c*wh)) for b, c, wh in gt_shapes]

        gt_gram = [K[i] * torch.matmul(gt_features[i], torch.transpose(gt_features[i], 1, 2)) for i in range(3)]
        out_gram = [K[i] * torch.matmul(out_features[i], torch.transpose(out_features[i], 1, 2)) for i in range(3)]
        comp_gram = [K[i] * torch.matmul(comp_features[i], torch.transpose(comp_features[i], 1, 2)) for i in range(3)]

        style_out_loss = sum(self.l1(out_gram[i], gt_gram[i]) for i in range(3))

        style_comp_loss = sum(self.l1(comp_gram[i], gt_gram[i]) for i in range(3))

        with torch.no_grad():
            dilated_mask = F.conv2d(1 - mask_in,
                                    torch.ones((mask_in.shape[1], mask_in.shape[1], 3, 3), device=mask_in.device),
                                    padding=1)
            dilated_mask = torch.clamp(dilated_mask, 0, 1)

        r_comp = torch.mul(dilated_mask, img_comp)  # img_comp restricted to the 1-pixel dilation of the hole region

        total_variation_loss = self.l1(r_comp[:, :, :-1, :], r_comp[:, :, 1:, :]) + self.l1(
            r_comp[:, :, :, :-1], r_comp[:, :, :, 1:])

        return 6 * hole_loss + valid_loss + 0.05 * perceptual_loss\
            + 120 * (style_out_loss + style_comp_loss) + 0.1 * total_variation_loss

