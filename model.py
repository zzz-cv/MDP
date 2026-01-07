import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import pytorch_colors as colors
import numpy as np
from nonlocal_block_embedded_gaussian import NONLocalBlock2D
from hist import get_hist, get_hist_conv, pack_tensor
from drconv1 import DRConv,HistDRConv



class DRDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, **kargs):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            DRConv(in_channels, mid_channels, kernel_size=3, region_num=2, padding=1, **kargs),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DRConv(mid_channels, out_channels, kernel_size=3, region_num=2, padding=1, **kargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        res = self.double_conv(x)
        self.guide_features = []
        self.guide_features.append(self.double_conv[0].guide_feature)
        self.guide_features.append(self.double_conv[3].guide_feature)
        return res


class enhance_net_nopool(nn.Module):

    def __init__(self):
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        n_bins = 8
        down_ratio = 1
        extra_c_num = 24

        self.hist_conv = get_hist_conv(n_bins, down_ratio)





        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True)
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv5 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv6 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv7 = nn.Conv2d(number_f,24,3,1,1,bias=True)
        self.e_conv8 = nn.Conv2d(number_f,24,3,1,1,bias=True)
        self.e_conv9 = nn.Conv2d(number_f,24,3,1,1,bias=True)         
        # self.h_conv1 = DRConv(3, number_f, 3)
        # self.h_conv2 = DRConv(number_f, number_f, 3)
        # self.h_conv3 = DRConv(number_f,number_f,3)
        # self.h_conv4 = DRConv(number_f,number_f,3)
        # self.h_conv5 = DRConv(number_f*2,number_f,3)
        # self.h_conv6 = DRConv(number_f*2,number_f,3)
        # self.h_conv7 = DRConv(number_f*2,24,3)

        self.h_conv1 = DRConv(number_f, number_f, 3)
        self.h_conv2 = DRConv(number_f, number_f, 3)
        self.h_conv3 = DRConv(number_f, number_f, 3)
        self.h_conv4 = DRConv(number_f, number_f, 3)
        self.h_conv5 = DRConv(number_f, number_f, 3)
        self.h_conv6 = DRConv(number_f, number_f, 3)
        self.h_conv7 = DRConv(number_f, 24, 3)
        


        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)



    def forward(self, x):
        n_bins = 2
        bs =x.shape[0]


        histmap = get_hist(x, n_bins,grayscale=False)
        histmap1 = pack_tensor(histmap, n_bins).detach()
        histmap_t = histmap1.reshape(bs, -1, *histmap1.shape[-2:])
        histmap_gray = get_hist(x, n_bins,grayscale=True)
        histmap_gray = pack_tensor(histmap_gray, n_bins).detach()
        guide_mask = torch.zeros_like(histmap_gray).scatter_(1, histmap_gray.argmax(dim=1, keepdim=True),
                                                              1)
        vector1 = guide_mask[:, 0:1, :, :]
        #print("guide_mask.shape()")
        #print(guide_mask.shape)
        vector2 = guide_mask[:, 1:2, :, :]
        #vector3 = guide_mask[:, 2:3, :, :]





        x1 = self.relu(self.e_conv1(x))
        x1 = self.relu(self.h_conv1(x1, histmap_t))
        #x2 = self.relu(self.e_conv2(x1))
        x2 = self.relu(self.h_conv2(x1, histmap_t))
        #x3 = self.relu(self.e_conv3(x2))
        x3 = self.relu(self.h_conv3(x2, histmap_t))
        #x4 = self.relu(self.e_conv4(x3))
        x4 = self.relu(self.h_conv3(x3, histmap_t))
        # x1 = self.relu(self.h_conv1(x, histmap_t))
        #x2 = self.relu(self.h_conv2(x1, histmap_t))
        #x3 = self.relu(self.h_conv3(x2, histmap_t))
        #x4 = self.relu(self.h_conv4(x3, histmap_t))


        #x5= self.e_conv5(torch.cat([x3, x4], 1))
        #x5 = self.relu(x5)
        #x6= self.e_conv6(torch.cat([x2, x5], 1))
        #x6 = self.relu(x6)
        #x5= self.e_conv5(x4)
        #x5 = self.relu(x5)
        x5 = self.relu(self.h_conv5(x4, histmap_t))
        #x6= self.e_conv6(x5)
        #x6 = self.relu(x6)
        x6 = self.relu(self.h_conv6(x5, histmap_t))
        
        #x7= self.h_conv5(x6, histmap_t)
        #x7= self.relu(x7)
        #x8= self.h_conv6(x7, histmap_t)
        #x8 = self.relu(x8)
        # x7 = self.h_conv7(torch.cat([x1, x6], 1), histmap_t)
        x7 = torch.cat([x1, x6], 1)
        x9 = self.e_conv7(x6)
        x10 = self.e_conv8(x6)
        #x11 = self.e_conv9(x6)
        xr_1 = F.tanh(x9)
        xr_2 = F.tanh(x10)
        #xr_3 = F.tanh(x11)

        xr_1 = xr_1*vector1
        xr_2 = xr_2*vector2
        #xr_3 = xr_3*vector3

        r1_1,r2_1,r3_1,r4_1,r5_1,r6_1,r7_1,r8_1 = torch.split(xr_1, 3, dim=1)
        r1_2,r2_2,r3_2,r4_2,r5_2,r6_2,r7_2,r8_2 = torch.split(xr_2, 3, dim=1)
        #r1_3,r2_3,r3_3,r4_3,r5_3,r6_3,r7_3,r8_3 = torch.split(xr_3, 3, dim=1)



        x1 = x* vector1
        x1 = x1 + r1_1*(torch.pow(x1,2)-x1)
        x1 = x1 + r2_1*(torch.pow(x1,2)-x1)
        x1 = x1 + r3_1*(torch.pow(x1,2)-x1)
        x1 = x1 + r4_1*(torch.pow(x1,2)-x1)
        x1 = x1 + r5_1*(torch.pow(x1,2)-x1)
        x1 = x1 + r6_1*(torch.pow(x1,2)-x1)
        x1 = x1 + r7_1*(torch.pow(x1,2)-x1)
        enhance_image1 = x1 + r8_1*(torch.pow(x1,2)-x1)
        r_1 = torch.cat([r1_1, r2_1, r3_1, r4_1, r5_1, r6_1, r7_1, r8_1], 1)

        x2 = x * vector2
        x2 = x2 + r1_2 * (torch.pow(x2 , 2) - x2 )
        x2 = x2 + r2_2 * (torch.pow(x2, 2) - x2)
        x2 = x2 + r3_2 * (torch.pow(x2, 2) - x2)
        x2 = x2 + r4_2 * (torch.pow(x2, 2) - x2)
        x2 = x2 + r5_2 * (torch.pow(x2, 2) - x2)
        x2 = x2 + r6_2 * (torch.pow(x2, 2) - x2)
        x2 = x2 + r7_2 * (torch.pow(x2, 2) - x2)
        enhance_image2 = x2 + r8_2 * (torch.pow(x2, 2) - x2)
        r_2 = torch.cat([r1_2, r2_2, r3_2, r4_2, r5_2, r6_2, r7_2, r8_2], 1)
        #
        #x3 = x * vector3
        #x3 = x3 + r1_3 * (torch.pow(x3, 2) - x3)
        #x3 = x3 + r2_3 * (torch.pow(x3, 2) - x3)
        #x3 = x3 + r3_3 * (torch.pow(x3, 2) - x3)
        #x3 = x3 + r4_3 * (torch.pow(x3, 2) - x3)
        #x3 = x3 + r5_3 * (torch.pow(x3, 2) - x3)
        #x3 = x3 + r6_3 * (torch.pow(x3, 2) - x3)
        #x3 = x3 + r7_3 * (torch.pow(x3, 2) - x3)
        #enhance_image3 = x3 + r8_3 * (torch.pow(x3, 2) - x3)
        #r_3 = torch.cat([r1_3, r2_3, r3_3, r4_3, r5_3, r6_3, r7_3, r8_3], 1)
        #enhance_image = enhance_image1 * mask0 +enhance_image1* mask1 + enhance_image1*mask2
        #r = r_1 * mask0 + r_2 * mask1 + r_3 * mask2



        return enhance_image2,enhance_image1,enhance_image2+enhance_image1,r_1+r_2

class DeepWBNet(nn.Module):

    def __init__(self, opt=None):
        super(DeepWBNet, self).__init__()
        nf = 32
        self.out_net = nn.Sequential(
            nn.Conv2d(6, nf, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.ReLU(inplace=True),
            NONLocalBlock2D(nf, sub_sample='bilinear', bn_layer=False),
            nn.Conv2d(nf, nf, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, 3, 1),
            NONLocalBlock2D(3, sub_sample='bilinear', bn_layer=False),
        )
        self.res = {}


    def forward(self,brighten_x1,darken_x1):
        # ──────────────────────────────────────────────────────────


        fused_x = torch.cat([brighten_x1, darken_x1], dim=1)

        weight_map = F.softmax(self.out_net(fused_x), dim=1)  # <- 3 channels, [ N, 3, H, W ]
        w2 = weight_map[:, 0, ...].unsqueeze(1)
        w3 = weight_map[:, 1, ...].unsqueeze(1)
        w3 = (1 - w2)

        out = brighten_x1 * w2 + darken_x1 * w3
        return out

