# -- coding: utf-8 --
import torch
import torchvision
from thop import profile

import model

# Model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
print('==> Building model..')

DCE_net = model.enhance_net_nopool()
DCE_net.apply(weights_init)
dummy_input = torch.randn(1, 3, 384, 384)
flops, params = profile(DCE_net, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
