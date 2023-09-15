import numpy as np
import jittor as jt
import jittor.nn as nn
from network.spectral_norm import spectral_norm

class ConvEncoder(nn.Module):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.nef  
        #ndf=64
        # ndf = 32
        #norm_layer = nn.InstanceNorm2d(ndf, affine=False)
        #conv_layer1 =nn.Conv2d(3, ndf, kw, stride=1, padding=pw)
        norm_layer = get_nonspade_norm_layer(opt, 'spectralinstance')
        #self.layer1 = norm_layer(conv_layer1)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=1, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.ConvTranspose2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        #self.layer4 = norm_layer(nn.ConvTranspose(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))
        
        self.tanh = nn.Tanh()
        self.pool = RegionWiseAvgPooling()
        #self.pool = nn.AdaptiveAvgPool2d(1)

        self.actvn = nn.LeakyReLU(0.2)
        self.opt = opt

    def execute(self, x,mask):
        '''
        if x.size(2) != 96 or x.size(3) != 128:
            x = nn.interpolate(x, size=(96, 128), mode='bilinear')
        '''
        if x.size(2) != 192 or x.size(3) != 256:
            x = nn.interpolate(x, size=(192, 256), mode='bilinear')
            #为了显存资源，这里调小
        '''
        #print(x.shape)
        if x.size(2) != 384 or x.size(3) != 512:
            x = nn.interpolate(x, size=(384, 512), mode='bilinear')'''

        x = self.layer1(x)
        #print(x.shape)#[1,3,384,512,]
        x = self.layer2(self.actvn(x))
        #print(x.shape)#[1,3,384,512,]
        x = self.layer3(self.actvn(x))
        #print(x.shape)#[1,3,384,512,]
        x = self.layer4(self.actvn(x))
        #print(x.shape)#[1,3,384,512,]
        x = self.layer5(self.actvn(x))
        #print(x.shape)
        x = self.tanh(x)
        #print(x.shape)[1,3,384,512,]
        out = self.pool(feature_map=x, mask=mask)
        #out = self.pool(x)
        #print(out.shape)
        #[1,30,3,]
        #print(mask.shape)[1,3,384,512,]

        return out


class RegionWiseAvgPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def execute(self, feature_map, mask):
        if mask.size(2) != feature_map.size(2) or mask.size(3) != feature_map.size(3):
            mask = nn.interpolate(mask, size=(feature_map.size(2), feature_map.size(3)), mode='bilinear',
                                 align_corners=True)
            mask = (mask >= 0.5).type_as(mask)
        out = list()
        
        for i in range(mask.size(1)):
            #region_mask = jt.concat([mask[:, i, :, :].unsqueeze(1)] * feature_map.size(1), dim=1)
            #out.append(self.avg_pool(region_mask * feature_map).squeeze(2).squeeze(2).unsqueeze(1))
            #修改为每一类都对于整张图来提取风格,逼迫不纳入纹理
            out.append(self.avg_pool(feature_map).squeeze(2).squeeze(2).unsqueeze(1))
            
        #修改为每一类都对于整张图来提取风格
        #out.append(self.avg_pool(feature_map).squeeze(2).squeeze(2).unsqueeze(1))
        return jt.concat(out, dim=1)


    
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            setattr(layer, 'bias', None)
            # layer.load_parameters({'bias': None})

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(
                get_out_channel(layer), affine=False)
        else:
            raise ValueError(
                'normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer
'''
if __name__ == '__main__':
    opt = None
    encoder = ConvEncoder(opt)
    x = jt.randn((3, 3, 256, 256))
    mask = jt.zeros((3, 10, 256, 256))
    mask[:, :, 128:, :] = 1
    out = encoder(x, mask)
    print(1)'''