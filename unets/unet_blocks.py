
import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):
    """conv-norm-relu"""
    def __init__(self, in_channels, out_channels,kernel_size=3,padding=1, norm_layer=None):
        """
        :param in_channels:  输入通道
        :param out_channels: 输出通道
        :param kernel_size: 默认为3
        :param padding:    默认为1，
        :param norm_layer:  默认使用 batch_norm
        """
        super(ConvBlock,self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            norm_layer(out_channels) if norm_layer is not None else  nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.convblock(x)

class UNetBlock(nn.Module):
    """conv-norm-relu,conv-norm-relu"""
    def __init__(self, in_channels, out_channels,mid_channels=None,padding=0, norm_layer=None):
        """
        :param in_channels:
        :param out_channels:
        :param mid_channels:  默认 mid_channels==out_channels
        :param padding:     缺省设置为padding==0(论文中)
        :param norm_layer: 默认使用 batch_norm
        """
        super(UNetBlock,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.unetblock=nn.Sequential(
            ConvBlock(in_channels,mid_channels,padding=padding,norm_layer=norm_layer),
            ConvBlock(mid_channels, out_channels,padding=padding,norm_layer=norm_layer)
        )
    def forward(self, x):
        return self.unetblock(x)


class UNetUpBlock(nn.Module):
    """Upscaling then unetblock"""

    def __init__(self, in_channels, out_channels,padding=0,norm_layer=None, bilinear=True):
        """
        :param in_channels:
        :param out_channels:
        :param padding:     缺省设置为padding==0(论文中)
        :param norm_layer: 默认使用 batch_norm
        :param bilinear:  使用双线性插值，或转置卷积
        """

        super(UNetUpBlock,self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels , in_channels // 2,1,1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = UNetBlock(in_channels, out_channels,padding=padding,norm_layer=norm_layer)


    def crop(self,tensor,target_sz):
        _, _, tensor_height, tensor_width = tensor.size()
        diff_y = (tensor_height - target_sz[0]) // 2
        diff_x = (tensor_width - target_sz[1]) // 2
        return tensor[:, :, diff_y:(diff_y + target_sz[0]), diff_x:(diff_x + target_sz[1])]

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW

        x2=self.crop(x2,x1.shape[2:])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class UNetDownBlock(nn.Module):
    """maxpooling-unetblock"""

    def __init__(self, in_channels, out_channels,padding=0, norm_layer=None):
        super(UNetDownBlock,self).__init__()

        self.down=nn.Sequential(
            nn.MaxPool2d(2),
            UNetBlock(in_channels, out_channels,padding=padding, norm_layer=norm_layer),
        )
    def forward(self, inputs):
        return self.down(inputs)


class Unet_Encoder(nn.Module):
    def __init__(self, in_channels,base_channels,level,padding=0,norm_layer=None,):
        super(Unet_Encoder,self).__init__()
        self.encoder=nn.ModuleList()
        for i in range(level):
            if i==0:
                #第一层，特征图尺寸和原图大小一致
                self.encoder.append(UNetBlock(in_channels, base_channels*(2**i),
                                              padding=padding,norm_layer=norm_layer))
            else:
                self.encoder.append(UNetDownBlock( base_channels*(2**(i-1)),  base_channels*(2**i),
                                                   padding=padding,norm_layer=norm_layer))

    def forward(self, inputs):
        features=[]
        for block in self.encoder:
            inputs=block(inputs)
            features.append(inputs)
        return features



class UNet(nn.Module):
    def __init__(self,n_classes,base_channels=64,level=5,padding=0,norm_layer=None,bilinear=True):
        super(UNet, self).__init__()
        self.level=level
        self.base_channels=base_channels
        self.norm_layer=norm_layer
        self.padding=padding
        self.bilinear=bilinear
        self.encoder=self.build_encoder()
        self.decoder=self.build_decoder()
        self.outBlock=nn.Sequential(nn.Conv2d(base_channels,n_classes,1,1),nn.Sigmoid())
    def build_encoder(self):
        return Unet_Encoder(in_channels=3, base_channels=self.base_channels, level=self.level, padding=self.padding)
    def build_decoder(self):
        decoder=nn.ModuleList()
        for i in range(self.level-1): #有 self.level-1 个上采样块
            in_channels= self.base_channels*(2**(self.level-i-1))
            out_channels= self.base_channels*(2**(self.level-i-2))
            decoder.append(UNetUpBlock(in_channels,out_channels,
                                       padding=self.padding,norm_layer= self.norm_layer,bilinear=self.bilinear))
        return  decoder

    def forward(self,x):
        features=self.encoder(x)[0:self.level]
        # for feat in features:
        #     print(feat.shape)
        assert len(features)==self.level
        x=features[-1]
        for i,up_block in enumerate(self.decoder):
            x=up_block(x,features[-2-i])
            #print("shape:{}".format(x.shape))
        if self.outBlock is not None:
            x=self.outBlock(x)
        #加一个softmax激活函数 或则sigmoid也行
        return  x


if __name__=="__main__":

    ipt=torch.rand(1,3,572,572)

    unet1=UNet(10,base_channels=16,level=5)
    opt = unet1(ipt)
    print(opt.shape)

    unet2=UNet(10,base_channels=16,level=5,padding=1)
    opt = unet2(ipt)
    print(opt.shape)

