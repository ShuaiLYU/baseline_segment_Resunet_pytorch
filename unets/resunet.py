from unets.unet_blocks import  *
from unets.resnet_blocks import  _resnet,BasicBlock,Bottleneck
"""
1. resnet_net 采用了5个不同尺度的特征图图  level：5
2. 用三个3*3卷积代替 7*7卷积，并且步长全部为1,得到与原始图片尺寸相同的特征
3. base_channels控制着网络的宽度
4.   stride：1   网络输出与输入尺寸相同
"""
class Res18_UNet(UNet):
    def __init__(self,n_classes,norm_layer=None,bilinear=True,**kwargs):
        self.base_channels = kwargs.get("base_channels",32)  # resnet18 和resnet34 这里为 32 , 64
        level=kwargs.get("level",5)
        self.b_RGB = kwargs.get("level", True)

        padding = 1
        super(Res18_UNet,self).__init__(n_classes, self.base_channels,level,padding,norm_layer,bilinear)

    def build_encoder(self):
        return _resnet('resnet18', BasicBlock, [2, 2, 2, 2],base_planes= self.base_channels,b_RGB=self.b_RGB )




class Res50_UNet(UNet):
    def __init__(self,n_classes,norm_layer=None,bilinear=True):
        self.base_channels = 64     # resnet50 ，resnet101和resnet152 这里为 64, 128,256
        level = 5
        padding = 1
        super(Res50_UNet,self).__init__(n_classes, self.base_channels,level,padding,norm_layer,bilinear)
    def build_encoder(self):
        return _resnet('resnet50', Bottleneck, [3, 4, 6, 3],base_planes=self.base_channels,)


if __name__=="__main__":

    ipt=torch.rand(1,3,512,512)
    res18net=Res18_UNet(n_classes=10,level=4)
    opt=res18net(ipt)
    print(opt.shape)

    # res50net=Res50_UNet(n_classes=10)
    # opt=res50net(ipt)
    # print(opt.shape)