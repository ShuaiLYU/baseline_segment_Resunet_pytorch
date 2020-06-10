from torch.hub import load_state_dict_from_url
from unets.unet_blocks import  ConvBlock
from torch import  nn
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, padding=1,bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=bias)

def conv1x1(in_planes, out_planes, stride=1,bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        if (stride != 1 or inplanes != planes * self.expansion):
            assert  downsample!=None, "downsample can't be None! "
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d # 如果bn层没有自定义，就使用标准的bn层
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # 保存x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)  # downsample调整x的维度，F(x)+x一致才能相加
        out += identity
        out = self.relu(out) # 先相加再激活
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        if (stride != 1 or inplanes != planes * self.expansion):
            assert  downsample!=None, "downsample can't be None! "
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion) # 输入的channel数：planes * self.expansion
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class InputStem(nn.Module):
    """
     A  implementation of "ResNet-C " from paper :  "Bag of Tricks for Image Classification with Convolutional Neural Networks"
     replace the 7 × 7 convolution in the input stem with three conservative 3 × 3 convolutions.
    it can be found on the implementations of other models, such as SENet , PSPNet ,DeepLabV3 , and ShuffleNetV2 .
    不同的是，我们这里把步长全部设置为1，获得与输入相同尺寸的特征图，以适应图像分割任务。
    """
    def __init__(self,in_planes,planes,norm_layer=None):
        super(InputStem,self).__init__()
        self.model=nn.Sequential(
            ConvBlock(in_planes,planes,3,1,norm_layer=norm_layer),
            ConvBlock(planes, planes, 3, 1,norm_layer=norm_layer),
            ConvBlock(planes, planes, 3, 1,norm_layer=norm_layer)
        )
    def forward(self, inputs):
        return  self.model(inputs)


class ResNet(nn.Module):
    def __init__(self, block, layers, norm_layer=None,b_RGB=True,base_planes=32):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        inplanes=3 if b_RGB==True else 1
        self.input_stem=InputStem(inplanes,base_planes,norm_layer)
        self.inplanes = base_planes
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,  base_planes*2//block.expansion, layers[0])
        self.layer2 = self._make_layer(block,  base_planes*4//block.expansion, layers[1], stride=2)
        self.layer3 = self._make_layer(block,  base_planes*8//block.expansion, layers[2], stride=2)
        self.layer4 = self._make_layer(block,  base_planes*16//block.expansion, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        # 生成不同的stage/layer
        # block: block type(basic block/bottle block)
        # blocks: blocks的数量
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            # 需要调整维度
            downsample = nn.Sequential(
            conv1x1(self.inplanes, planes * block.expansion, stride),  # 同时调整spatial(H x W))和channel两个方向
            norm_layer(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer)) # 第一个block单独处理
        self.inplanes = planes * block.expansion  # 记录layerN的channel变化，具体请看ppt resnet表格
        for _ in range(1, blocks): # 从1开始循环，因为第一个模块前面已经单独处理
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)  # 使用Sequential层组合blocks，形成stage。如果layers=[2,3,4]，那么*layers=？

    def forward(self, x):
        #[ b，c, h，w] c=1 or c=3
        x0 = self.input_stem(x)                 #[b,c1,h, w]
        x1 = self.layer1(self.maxpool(x0))    #[b,c2,h//2, w//2]
        x2 = self.layer2(x1)      #[b,c3,h//4, w//4]
        x3 = self.layer3(x2)   #[b,c4,h//8, w//8]
        x4 = self.layer4(x3)  #[b,c5,h//16, w//16]

        return [x0,x1,x2,x3,x4]

def _resnet(arch, block, layers, pretrained=False, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=True)
        # for key,val in state_dict.items():
        #     print(key)
        model.load_state_dict(state_dict, False)

    return model