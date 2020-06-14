# 一个缺陷分割的baseline（缺陷检测， 语义分割）

## 数据集
1. Weakly Supervised Learning for Industrial Optical Inspection（https://hci.iwr.uni-heidelberg.de/node/3616）
  这个链接中有10个不同的数据集，我们对2,4,8,10进行了像素级别标注，数据集和像素标注放在下面的百度网盘链接中：
  链接：https://pan.baidu.com/s/1fkmUTPH0Di8p2C7A8P0fbg  提取码：tpnb  （使用请注明数据来源：本仓库地址）

2. 参考本仓库配置自己的数据集。



## 实现功能
  1. 数据读取
  2.  loss 可视化（tensorboard）
  3.  metrics(缺陷的iou 和 pa)
  4.  训练过程中保存预测结果
  5.  参数结构化管理
  6. 通过将残差网络作为编码器，改进UNet （improving the unet by  using the resnet as the encoder） 
  
  
## visualization:
### 分割结果：
   1.输入图像                              2.像素标注                              3.分割结果
![1](https://github.com/Wslsdx/baseline_segment_Resunet_pytorch/blob/master/photo/Train_0576.PNG)

![2](https://github.com/Wslsdx/baseline_segment_Resunet_pytorch/blob/master/photo/Train_0588.PNG)

![3](https://github.com/Wslsdx/baseline_segment_Resunet_pytorch/blob/master/photo/Train_0609.PNG)

### 损失曲线：

![step_loss](https://github.com/Wslsdx/baseline_segment_Resunet_pytorch/blob/master/photo/step_loss.png)

### 训练集IOU：

![iou_train](https://github.com/Wslsdx/baseline_segment_Resunet_pytorch/blob/master/photo/iou_train.png)

### 验证集IOU：

![iou_valid](https://github.com/Wslsdx/baseline_segment_Resunet_pytorch/blob/master/photo/iou_valid.png)
 

 
  
