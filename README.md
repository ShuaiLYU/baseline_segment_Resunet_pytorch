# 一个缺陷分割的baseline（缺陷检测， 语义分割）


# 实现功能
  1. 数据读取
  2.  loss 可视化（tensorboard）
  3.  metrics(缺陷的iou 和 pa)
  4.  训练过程中保存预测结果
  5.  参数结构化管理
  6. 通过将残差网络作为编码器，改进UNet （improving the unet by  using the resnet as the encoder） 
  
  
**visualization:**
### 分割结果：
![1](https://github.com/Wslsdx/baseline_segment_Resunet_pytorch/blob/master/photo/Train_0576.PNG)

![2](https://github.com/Wslsdx/baseline_segment_Resunet_pytorch/blob/master/photo/Train_0588.PNG)

![3](https://github.com/Wslsdx/baseline_segment_Resunet_pytorch/blob/master/photo/Train_0609.PNG)

### 损失曲线：

![step_loss](https://github.com/Wslsdx/baseline_segment_Resunet_pytorch/blob/master/photo/step_loss.png)

### 训练集IOU：

![iou_train](https://github.com/Wslsdx/baseline_segment_Resunet_pytorch/blob/master/photo/iou_train.png)

### 验证集IOU：

![iou_valid](https://github.com/Wslsdx/baseline_segment_Resunet_pytorch/blob/master/photo/iou_valid.png)
 

 
  
