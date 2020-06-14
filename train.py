import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from unets.resunet import  Res18_UNet
from WSLDatasets.wsl_dataset import  WSLDataset_split
from torch.utils.tensorboard import  SummaryWriter
from torch.utils.data import WeightedRandomSampler
from params import  PARAM
from funcs import  iter_on_a_epoch
from  utils.data import  my_transforms
import utils
from visual import  Visualer
import  os
###定义数据
transform={"train": my_transforms.ComposeJoint([
     my_transforms.ToPIL(),
     my_transforms.GroupRandomHorizontalFlip(),
     my_transforms.GroupRandomVerticalFlip(),
     my_transforms.GroupResize(size=(256,256)),
 ]),
"valid": my_transforms.ComposeJoint([
     my_transforms.ToPIL(),
     my_transforms.GroupResize(size=(256,256)),
 ])
}

train_data = WSLDataset_split(transform_PIL=transform["train"],**(PARAM.dataset_train))
#使用WeightedRandomSampler解决训练样本不平衡的问题
weights=[ 1 if data[2]==0 else  6 for data in train_data ] #正负样本采样6:1
sampler=WeightedRandomSampler(weights=weights,num_samples=len(train_data),replacement=True)
PARAM.dataloader_train.shuffle=False #自定义sampler和DataLoader的shuffle参数互斥
train_loader = DataLoader(train_data,sampler=sampler, **(PARAM.dataloader_train))
valid_data = WSLDataset_split(transform_PIL=transform["valid"],**(PARAM.dataset_valid))
valid_loader = DataLoader(train_data, **(PARAM.dataloader_valid))
DATA_LOADERS={ "train":train_loader, "valid":valid_loader }

#设备
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
print(DEVICE)
#定义模型
MODEL = Res18_UNet(**(PARAM.model))
MODEL.to(DEVICE)
#定义二分类交叉熵损失函数：
LOSSES={"supervise":nn.BCELoss()}
#使用Adam优化器
parameters = filter(lambda p: p.requires_grad, MODEL.parameters())
OPTIM=torch.optim.Adam(params=parameters,**(PARAM.Adam))

#定义一个metrics评价网络的性能
METRICS=utils.segment_metrics.SegmentationMetric(numClass=2)
#定义一个writer保存输出结果
WRITER=SummaryWriter(log_dir=PARAM.train.log_dir)

#定义一个对象保存图像
VISUALER=Visualer(**(PARAM.visualer))
TRAIN_PARAM=PARAM.train

def train(train_param,data_loaders,model,losses,optim,metrics,writer,visualer,device):
    for epo  in  range(1,train_param.epoch+1):
        # print("epoch:{}......".format(epo))
        iter_on_a_epoch(epo,"train",data_loaders["train"],model,losses,optim,metrics,writer,visualer,device)
        #验证
        with_valid=True if epo%train_param.valid_frequency==0 else False
        if  with_valid:
            iter_on_a_epoch(epo,"valid",data_loaders["valid"],model,losses,optim,metrics,writer,visualer,device)
        #保存模型
        if epo%train_param.save_frequency==0:
            if not os.path.exists(train_param.model_dir):
                os.makedirs(train_param.model_dir)
            model_path=os.path.join(train_param.model_dir,"epoch-{}.pth".format(epo))
            torch.save(model.state_dict(),model_path)

if __name__=="__main__":
    train(TRAIN_PARAM,DATA_LOADERS,MODEL,LOSSES,OPTIM,METRICS,WRITER,VISUALER,DEVICE)



