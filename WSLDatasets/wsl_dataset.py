from utils.data.base_dataset import  *
from utils.cv_utiles import cv_imread
from utils.data import  my_transforms
from utils.param import Param
import utils
from utils import  plt_utils
from torchvision import transforms
import cv2
import os
import numpy as np
from random import shuffle
from torch.utils.data import DataLoader

"""
##  dataloader 将数据打包为batch 
1. 自己写也是可以，锻炼下
2， 数据读取是在cpu上运行的， 训练在GPU上运行的 （木桶定理）
3.  官方提供的接口：有多线程的功能。
"""

Trans = {
    "numpy":my_transforms.ComposeJoint([
        [transforms.ToTensor(), transforms.ToTensor()],  #ToTensor 转化为【0-1】
        # [transforms.Normalize(*mean_std), None],
        my_transforms.Tensor2Numpy(),
        [my_transforms.ToFloatNumpy(), my_transforms.ToMask(0.2)]
    ]),
    "torch":my_transforms.ComposeJoint([
        [transforms.ToTensor(), transforms.ToTensor()],
        # [transforms.Normalize(*mean_std), None],
        [None, my_transforms.ToMask(0.2)]
    ])
}


class WSLDataset(BDataset):
    def __init__(self,root,transform_PIL=None,return_numpy=True):
        self.root=root


        #把所有图片地址导入内存中
        self.list_dir(self.root,use_absPath=False)



        #把数据打包为样本： （ 图片地址， 像素标签的地址， 类别）
        self.make_dataset()


        # if transform_PIL is None:
        #     self.transform_PIL = my_transforms.ComposeJoint([
        #      my_transforms.ToPIL(),
        #      my_transforms.GroupRandomHorizontalFlip(),
        #      my_transforms.GroupRandomVerticalFlip(),
        #      my_transforms.GroupResize(size=(512,512)),
        #      ])
        # else:
        self.transform_PIL=transform_PIL

        self.transform_array = Trans["numpy"]  if return_numpy  else Trans["torch"]

    def get_label(self,path):
        img=cv_imread(path)
        if img is None:
            raise Exception("read image wrong")
        mask=np.where(img>0,1,0).astype(np.uint8)
        target = 1 if np.sum(mask) > 10 else 0
        return target
    def make_dataset(self):
        print("生成数据集......")
        samples = []
        # 添加样本列表
        for img in sorted(self.imgs):
            label_pixel=os.path.join(os.path.dirname(img),"Label",os.path.basename(img).replace(".","_label."))
            #print(label_pixel)
            # 过滤没有语义标签的图片
            label= 0 if label_pixel not in self.imgs_pixel else 1
            label_pixel=label_pixel if label_pixel in self.imgs_pixel else None
            # 过滤不符合类别的图片
            item = (img, label_pixel, label)
            samples.append(item)
        print("	总样本数：{}".format(len(samples)))
        cls_dict=self.split_dataset_by_cls(samples,2)
        self.samples, self.cls_dict= samples, cls_dict
        for sample in self.samples:
            print(sample)

    def list_dir(self,root,use_absPath=False, func=None):
        def is_train(path):
            return True  if "Train" in path else False
        def is_test(path):
            return True if "Test" in path else False
        def is_img(path):
            return True if path.endswith(".PNG") and "_label.PNG" not in path else False
        def is_imgPixel(path):
            return True if path.endswith("_label.PNG") else False
        self.imgs=super(WSLDataset,self).list_dir(root,use_absPath,is_img)
        # for img in self.imgs : print( img)
        self.imgs_pixel=super(WSLDataset, self).list_dir(root, use_absPath, is_imgPixel)
        # for img in self.imgs_pixel : print( img)
        self.imgs_train=[img_path for img_path in self.imgs if is_train(img_path)]
        self.imgs_test = [img_path for img_path in self.imgs if is_test(img_path)]
        self.imgs_pixel_train=[img_path for img_path in self.imgs_pixel if is_train(img_path)]
        self.imgs_pixel_test = [img_path for img_path in self.imgs_pixel if is_test(img_path)]

    def gen_a_sample(self,sample):
        """
        过程： 1.  数据增强  3. 读图片  2. 把数据转化为tensor
        :param sample:
        :return:
        """

        file_basename_image, file_basename_label, label =sample
        # 1. 读图
        image_path = os.path.join(self.root, file_basename_image)
        image=cv_imread(image_path,-1)
        #print(image.shape)
        image = np.array(image).astype(np.uint8)
        if file_basename_label is not None:
            label_path = os.path.join(self.root, file_basename_label)
            pixel_label = cv_imread(label_path, -1)
            label_pixel = np.array(pixel_label).astype(np.uint8)
        else:
            label_pixel=np.zeros_like(image).astype(np.uint8)
        # 2.  数据增强
        if self.transform_PIL is not None:
             image,label_pixel=self.transform_PIL([image,label_pixel])

        # 3. 数据格式转化
       # utils.plt_utils.plt_show_imgs([image, label_pixel])
        image, label_pixel = self.transform_array([image, label_pixel])
       # utils.plt_utils.plt_show_imgs([image.squeeze(), label_pixel.squeeze()])
        return image, label_pixel, int(label), file_basename_image


class WSLDataset_train(WSLDataset):

    def make_dataset(self):
        imgs=self.imgs_train
        imgs_pixel=self.imgs_pixel_train
        #print("生成数据集......")
        samples = []
        # 添加样本列表
        for img in sorted(imgs):
            label_pixel=os.path.join(os.path.dirname(img),"Label",os.path.basename(img).replace(".","_label."))
            # 过滤没有语义标签的图片
            label= 0 if label_pixel not in imgs_pixel else 1
            label_pixel=label_pixel if label_pixel in imgs_pixel else None
            # 过滤不符合类别的图片
            item = (img, label_pixel, label)
            samples.append(item)
        print("	总样本数：{}".format(len(samples)))
        cls_dict=self.split_dataset_by_cls(samples,2)
        self.samples, self.cls_dict= samples, cls_dict

class WSLDataset_test(WSLDataset):
    def make_dataset(self):
        imgs=self.imgs_test
        imgs_pixel=self.imgs_pixel_test
        print("生成数据集......")
        samples = []
        # 添加样本列表
        for img in sorted(imgs):
            label_pixel=os.path.join(os.path.dirname(img),"Label",os.path.basename(img).replace(".","_label."))
            # 过滤没有语义标签的图片
            label= 0 if label_pixel not in imgs_pixel else 1
            label_pixel=label_pixel if label_pixel in imgs_pixel else None
            # 过滤不符合类别的图片
            item = (img, label_pixel, label)
            samples.append(item)
        print("	总样本数：{}".format(len(samples)))
        cls_dict=self.split_dataset_by_cls(samples,2)
        self.samples, self.cls_dict= samples, cls_dict


class WSLDataset_split(WSLDataset):
    def __init__(self,root,data_config,phase="training",transform_PIL=None,return_numpy=True,**kwargs):
        self.root=root
        self.data_config=data_config
        self.phase=phase
        self.transform_PIL=transform_PIL
        self.transform_array = Trans["numpy"] if return_numpy else Trans["torch"]
        self.make_dataset()

    def make_dataset(self):
        from WSLDatasets import get_cur_path
        config_path=os.path.join(get_cur_path(),"configs",self.data_config)
        self.samples=read_txt(config_path,self.phase)
        for sample in self.samples: sample[2]=int(sample[2])
        # print("	总样本数：{}".format(len(self.samples)))
        self.cls_dict=self.split_dataset_by_cls(self.samples,loc=2)

        #####从新排列样本
        sample=[]
        max_len=max([ len(self.cls_dict[i]) for i  in range(len(self.cls_dict)) ])
        for i  in range(max_len):
            for j in range(len(self.cls_dict)):
                if i<len(self.cls_dict[j]):
                    sample.append(self.cls_dict[j][i])
        self.samples=sample



if  __name__=="__main__":
   # root="G:\数据集\KolektorSDD"

    root=r"G:\数据集\Weakly Supervised Learning for Industrial Optical Inspection\Class1"
    # root="/home/gdut/disk/datasets/wsl_datasets/Class1"
    data=WSLDataset(root)


    valid_data=WSLDataset_split(root,data_config="class1_train5valid5",phase="validation",transform_PIL=None,return_numpy=True)


    #[batch, channel ,height, width]
    # for  idx in range(len(train_data)):
    #     img,pixel_label, label, img_name=data[idx]
    #     print(img.shape)
    #     print(pixel_label.shape)
    #     plt_utils.plt_show_imgs([img.squeeze(),pixel_label.squeeze()])
    #     print(label)
    #     print(img_name)
    param=Param()

    param.dataset=Param(root=root, data_config="class1_train5valid5", phase="training", transform_PIL=None,
                                  return_numpy=True)
    param.dataloader=Param(batch_size=64,shuffle=True,num_workers=8,drop_last=False)



    train_data = WSLDataset_split(**(param.dataset))
    print(len(train_data))
    loader=DataLoader(train_data,**(param.dataloader))

    for batch in loader:
        img, pixel_label, label, img_name = batch
        print(pixel_label.shape)


