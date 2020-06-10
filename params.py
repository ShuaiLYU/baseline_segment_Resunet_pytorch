from utils.param import  Param


PARAM=Param()

PARAM.dataset_train = Param(
    root="./dataset/Class2",
                      data_config="class2_train5valid5",
                      phase="training",
                      return_numpy=True)
PARAM.dataloader_train = Param(batch_size=32,
                         shuffle=True,
                         num_workers=8,
                         drop_last=False)

PARAM.dataset_valid = Param(
    root="./dataset/Class2",
                      data_config="class2_train5valid5",
                      phase="validation",
                      return_numpy=True)

PARAM.dataloader_valid = Param(batch_size=32,
                         shuffle=False,
                         num_workers=8,
                         drop_last=False)

PARAM.model = Param(n_classes=1, #类别数，二分类所以为1
                    level=4,
                    b_RGB=False,
                    base_channels=32
                    ) # 0-1值

PARAM.Adam = Param(
    lr=0.001,
    weight_decay=0.001,
    betas=(0.9, 0.999))

PARAM.train=Param(
    epoch=100,
    valid_frequency=1, #几个epoch 验证一次
    save_frequency=10,  #几个epoch 保存一次模型
    model_dir="./save/pth/",
    log_dir="./save/log/",
)
PARAM.visualer=Param(
    save_dir="./save/visualizaiton/",     #保存路径
    visual_frequency=3,  #几个epoch保存一次
    visual_batchs = 5,   #保存的记录
)
