from datasets.WSLDatasets import *
from utils.data.base_dataset import  *
if __name__=="__main__":
    cls=9
    # root = "/home/gdut/disk/datasets/wsl_datasets/configs/Class{}".format(cls)
    root=r"G:\数据集\Weakly Supervised Learning for Industrial Optical Inspection\Class{}".format(cls)
    data=WSLDataset_train(root)

    data_split=divide_dataset(data.cls_dict,0.5,0.5)

    config_name="class{}_train5valid5".format(cls)
    config_path=os.path.join(get_cur_path(),config_name)
    write_txt(config_path,data_split)
