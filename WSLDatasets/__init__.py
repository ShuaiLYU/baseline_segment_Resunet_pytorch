import os
def get_cur_path():
    return os.path.abspath(os.path.dirname(__file__))
from .wsl_dataset import WSLDataset_train,WSLDataset_split