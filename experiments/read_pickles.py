import pandas
import pickle
from options import HiDDenConfiguration, TrainingOptions

# 读取.pickle文件,获取训练选项、噪声层配置、隐藏层配置
with open("crop-0.2-0.25/options-and-config.pickle", 'rb') as f:
    train_options = pickle.load(f)
    noise_config = pickle.load(f)
    hidden_config = pickle.load(f)
    print(train_options)
    print(noise_config)
    print(hidden_config)
