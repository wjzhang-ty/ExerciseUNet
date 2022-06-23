import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # which gpu
from pathlib import Path

# 数据集
from dataset import drishti_rand_loader

# 引入网络
from process import train_unet


# 检查目录下文件夹是否齐全
def check_dir():
   Path('./checkpoints/').mkdir(parents=True, exist_ok=True) # 存放训练好的参数
   Path('./imgs/').mkdir(parents=True, exist_ok=True) # save evalute img


# 入口。提示！执行文件的pwd位置即为整个项目的根目录
if __name__ == '__main__':
   check_dir() # 检查是否有日志、参数、可视化文件夹。没有的话创建

   #####################
   ### dataset数据集 ###
   #####################
   batch_size = 16
   train_loader,val_loader,_ = drishti_rand_loader('./data',256,batch_size,extend=[5, 1, 1])


   ###############
   ## 通用超参数 ##
   ###############
   common_args = dict(
      train_loader=train_loader, # 训练集
      epochs=50, # 全部数据集反复训练多少次
      batch_size=batch_size, # 单次训练使用几张图。越大越快越吃显卡
      val_loader=val_loader, # 验证集
      ) 


   train_unet(**common_args)
