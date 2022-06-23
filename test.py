import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # which gpu
from pathlib import Path

# 数据集
from dataset import drishti_rand_loader

# 引入网络
from process import evaluate_unet

# 检查目录下文件夹是否齐全
def check_dir():
   Path('./checkpoints/').mkdir(parents=True, exist_ok=True) # 存放训练好的参数 save pth
   Path('./imgs/').mkdir(parents=True, exist_ok=True) # save evalute img


# 入口。提示！执行文件的pwd位置即为整个项目的根目录
if __name__ == '__main__':
   check_dir() # 检查是否有日志、参数、可视化文件夹。没有的话创建

   ###############
   ## 准备数据集 ##
   ###############
   batch_size = 1
   trainloader,valloader,testloader = drishti_rand_loader('./data',256,batch_size)


   ###############
   ## 通用超参数 ##
   ###############
   common_args = dict(
      dataloader = testloader, # 测试集
      batch_size = batch_size, # 单次训练使用几张图。越大越快越吃显卡
      train = False,
      ) 

   score = evaluate_unet(**common_args)
   print(score)
