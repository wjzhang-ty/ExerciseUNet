import torch
import numpy as np
from PIL import Image
from pathlib import Path
from os import listdir
from os.path import splitext
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A

#数据增强
p1=0.4
data_enhance=A.Compose([
    A.RandomBrightnessContrast(p=1),
    A.HorizontalFlip(p=p1), # 竖直翻转
    A.RandomRotate90(p=p1),
    A.VerticalFlip(p=p1), # 水平翻转
    A.ShiftScaleRotate(p=p1), # 旋转+缩放+平移
    A.GaussNoise(p=p1), # 高斯噪声
    ])


class REFUGE(Dataset):
    """
    images_dir：image path。
    masks_dir：mask path。
    extend：Data enhancement factor
    """

    def __init__(self, images_dir, masks_dir, extend=1, imgsize=256) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.extend = extend
        self.imgsize = imgsize

        self.ids = [
            splitext(file)[0]
            for file in listdir(images_dir)
            if not file.startswith(".")
        ]
        
    def __len__(self):
        return len(self.ids)*abs(self.extend)

    def __getitem__(self, idx):
        # 组装path
        index=idx//abs(self.extend)
        name = self.ids[index]

        # 名称
        mask_file = list(self.masks_dir.glob(name + ".*"))
        img_file = list(self.images_dir.glob(name + ".*"))

        # 处理图片
        mask = self.preprocess(filename=mask_file[0], imgsize=self.imgsize, is_mask=True)
        img = self.preprocess(filename=img_file[0], imgsize=self.imgsize)

        # 第一张原图，其余albumentations增强，参考上方extend注释
        if idx%self.extend > 0:
            enhance = data_enhance(image=img,mask=mask)
            img,mask = enhance['image'],enhance['mask']
        
        # 返回数据
        return {
            "image": torch.as_tensor((img/255).transpose((2, 0, 1)).copy()).float().contiguous(),# /255归一化，转BRG，转float
            "mask": torch.as_tensor(mask.copy()).long().contiguous(),
            "name": name
        }

        
    @staticmethod
    def preprocess(filename, imgsize=256, is_mask=False):
        """
        图片处理
        filename：图片路径+名字
        is_mask；是否为标注图片 Bollean
        """
        # 打开文件
        img = Image.open(filename)

        # 宽高对齐，保证上采样、跨层链接维度对齐
        if imgsize!=img.height or imgsize != img.width:
            img = img.resize((imgsize, imgsize), resample=Image.NEAREST if is_mask else Image.BICUBIC)

        # 转格式 to uint8 for albumentations
        img = np.asarray(img, dtype=np.uint8)

        # mask => background：0、 od：1、 oc：2
        if is_mask:
            img = np.zeros((imgsize,imgsize)) + (img[...,1]/255) + (img[...,0]/255)

        return img

# Get dataloader
def drishti_rand_loader(path,img_size=256,batch_size=4,extend=[1,1,1]):
    train_set = REFUGE(path+'/train/', path+'/mask', imgsize=img_size, extend=extend[0]) # 训练集，训练集扩大4倍（数据增强）
    n_val = int(len(train_set)*0.2)
    train_set, val_set = random_split(train_set, [len(train_set) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    test_set = REFUGE(path+'/test/', path+'/mask', imgsize=img_size, extend=extend[2])

    loader_args = dict( num_workers=3, batch_size=batch_size, pin_memory=False)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_args)
    return train_loader,val_loader,test_loader

