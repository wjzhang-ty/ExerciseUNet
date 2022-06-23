import torch
import torch.nn as nn
import torch.nn.functional as F
from net import UNet
from tqdm import tqdm
from torchvision import utils as vutils

# 损失，评分
from metrics import Scoring,get_Dice


classes = 3
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 评估函数
def evaluate_unet(net=False, dataloader=False, batch_size=1, train=True):
    # 1. 如果是测试训练集，自行加载net，并计算网络的macs、params、time
    if not train:
        net = UNet(in_cannel=3,n_classes=classes).to(device=device)
        net.load_state_dict(torch.load('./checkpoints/UNet.pth', map_location=device))

    # 1. 准备变量
    net.eval()
    dataloader_len = len(dataloader)
    score = Scoring(classes) # init metrics

    # 2. 进度条 + 循环验证
    with tqdm(total=dataloader_len*batch_size,  desc='验证', unit='batch', leave=False) as pbar:
        for batch in dataloader:
            # 2.1 准备图片
            image = batch['image'].to(device=device, dtype=torch.float32)
            mask = batch['mask'].to(device=device, dtype=torch.long)

            # 2.2 核心，区别于训练方法，不监督梯度信息
            with torch.no_grad():
                # 2.2.1 预测
                mask_pred = net(image)

                # 2.2.2 评估分数
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), classes).permute(0, 3, 1, 2).float()
                if not train: vutils.save_image(mask_pred, './imgs/'+batch['name'][0]+'.jpg')
                mask = F.one_hot(mask, classes).permute(0, 3, 1, 2).float()
                mask_pred[:,1,...] = mask_pred[:,1,...] + mask_pred[:,2,...] # Od contains OC
                mask[:,1,...] = mask[:,1,...] + mask[:,2,...] # Od contains OC
                dice = score.add(mask_pred,mask) # 计分


                # 2.2.3 更新进度条
                pbar.update(image.shape[0])
                pbar.set_postfix(**{"od": dice[1].item(),"oc": dice[2].item()})

    # 5. 返回分数，调用函数时train=True会返回单位时间、参数、计算量信息
    net.train()
    return score.result()


# 训练函数
def train_unet(train_loader, val_loader, epochs=40, batch_size=20):
    # 1. 准备网络
    net = UNet(in_cannel=3,n_classes=classes).to(device=device)

    # 2. 优化器
    optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-3, weight_decay=1e-8, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)  # 两次Dice不提升，下调学习率

    # 3.损失函数。利用权重帮助交叉熵客服样本不均
    CrossLoss = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.2,0.4,0.4]).to(device=device))
    
    # 5. 开始训练
    soft = nn.Softmax(dim=1)
    for epoch in range(epochs):
        global_step = 0
        net.train()
        with tqdm(total=len(train_loader)*batch_size, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # 5.1 导入图片 load image and mask
                image = batch['image'].to(device=device, dtype=torch.float32)
                mask = batch['mask'].to(device=device, dtype=torch.long)

                # 5.2 预测 pred
                mask_pred = net(image)
                i=soft(mask_pred)
                m=F.one_hot(mask, classes).permute(0, 3, 1, 2).long()

                # 5.4 计算损失 calc loss
                cross = CrossLoss(mask_pred, mask)
                dice = get_Dice(i, m)[0] # get_Dice(i, m) => [1-average(oc,od),od,oc]
                loss = cross + dice

                # 5.5 后向传播
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # 5.6 打印数据
                pbar.update(image.shape[0])
                global_step += 1
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # 5.7 每轮50%进行评估
                if global_step % (len(train_loader) // 2) == 0:
                    score = evaluate_unet(net, val_loader, batch_size)
                    scheduler.step(score['dice'][0])
                    print(score)
                    
        torch.save(net.state_dict(), str('./checkpoints/UNet.pth'.format(epoch + 1)))
