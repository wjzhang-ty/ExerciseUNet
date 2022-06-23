import torch
from torch import Tensor

#########################
## calc TP、TN、FN、FP ##
#########################
def get_table(pred: Tensor, target: Tensor):
    TP = (pred*target).sum()
    TN = ((1-pred)*(1-target)).sum()
    FN = ((1-pred)*target).sum()
    FP = (pred*(1-target)).sum()
    return TP,TN,FN,FP


##########
## 原型 ##
##########
def Iou_score(pred: Tensor, target: Tensor):
    TP,TN,FN,FP = get_table(pred,target)
    return TP / (TP + FP + FN + 1e-5) 

def Dice_score(pred: Tensor, target: Tensor):
    # predictions=pred
    # gt_mask =target
    # gt_mask = gt_mask.float()
    # smooth = 1e-5
    # intersect = torch.sum(predictions * gt_mask, dim=(-1,-2))
    # y_sum = torch.sum(gt_mask * gt_mask, dim=(-1,-2))
    # z_sum = torch.sum(predictions * predictions, dim=(-1,-2))
    # batch_dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    # return batch_dice
    TP,TN,FN,FP = get_table(pred,target)
    return (2. * TP) / (2. * TP + FP + FN + 1e-5) 

def Spe_score(pred: Tensor, target: Tensor):
    TP,TN,FN,FP = get_table(pred,target)
    return TN / (TN + FP + 1e-5)

def Sen_score(pred: Tensor, target: Tensor):
    TP,TN,FN,FP = get_table(pred,target)
    return TP / (TP + FN + 1e-5)

def Acc_score(pred: Tensor, target: Tensor):
    TP,TN,FN,FP = get_table(pred,target)
    return (TP + TN) / (TP + TN + FP + FN + 1e-5)

def Pre_score(pred: Tensor, target: Tensor):
    TP,TN,FN,FP = get_table(pred,target)
    return TP / (TP + FP + 1e-5)


#####################################
#    获取可用的值，针对青光眼数据集    #
# [损失（1-均分），视盘分数，视杯分数] #
#####################################

# 获取dice。
def get_Dice(input: Tensor, target: Tensor):
    assert input.size() == target.size()
    od = Dice_score(input[:, 1, ...], target[:, 1, ...])
    oc = Dice_score(input[:, 2, ...], target[:, 2, ...])
    return [(1. - (oc + od) / 2), od, oc]

# 获取IOU，Jaccard
def get_Iou(input: Tensor, target: Tensor):
    assert input.size() == target.size()
    od = Iou_score(input[:, 1, ...], target[:, 1, ...])
    oc = Iou_score(input[:, 2, ...], target[:, 2, ...])
    return [(1. - (oc + od) / 2), od, oc]

# 获取Specificity
def get_Spe(input:Tensor, target:Tensor):
    assert input.size() == target.size()
    od = Spe_score(input[:, 1, ...], target[:, 1, ...])
    oc = Spe_score(input[:, 2, ...], target[:, 2, ...])
    return [(1. - (oc + od) / 2), od, oc]

# 获取sensitivity，Recall
def get_Sen(input:Tensor, target:Tensor):
    assert input.size() == target.size()
    od = Sen_score(input[:, 1, ...], target[:, 1, ...])
    oc = Sen_score(input[:, 2, ...], target[:, 2, ...])
    return [(1. - (oc + od) / 2), od, oc]

# 获取Accuracy
def get_Acc(input:Tensor, target:Tensor):
    assert input.size() == target.size()
    od = Acc_score(input[:, 1, ...], target[:, 1, ...])
    oc = Acc_score(input[:, 2, ...], target[:, 2, ...])
    return [(1. - (oc + od) / 2), od, oc]

# 获取Precision
def get_Pre(input:Tensor, target:Tensor):
    assert input.size() == target.size()
    od = Pre_score(input[:, 1, ...], target[:, 1, ...])
    oc = Pre_score(input[:, 2, ...], target[:, 2, ...])
    return [(1. - (oc + od) / 2), od, oc]


class Scoring():
    def __init__(self,classes) -> None:
        self.number = 0
        self.dice = torch.zeros(classes)
        self.iou = torch.zeros(classes)
        self.spe = torch.zeros(classes)
        self.sen = torch.zeros(classes)
        self.acc = torch.zeros(classes)
        self.pre = torch.zeros(classes)
    
    def add(self,pred,target):
        self.number += 1
        dice = torch.tensor(get_Dice(pred,target))
        self.dice += dice
        self.iou += torch.tensor(get_Iou(pred,target))
        self.spe += torch.tensor(get_Spe(pred,target))
        self.sen += torch.tensor(get_Sen(pred,target))
        self.acc += torch.tensor(get_Acc(pred,target))
        self.pre += torch.tensor(get_Pre(pred,target))
        return dice
    
    def result(self):
        return {
            'dice': [round(i, 5) for i in (self.dice / self.number).numpy().tolist()],
            'iou': [round(i, 5) for i in (self.iou / self.number).numpy().tolist()],
            'spe': [round(i, 5) for i in (self.spe / self.number).numpy().tolist()],
            'sen': [round(i, 5) for i in (self.sen / self.number).numpy().tolist()],
            'acc': [round(i, 5) for i in (self.acc / self.number).numpy().tolist()],
            'pre': [round(i, 5) for i in (self.pre / self.number).numpy().tolist()],
        }
