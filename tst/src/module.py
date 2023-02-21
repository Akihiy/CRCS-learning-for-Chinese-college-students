from decimal import Decimal
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision
from PIL import Image
from torchvision.models import resnet50, resnet18, mobilenet_v2, resnext50_32x4d
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import json

train_path = glob.glob('/home/thyme/homework/hello-dian.ai/tst/mchar_train/mchar_train/*.png')
train_path.sort()
train_json = json.load(open('/home/thyme/homework/hello-dian.ai/tst/mchar_train.json'))
train_label = [train_json[x]['label'] for x in train_json]


val_path = glob.glob('/home/thyme/homework/hello-dian.ai/tst/mchar_val/mchar_val/*.png')
val_path.sort()
val_json = json.load(open('/home/thyme/homework/hello-dian.ai/tst/mchar_val.json'))
val_label = [val_json[x]['label'] for x in val_json]


class SVHNDataset(Dataset):
    def __init__(self, img_path: list[str] = train_path, img_label: list[str] = train_label, mode = 'train'):
        assert mode in ['train', 'validate', 'test']
        
        super().__init__()
        self.img_path = img_path
        self.img_label = img_label 
        self.mode = mode

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.mode != 'test':
            label = self.img_label[index]
            if len(label) == 6:
                label = label[:-1]
            label = torch.tensor(label+(5-len(label))*[10]).long()

        if self.mode == 'validate':
            return transforms.Compose([
                transforms.Resize((128, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(img), label
        elif self.mode == 'test':
            return transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop((128, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(img)

        tf = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop((128, 224)),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.RandomGrayscale(0.1),
            transforms.RandomAffine(15,translate=(0.05, 0.1), shear=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return tf(img), label

    def __len__(self):
        return len(self.img_path)

class Detector(nn.Module):

  def __init__(self, class_num=11, pretrained = False):
    super(Detector, self).__init__()

    if pretrained == False:
        weights = None
    else:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1

    self.net = resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    self.net.fc = nn.Identity()

    self.bn = nn.BatchNorm1d(2048)
    self.fc1 = nn.Linear(512, class_num)
    self.fc2 = nn.Linear(512, class_num)
    self.fc3 = nn.Linear(512, class_num)
    self.fc4 = nn.Linear(512, class_num)

  def forward(self, img):
    features = self.net(img).squeeze()

    fc1 = self.fc1(features)
    fc2 = self.fc2(features)
    fc3 = self.fc3(features)
    fc4 = self.fc4(features)

    return fc1, fc2, fc3, fc4

class Detector2(nn.Module):

  def __init__(self, class_num=11, pretrained = False):
    super(Detector2, self).__init__()

    if pretrained == False:
        weights = None
    else:
        weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1

    self.net = mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1).features
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.bn = nn.BatchNorm1d(1280)
    self.fc1 = nn.Linear(1280, class_num)
    self.fc2 = nn.Linear(1280, class_num)
    self.fc3 = nn.Linear(1280, class_num)
    self.fc4 = nn.Linear(1280, class_num)

  def forward(self, img):
    features = self.avgpool(self.net(img)).view(-1, 1280)
    features = self.bn(features)

    fc1 = self.fc1(features)
    fc2 = self.fc2(features)
    fc3 = self.fc3(features)
    fc4 = self.fc4(features)

    return fc1, fc2, fc3, fc4

class Detector3(nn.Module):

  def __init__(self, class_num=11, pretrained = False):
    super(Detector3, self).__init__()

    if pretrained == False:
        weights = None
    else:
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1

    self.net = resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    self.net.fc = nn.Identity()

    self.bn = nn.BatchNorm1d(2048)
    self.fc1 = nn.Linear(2048, class_num)
    self.fc2 = nn.Linear(2048, class_num)
    self.fc3 = nn.Linear(2048, class_num)
    self.fc4 = nn.Linear(2048, class_num)

  def forward(self, img):
    features = self.net(img).squeeze()

    fc1 = self.fc1(features)
    fc2 = self.fc2(features)
    fc3 = self.fc3(features)
    fc4 = self.fc4(features)

    return fc1, fc2, fc3, fc4


class Detector4(nn.Module):

  def __init__(self, class_num=11, pretrained = False):
    super(Detector4, self).__init__()

    if pretrained == False:
        weights = None
    else:
        weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT

    self.net = resnext50_32x4d(weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT)
    self.net.fc = nn.Identity()

    self.bn = nn.BatchNorm1d(2048)
    self.fc1 = nn.Linear(2048, class_num)
    self.fc2 = nn.Linear(2048, class_num)
    self.fc3 = nn.Linear(2048, class_num)
    self.fc4 = nn.Linear(2048, class_num)

  def forward(self, img):
    features = self.net(img).squeeze()

    fc1 = self.fc1(features)
    fc2 = self.fc2(features)
    fc3 = self.fc3(features)
    fc4 = self.fc4(features)

    return fc1, fc2, fc3, fc4



class LabelSmoothEntropy(nn.Module):
  def __init__(self, smooth=0.1, class_weights=None, size_average='mean'):
    super(LabelSmoothEntropy, self).__init__()
    self.size_average = size_average
    self.smooth = smooth
    self.class_weights = class_weights

  def forward(self, preds, targets):

    lb_pos, lb_neg = 1 - self.smooth, self.smooth / (preds.shape[0] - 1)
    smoothed_lb = torch.zeros_like(preds).fill_(lb_neg).scatter_(1, targets[:, None], lb_pos)
    log_soft = F.log_softmax(preds, dim=1)

    if self.class_weights is not None:
      loss = -log_soft * smoothed_lb * self.class_weights[None, :]
    else:
      loss = -log_soft * smoothed_lb

    loss = loss.sum(1)
    if self.size_average == 'mean':
      return loss.mean()

    elif self.size_average == 'sum':
      return loss.sum()

d = Detector()
print(d)