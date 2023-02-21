from decimal import Decimal
from random import shuffle
from statistics import correlation
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import json

from src.module import *

def main():
    train_path = glob.glob('/home/thyme/homework/hello-dian.ai/tst/mchar_train/mchar_train/*.png')
    train_path.sort()
    train_json = json.load(open('/home/thyme/homework/hello-dian.ai/tst/mchar_train.json'))
    train_label = [train_json[x]['label'] for x in train_json]


    val_path = glob.glob('/home/thyme/homework/hello-dian.ai/tst/mchar_val/mchar_val/*.png')
    val_path.sort()
    val_json = json.load(open('/home/thyme/homework/hello-dian.ai/tst/mchar_val.json'))
    val_label = [val_json[x]['label'] for x in val_json]

    device = "cuda" if torch.cuda.is_available() else "gpu"
    batch_size = 32         
    n_epoch = 60


    train_loader = torch.utils.data.DataLoader(SVHNDataset(train_path, train_label, 'train'), batch_size = batch_size,   shuffle = True)
    val_loader = torch.utils.data.DataLoader(SVHNDataset(val_path, val_label, 'validate'), batch_size = batch_size,   shuffle = True)

    model = Detector4().to(device)
    model.load_state_dict(torch.load('./model4.pth'))

    criterion = LabelSmoothEntropy()
    # criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00075)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2, eta_min=10e-4)

    for epoch in range(n_epoch):
        model.train()
        for i, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = criterion(pred[0], y[:, 0]) + criterion(pred[1], y[:, 1]) +criterion(pred[2], y[:, 2]) + criterion(pred[3], y[:, 3])

            temp = torch.stack([pred[0].argmax(1) == y[:, 0],
                                pred[1].argmax(1) == y[:, 1],
                                pred[2].argmax(1) == y[:, 2],
                                pred[3].argmax(1) == y[:, 3],], dim=1)

            corrects = torch.all(temp, dim=1).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            print(f'\rEpoch [{epoch+1}/{n_epoch}] {i+1:>03}/{len(train_loader)} Loss: {loss.item():.6f} Acc: {corrects/(batch_size): .6f} lr: {lr_scheduler.get_last_lr()[-1]: .6f}', end = '')

        print('\n')
        model.eval()
        with torch.no_grad():
            c = 0
            for X, y in tqdm(val_loader):
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                temp = torch.stack([pred[0].argmax(1) == y[:, 0],
                                    pred[1].argmax(1) == y[:, 1],
                                    pred[2].argmax(1) == y[:, 2],
                                    pred[3].argmax(1) == y[:, 3],], dim=1)

                c += torch.all(temp, dim=1).sum().item()
            print(f'test acc: {c/10000 : .4f}')
        
        torch.save(model.state_dict(), "./model4.pth")

    




if __name__ == "__main__":
    main()