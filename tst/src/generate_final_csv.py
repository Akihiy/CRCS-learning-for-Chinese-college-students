from statistics import correlation
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import json
import numpy as np
import pandas as pd

from src.module import *


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_path = glob.glob('/home/thyme/homework/hello-dian.ai/tst/mchar_test_a/mchar_test_a/*.png')
    test_path.sort()
    test_label = [[1]] * len(test_path)
    test_loader = torch.utils.data.DataLoader(SVHNDataset(test_path, None, 'test'),batch_size=32, shuffle=False, num_workers=10)

    resnet18_model = Detector().to(device)
    resnet18_model.load_state_dict(torch.load('./model.pth'))
    
    resnet50_model = Detector3().to(device)
    resnet50_model.load_state_dict(torch.load('./model3.pth'))

    resnext50_model = Detector4().to(device)
    resnext50_model.load_state_dict(torch.load('./model4.pth'))

    mobilenet_model = Detector2().to(device)
    mobilenet_model.load_state_dict(torch.load('./model2.pth'))

    mobilenet_model.eval()
    resnet18_model.eval()
    resnet50_model.eval()
    resnext50_model.eval()
    test_pred = []
    with torch.no_grad():
        for X in tqdm(test_loader):
            X = X.to(device)
            resnet18_pred = resnet18_model(X)
            resnet50_pred = resnet50_model(X)
            resnext50_pred = resnext50_model(X)
            mobilenet_pred = mobilenet_model(X)

            pred = [ (a +  b + c + d)/4 for a, b, c, d in zip(resnet18_pred, mobilenet_pred, resnet50_pred, resnext50_pred)]

            output = np.concatenate([
                    pred[0].data.cpu().numpy(), 
                    pred[1].data.cpu().numpy(),
                    pred[2].data.cpu().numpy(), 
                    pred[3].data.cpu().numpy()], axis=1)
            test_pred.append(output)

    
    test_pred = np.vstack(test_pred)

    test_pred = np.vstack([
        test_pred[:, :11].argmax(1),
        test_pred[:, 11:22].argmax(1),
        test_pred[:, 22:33].argmax(1),
        test_pred[:, 33:44].argmax(1)]).T

    out = []
    for x in test_pred:
        out.append(''.join(map(str, x[x!=10])))
    df_submit = pd.read_csv('/home/thyme/homework/hello-dian.ai/tst/mchar_sample_submit_A.csv')
    df_submit['file_code'] = out
    df_submit.to_csv('submit_final5.csv', index=None)


if __name__ == "__main__":
    main()