from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

import sys
sys.path.append("..") # 工作目录不一定是当前目录
from module import *

DEBUG = 0
__all__ = ['Config', 'Trainer']


def getArgParser()->argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Trainer for sentiment classifier')
    parser.add_argument('-d', '--data_path', default='./train.tsv', help='data path for training', metavar='')
    parser.add_argument('-c', '--cpu', action='store_true', help='use cpu to train, default gpu')
    parser.add_argument('-o', '--out', type=str, default='./src_final/model.pt', help='output model path', metavar='')
    parser.add_argument('--pretrained_path', type=str, default='./src_final/model.pt', help='pretrained model path', metavar='')
    parser.add_argument('-p', '--pretrained', action='store_true', help='use pretrained model')
    parser.add_argument('-v', '--validate', action='store_true', help='validate only')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate for training', metavar='')
    parser.add_argument('-e', '--epoch', type=int, default=20, help='training epoch', metavar='')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='training epoch', metavar='')
    parser.add_argument('-f', '--fig_dir', type=str, default='./src_final/images', help='output figures path', metavar='')
    return parser


class Config:
    vocab_size = 128 # ASCII码的个数
    max_length = 283
    embedding_size = 128
    hidden_size = 128
    num_layers = 2
    bidirectional = True
    n_class = 5

    def __init__(self, args:argparse.Namespace) -> None:
        self.train_path = args.data_path
        if not args.cpu and not torch.cuda.is_available():
            raise EnvironmentError('cuda is not available')
        self.device = 'cpu' if args.cpu else 'cuda'
        self.out_model_path = args.out
        self.pretrained_model_path = args.pretrained_path
        self.pretrained = args.pretrained
        self.validate = args.validate
        self.lr = args.learning_rate
        self.n_epoch = args.epoch
        self.batch_size = args.batch_size
        self.fig_dir = args.fig_dir

class Trainer():
    '''
        训练小帮手
    '''
    def __init__(self,n_epoch, train_dataset, validate_dataset, device, model, criterion, optimizer, lr_scheduler):
        self.n_epoch = n_epoch
        self.train_dataset = train_dataset
        self.validate_dataset = validate_dataset
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_acc = []
        self.validate_acc = []
        self.train_loss = []

    def _train_epoch(self, epoch):
        '''return average_accuracy, average_loss'''
        tot_acc, tot_loss, tot_cnt = [0]*3
        pbar = tqdm(self.train_dataset, desc = f'Epoch {epoch}', total=len(self.train_dataset))
        for X, y in pbar:
            X = X.to(self.device)
            y = y.to(self.device)
            out = self.model(X)
            pred = out.argmax(1)
            acc = (pred == y).sum().item()
            loss = self.criterion(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            pbar.set_postfix_str(f'acc:{acc/len(y):>.6f} loss:{loss.item():>.6f}')

            tot_acc += acc
            tot_cnt += len(y)
            tot_loss += loss.item()

        return tot_acc/tot_cnt, tot_loss/len(pbar)

    def _validate(self):
        '''return validate_accuracy'''
        acc = 0
        total = 0
        for X, y in self.validate_dataset:
            X = X.to(self.device)
            y = y.to(self.device)
            out = self.model(X)
            pred = out.argmax(1)
            acc += (pred == y).sum().item()
            total += len(y)
        print(f'validate acc:{acc/total:.6f}')
        return acc/total

    def train(self, path = './src_final/model.pt'):
        for epoch in range(self.n_epoch):
            average_accuracy, average_loss = self._train_epoch(epoch+1)
            with torch.no_grad():
                validate_accuracy = self._validate()
            self.train_acc.append(average_accuracy)
            self.train_loss.append(average_loss)
            self.validate_acc.append(validate_accuracy)
            self.save(path)

    def validate(self):
        self._validate()
    
    def save(self, path = './src_final/model.pt'):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path = './src_final/model.pt'):
        self.model.load_state_dict(torch.load(path))

    def savefig(self, root_path = './src_final/images'):
        plt.plot(range(len(self.train_acc)), self.train_acc, 'r--', label = 'train_acc')
        plt.plot(range(len(self.validate_acc)), self.validate_acc, 'g--', label = 'validate_acc')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(root_path+'/accuracy.png')
        plt.close()
        plt.plot(range(len(self.train_loss)), self.train_loss, 'b--', label = 'train_loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(root_path+'/loss.png')
        plt.close()




def main():
    parser = getArgParser()

    config = Config(parser.parse_args())
    
    data_reader = DataReader()
    train_phrases, train_sentiments,\
    validate_phrases, validate_sentiments = data_reader.read()

    tokenizer = Tokenizer()
    tokenized_train_phrases = tokenizer(train_phrases)
    tokenized_validate_phrases = tokenizer(validate_phrases)

    train_data = MyDataset(tokenized_train_phrases, train_sentiments)
    validate_data = MyDataset(tokenized_validate_phrases, validate_sentiments)

    train_loader = DataLoader(train_data, config.batch_size, shuffle=True, num_workers=2)
    validate_loader = DataLoader(validate_data, config.batch_size, shuffle=False, num_workers=2)

    model = Classifier(config.vocab_size, config.embedding_size, config.hidden_size,\
                       config.num_layers, config.n_class, config.device, config.bidirectional).to(config.device)
    
    optimizer = torch.optim.Adam(model.parameters(), config.lr)

    criterion = torch.nn.CrossEntropyLoss()

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.batch_size, 0.99)

    trainer = Trainer(config.n_epoch, train_loader, validate_loader, config.device,\
                      model, criterion, optimizer, lr_scheduler)

    if config.pretrained:
        trainer.load(config.pretrained_model_path)

    if not config.validate:
        trainer.train(config.out_model_path)
        trainer.save(config.out_model_path)
    else:
        trainer.validate()

    trainer.savefig(config.fig_dir)


if __name__ == '__main__':
    main()