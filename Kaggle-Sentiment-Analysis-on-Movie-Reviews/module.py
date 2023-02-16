import pandas as pd
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

DEBUG = 0
__all__ = ['Tokenizer', 'DataReader', 'MyDataset', 'Classifier']

class Tokenizer():
    '''
        将句子(str)转化成固定长度的向量
    '''
    def __init__(self, padding: int=0, padding_length = 283):
        '''
            padding : 填充的数字
            padding_length : 填充长度，可以填写数字
        '''
        try:
            assert padding_length == 'max_length' or type(padding_length) == int
        except:
            raise TypeError(f'padding_lenght must be str(\'max_length\') or int\nbut got {padding_length} with type of {type(padding_length)}')

        self.padding = padding
        self.padding_length = padding_length

    def __call__(self, phrases:list[str]) ->list[list[int]]:
        '''
            将phrases中的phrase转化成固定长度的向量
        '''
        tokenized_phrases = []
        if self.padding_length == 'max_length':
            lengths = map(len, phrases)
            padding_length = max(lengths)
        else:
            padding_length = self.padding_length
        
        for phrase in phrases:
            tokenized_phrase = self._tokenize(phrase)
            phrase_length = len(phrase)
            if phrase_length > padding_length:
                tokenized_phrases.append(tokenized_phrase[:padding_length])
            else:
                tokenized_phrases.append(tokenized_phrase + [self.padding]*(padding_length-phrase_length))
        
        return tokenized_phrases

    def _tokenize(self, phrase:str) ->list[int]:
        return [ord(c) for c in phrase]


class DataReader:
    def __init__(self, path = './train.tsv', p = 0.1) -> None:
        '''
            read tsv file from path
            divide data into a validating set and a training set with p percentage for validating
        '''
        self.path = path
        self.p = p
    
    def read(self)->tuple[list[str], list[int], list[str], list[int]]:
        '''
            returns:
                train_phrases : list[str]
                train_sentimemts : list[int]
                validate_phrases : list[str]
                validate_sentimemts : list[int]
        '''
        data = pd.read_csv(self.path, sep = '\t')
        validate_data = data.sample(frac = self.p)
        train_data = data.drop(validate_data.index)

        train_phrases = list(train_data['Phrase'])
        train_sentimemts = list(train_data['Sentiment'])
        validate_phrases = list(validate_data['Phrase'])
        validate_sentimemts = list(validate_data['Sentiment'])

        return train_phrases, train_sentimemts,\
               validate_phrases, validate_sentimemts


class MyDataset(Dataset):
    '''
        重写dataset类，继承自Dataset
    '''
    def __init__(self, tokenized_phrases:list[list[int]], sentiments:list[int]):
        super().__init__()
        self.tokenized_phrases = tokenized_phrases
        self.sentiments = sentiments
    
    def __getitem__(self, index:int):
        '''
            returns:
                tokenized_phrase: tensor(lenght, ) 固定长度的句子
                sentiment: int 表示情感
        '''

        return torch.tensor(self.tokenized_phrases[index]).long(), self.sentiments[index]

    def __len__(self):
        '''
            return length of dataset
        '''
        return len(self.sentiments)


class Classifier(nn.Module):
    '''
        RNN classifier
    '''
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, n_class, device, bidirectional=True) -> None:
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_class = n_class
        self.device = device
        self.n_direction = 2 if bidirectional else 1
        
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers, bidirectional = bidirectional, batch_first = True)
        self.fc = nn.Linear(self.n_direction*hidden_size, n_class)
    
    def _init_hidden(self, batch_size:int):
        '''
            return hidden tensor of GRU
        '''
        return torch.zeros((self.n_direction*self.num_layers, batch_size, self.hidden_size)).to(self.device)

    def forward(self, input):
        '''
            input:(B, L)
        '''
        # (B, L) -> (B, L, embedding_size)
        embeded_input = self.embedding(input)
        
        hidden = self._init_hidden(embeded_input.size()[0])

        # hidden : (n_dir*num_layers, B, hidden_size)
        _, hidden = self.gru(embeded_input, hidden)
        if self.n_direction == 2:
            hidden = torch.cat([hidden[-1], hidden[-2]], dim = 1)
        else:
            hidden = hidden[-1]
        return self.fc(F.dropout(hidden, p = 0.2))