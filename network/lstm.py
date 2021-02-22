import torch
from torch import nn
import torch.nn.functional as F
import os
import pandas as pd
import csv
import numpy as np
class Model(nn.Module):
    def __init__(self, args, n_items, DEVICE):
        super(Model, self).__init__()
        self.args = args
        self.lstm_size = args.lstm_size
        self.embedding_dim = args.embedding_dim
        self.num_layers = args.num_layers
        self.sequence_length = args.sequence_length
        self.DEVICE = DEVICE 
        self.use_glove = True
        self.bi_lstm = False
        #word embedding
        if self.use_glove:
            dict = pd.read_csv(filepath_or_buffer='./glove.6B.50d.txt', header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
            dict_len, embed_size = dict.shape
            dict_len += 1 #unknow word
            unknown_word = np.zeros((1, embed_size))
            dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float)).float() #numpy->torch
            print('the pretrained embeddding has been loaded...')

            self.embedding = nn.Embedding(
                                            num_embeddings=dict_len,
                                            embedding_dim=embed_size,
            ).from_pretrained(dict)
        
        else:
            self.embedding = nn.Embedding(
                                            num_embeddings=n_items,
                                            embedding_dim=self.embedding_dim,
            )
        
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
            bidirectional=self.bi_lstm
        )

        self.dropout = nn.Dropout(0.1)

        if self.bi_lstm:
            self.fc = nn.Linear(2*self.lstm_size, n_items)
        else:
            self.fc = nn.Linear(self.lstm_size, n_items)

    def forward(self, x, prev_state):
        embed = self.embedding(x) #x[256,128], embed[256,128]
        embed = embed.to(self.DEVICE)
        # print(embed.shape)
        # input('check')

        output, state = self.lstm(embed, prev_state)  #output[256,128,128]
        # print(output.shape)
        # input('check')
        logits = self.fc(output)  #output[256,128,3706]

        return logits[:,-1,:], state

    def init_state(self):
        if self.bi_lstm:
            hidden = (torch.zeros(2*self.num_layers, self.sequence_length, self.lstm_size).to(self.DEVICE),
                      torch.zeros(2*self.num_layers, self.sequence_length, self.lstm_size).to(self.DEVICE))
        else:
            hidden = (torch.zeros(self.num_layers, self.sequence_length, self.lstm_size).to(self.DEVICE),
                      torch.zeros(self.num_layers, self.sequence_length, self.lstm_size).to(self.DEVICE))
        return hidden