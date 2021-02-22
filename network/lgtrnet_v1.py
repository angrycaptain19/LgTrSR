import torch
from torch import nn
import torch.nn.functional as F
import os
import pandas as pd
import csv
import numpy as np
from util import self_att
import math
class PoMHSABlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, num_head=8):
        super(PoMHSABlock, self).__init__()
        self.num_head = num_head
        self.d_k = in_channels // num_head
        assert in_channels % num_head == 0
        self.query_conv = nn.Conv1d(in_channels, in_channels, stride=stride, kernel_size=1, bias=False)
        self.key_conv = nn.Conv1d(in_channels, in_channels, stride=stride, kernel_size=1, bias=False)
        self.value_conv = nn.Conv1d(in_channels, in_channels, stride=stride, kernel_size=1, bias=False)
        self.last_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        #self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        batch_size, c, width = proj_query.shape

        proj_query = proj_query.permute(0, 2, 1) #[b, w, c]
        proj_key = proj_key.permute(0, 2, 1)#[b, w, c]
        proj_value = proj_value.permute(0, 2, 1)#[b, w, c]

        proj_query = proj_query.view(batch_size, -1, self.num_head, self.d_k).permute(0, 2, 1, 3) #[b, head, w, d_k]
        proj_key = proj_key.view(batch_size, -1, self.num_head, self.d_k).permute(0, 2, 1, 3) #[b, head, w, d_k]
        proj_value = proj_value.view(batch_size, -1, self.num_head, self.d_k).permute(0, 2, 1, 3) #[b, head, w, d_k]

        scores = torch.matmul(proj_query, proj_key.transpose(-2, -1)) / math.sqrt(self.d_k)
        p_attn = self.softmax(scores) #[b, head, w, w] self.sigmoid(scores)
        #print(sum(p_attn[0, 0, 0, :]))
        #output = torch.matmul(p_attn, proj_value).transpose(1, 2).contiguous().view(batch_size, width, -1)#[b, w, c]
        output = self.relu(torch.matmul(p_attn, proj_value)).transpose(1, 2).contiguous().view(batch_size, width, -1)#[b, w, c]
        output = self.last_conv(output.permute(0, 2, 1))

        return output #[b, w, c]

class LgTrBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, num_head=5):
        super(LgTrBlock, self).__init__()

        #residual function
        self.reduce_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.local_branch = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.weight_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, 2, kernel_size=1, bias=False),
            #nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.global_branch = PoMHSABlock(out_channels, out_channels, stride=stride, num_head=num_head)
        
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels, out_channels * LgTrBlock.expansion, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels * LgTrBlock.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * LgTrBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * LgTrBlock.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels * LgTrBlock.expansion)
            )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_reduce = self.reduce_function(x)

        x_local = self.local_branch(x_reduce)
        x_weight = self.weight_branch(x_reduce) #[b, 2, 1, 1]
        x_global = self.global_branch(x_reduce)
        # print(x_local.shape, x_weight.shape, x_global.shape)
        # input('check shape')
        gamma_1 = x_weight[:, 0, :].unsqueeze(1) #[b, 1, 1, 1]
        gamma_2 = x_weight[:, 1, :].unsqueeze(1) #1.0 - gamma_1

        local_feature =  x_local * gamma_1.expand_as(x_local)
        global_feature = x_global * gamma_2.expand_as(x_global)
        output = local_feature + global_feature #torch.cat((local_feature, global_feature), 1) #[b, c, h, w]
        output = self.fusion(output)
        output = nn.ReLU(inplace=True)(output + self.shortcut(x))

        return output.permute(0, 2, 1)
        
class ConvSABlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvSABlock, self).__init__()
        #local-global block
        self.conv_op = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels * LgtrBlock.expansion, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels * LgtrBlock.expansion),
                nn.ReLU(inplace=True)
        )
        self.gamma_wt = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(in_channels, 2, kernel_size=1, bias=False),
                #nn.BatchNorm2d(1),
                nn.Sigmoid()
        )
    def forward(self, x):
        gamma = self.gamma_wt(x.permute(0, 2, 1))
        gamma1 = gamma[:, 0, :].unsqueeze(1)
        gamma2 = gamma[:, 1, :].unsqueeze(1)
       
        #print(gamma1[0, 0, 0], gamma2[0, 0, 0])

        sa_x, scores = self_att(x)
        conv_x = self.conv_op(x.permute(0, 2, 1)).permute(0, 2, 1)

        output = gamma1.expand_as(sa_x) * sa_x + gamma2.expand_as(conv_x) * conv_x
        #output = 0.8 * sa_x + 0.2 * conv_x
        
        output = output + x 
        return output


class Model(nn.Module):
    def __init__(self, args, n_items, DEVICE):
        super(Model, self).__init__()
        self.args = args
        self.lstm_size = args.lstm_size
        self.embedding_dim = args.embedding_dim
        self.num_layers = args.num_layers
        self.sequence_length = args.sequence_length
        self.DEVICE =DEVICE 

        #word embedding
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


        self.embed_lg = LgTrBlock(embed_size, embed_size, 1)

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )

        #self.lstm_lg = LgtrBlock(self.lstm_size, self.lstm_size, 1)

        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.lstm_size, n_items)

    def forward(self, x, prev_state):
        embed = self.embedding(x) #x[256,128], embed[256,128]
        embed = embed.to(self.DEVICE)
        # print(embed.shape)
        # input('check')

        embed = self.embed_lg(embed)

        output, state = self.lstm(embed, prev_state)  #output[256,128,128]
        #output = self_att(output)

        #output = self.lstm_lg(output)
        # print(output.shape)
        # input('check')
        logits = self.fc(output)  #output[256,128,3706]
        

        return logits[:,-1,:], state

    def init_state(self):
        hidden = (torch.zeros(self.num_layers, self.sequence_length, self.lstm_size).to(self.DEVICE),
                torch.zeros(self.num_layers, self.sequence_length, self.lstm_size).to(self.DEVICE))
        return hidden