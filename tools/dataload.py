import torch
import pandas as pd
from collections import Counter
from util import get_device, get_class
import pickle
import os
# 'train_data.txt'
class DataProcess():
    def __init__(self, file_name):
        self.file_name = file_name
        max_len = 64 #self.get_seq_len(file_name)
        self.max_len = max_len + 1 ## for GT in test set
        #self.max_len = max_len  ## for GT in test set

        self.dataset = self.load_dataset()
        self.dataset = self.dataset.reset_index() #数据清洗时，会将带空值的行删除，此时数据不再是连续的索引，可以使用reset_index()重置索引
        self.dataset.set_index("idx", inplace=True)  ## to use index for comparing timestamp

        self.count = len(self.dataset) #65427
        self.save_pickle()

    def get_seq_len(self, file_name):
        len_list = []
        with open(file_name, mode='r') as f:
            data = f.readlines()
            for line in data:
                line = line.strip('\n')
                seq = line.split(':')[1].split(',')
                len_list.append(len(seq))
            sorted_len_list = sorted(len_list)

        return sorted_len_list[int(0.8*len(sorted_len_list))]

    def load_dataset(self):
        names = ['user_id', 'sequence']
        train_df = pd.read_csv(self.file_name, delimiter=':', names=names)
        train_df['idx'] = range(0, len(train_df))
        train_df['sequence'] = train_df['sequence'].map(self.str_to_list_max_len).apply(lambda x: list(map(int, x)))
        #print(train_df.head())
        return train_df

    def str_to_list_max_len(self, seq):
        _seq = seq.split(',')
        len_seq = len(_seq)
        if len_seq < self.max_len:
            ls_zero = [0 for i in range(self.max_len - len_seq)]
            ls_zero.extend(_seq)
            _seq = ls_zero
        else:
            _seq = _seq[-self.max_len:]

        return _seq

    def save_pickle(self):
        print('the sequence length: ', self.max_len - 1)
        pickle_path = self.file_name.split('.txt')[0] + '.pickle'
        # return input_seq and gt
        #seq = self.dataset.loc[index]['sequence'][:-1]
        #gt = self.dataset.loc[index]['sequence'][-1]
        #print('index=', index, self.dataset.loc[index]['user_id'], len(seq), gt)
        user_list, seq_list = [], []
        for index in range(self.count):
            user_list.append(self.dataset.loc[index]['user_id'])
            seq_list.append(self.dataset.loc[index]['sequence'])
        
        with open(pickle_path,'wb') as g:
            pickle.dump((user_list, seq_list), g)
            print("The data has been saved the format of pickle! The number of users: ", len(user_list))

class Dataset():
    def __init__(self, file_name, max_len=128):
        #DataProcess(file_name)
        self.file_name = file_name.split('.txt')[0] + '.pickle'
        self.ml_file_name = os.path.join('./data-ml', 'train_data.txt').split('.txt')[0] + '.pickle'
        with open(self.file_name, 'rb') as f:
            self.users, self.sequences = pickle.load(f) 
            print("Dataset has been loaded......")

        with open(self.ml_file_name, 'rb') as f:
            self.users_ml, self.sequences_ml = pickle.load(f) 
            print("Dataset has been loaded......")
        
        self.users.extend(self.users_ml)
        self.sequences.extend(self.sequences_ml)

        self.count = len(self.users)
        self.nuniq_items = 65427 #max(get_class(file_name), get_class(os.path.join('./data-ml', 'train_data.txt'))) + 1#get_class(file_name) + 1 #max(get_class(zf_file_name), get_class(ml_file_name)) + 1
        self.uniq_items = [i+1 for i in range(self.nuniq_items)]
        print('the class of dataset: ', self.nuniq_items)
        print('the length of dataset: ', self.count)
        

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        # return input_seq and gt
        return (torch.tensor(self.users[index]), \
                torch.LongTensor(self.sequences[index]))

if __name__ == '__main__' :
    data_ml = DataProcess('./data-ml/train_data.txt') #224
    #data_zf = DataProcess('./data-zf/train_data.txt') #64
    # dataset = Dataset('./data-zf/train_data.pickle')
    # #print(dataset.count)
    # # for i in range(10):
    # #     print(i, dataset[i])
