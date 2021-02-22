import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

from network.lgtrnet_v1 import Model

#from dataset import Dataset
from tools.dataload import Dataset
from inference import inference

from util import save_model, load_model, set_env, get_device
import shutil

def train_model(args, data_loaders, data_lengths, DEVICE, model_dir):
    model = Model(args, data_lengths['nuniq_items'], DEVICE)

    # if torch.cuda.is_available():
    #     if torch.cuda.device_count() > 1:
    #         model = nn.DataParallel(model)

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_loss = float('inf')
    for epoch in range(args.max_epochs):
        print('Epoch {}/{}'.format(epoch+1, args.max_epochs),)
        epoch_loss = {}
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            for batch, (user_id, sequence) in enumerate(data_loaders[phase]):
                if phase == 'train':
                    optimizer.zero_grad()

                state_h = model.init_state()

                x = sequence[:,:-1].to(DEVICE)#[batch_size, length]
                y = sequence[:,-1].to(DEVICE)
                # print(y.shape)
                # input('check')

                state_h = tuple(each.data for each in state_h)

                y_pred, state_h = model(x, state_h) #[batch_size, n_items]

                loss = criterion(y_pred.float(), y.squeeze())
                #input('check')

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data

            epoch_loss[phase] = running_loss / data_lengths[phase]
            #input('check')
            if phase == 'val':
                print('Train loss: {:.4f} Val loss: {:.4f}'.format(epoch_loss['train'], epoch_loss['val']))
                if epoch_loss['train'] < best_loss:
                    best_loss = epoch_loss['train']
                    save_model(model, model_dir, best_loss, epoch_loss['val'])
    return model

def test_inference(args, DEVICE):
    data_dir = os.environ['SM_CHANNEL_EVAL']
    output_dir = os.environ['SM_OUTPUT_DATA_DIR']

    data_path = os.path.join(data_dir, 'train_data.txt')
    #ml_data_path = os.path.join('./data-ml', 'train_data.txt')
    output_path = os.path.join(output_dir, 'output.txt')

    dataset = Dataset(data_path, max_len=args.sequence_length)
    tr_dl = torch.utils.data.DataLoader(dataset, 1)

    model = Model(args, dataset.nuniq_items, DEVICE)
    #model_dir = 'output/model.pth'
    print("model_dir=", model_dir)

    model = load_model(model, model_dir)
    model = model.to(DEVICE)
    #print('check')
    inference(args, tr_dl, model, output_path, DEVICE)

if __name__ == '__main__':
    args = set_env(kind='zf')   #kind=['ml' or 'zf']
    data_dir = os.environ['SM_CHANNEL_TRAIN']
    model_dir = os.environ['SM_MODEL_DIR']
    output_dir = os.environ['SM_OUTPUT_DATA_DIR']
    DEVICE = get_device()

    import os
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
        
    data_path = os.path.join(data_dir, 'train_data.txt')
    #ml_data_path = os.path.join('./data-ml', 'train_data.txt')

    dataset = Dataset(data_path, max_len=args.sequence_length)
    lengths = [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)]
    print('the split of dataset: ', lengths)
    # input('check')

    tr_dt, vl_dt = torch.utils.data.dataset.random_split(dataset, lengths)
    tr_dl = torch.utils.data.DataLoader(tr_dt, args.batch_size)
    vl_dl = torch.utils.data.DataLoader(vl_dt, args.batch_size)
    
    data_loaders = {"train": tr_dl, "val": vl_dl}
    data_lengths = {"train": len(tr_dl), "val": len(vl_dl), "nuniq_items": dataset.nuniq_items}
    print(DEVICE)
    print(data_lengths)
    #input('check')

    model = train_model(args, data_loaders, data_lengths, DEVICE, model_dir)
    #save_model(model, model_dir)
    
    print('Now it starts to test my trained model...')
    #test_inference(args, DEVICE)
