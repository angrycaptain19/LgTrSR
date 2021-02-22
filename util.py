import torch
import os
import argparse
import tarfile
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=2)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--sequence_length', type=int, default=64)
    parser.add_argument('--lstm_size', type=int, default=256)
    parser.add_argument('--embedding_dim', type=int, default=50)

    #args = parser.parse_args(args=[])  ##for colab
    args = parser.parse_args()
    return args

def get_device():
    USE_CUDA = torch.cuda.is_available()
    return torch.device("cuda" if USE_CUDA else "cpu")

def set_env(root_path='.', kind='zf'):
    # for train
    if 'SM_CHANNEL_TRAIN' not in os.environ:
        os.environ['SM_CHANNEL_TRAIN'] = '%s/data-%s/' % (root_path, kind)
    if 'SM_MODEL_DIR' not in os.environ:
        os.environ['SM_MODEL_DIR'] = '%s/model/' % root_path

    # for inference
    if 'SM_CHANNEL_EVAL' not in os.environ:
        os.environ['SM_CHANNEL_EVAL'] = '%s/data-%s/' % (root_path, kind)
    if 'SM_CHANNEL_MODEL' not in os.environ:
        os.environ['SM_CHANNEL_MODEL'] = '%s/model/' % root_path
    if 'SM_OUTPUT_DATA_DIR' not in os.environ:
        os.environ['SM_OUTPUT_DATA_DIR'] = '%s/output/' % root_path

    return get_args()

def save_model(model, model_dir, train_loss, val_loss):
    path = os.path.join(model_dir, '{:.4f}-{:.4f}-model.pth'.format(train_loss, val_loss))
    torch.save(model.state_dict(), path)
    #torch.save(model.state_dict(), model_dir)

def load_model(model, model_dir):
    tarpath = os.path.join(model_dir, 'model.tar.gz')
    if os.path.exists(tarpath):
        tar = tarfile.open(tarpath, 'r:gz')
        tar.extractall(path=model_dir)
    model_path = os.path.join(model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path))
    return model


##################################################################################################
#self-attention: no multi-head and FNN
def self_att(output): #b,l,d
    scores = torch.matmul(output, output.permute(0, 2, 1))
    att_scores = F.softmax(scores, dim=2)
    feature = F.relu(att_scores.matmul(output)) #b,l,d

    return feature, att_scores #b,l,d

#subspace transform 
def space_chg(input, weight, bias=False):
    feature = input.matmul(weight)
    if isinstance(bias, torch.nn.parameter.Parameter):
        feature = feature + bias
    feature = torch.tanh(feature)

    return feature

#multi-query attention
def multi_q_att(input, query):
    mq_att = input.matmul(query) 
    mq_att = F.softmax(mq_att, dim=1)
    return mq_att

#w,n,h w,n,1 -->n,h
def mq_func(input, mq_att):
    input = input.unsqueeze(3) #b, l, d -> b, l, d, 1
    mq_att = mq_att.unsqueeze(2) #b, l, q -> b, l, 1, q
    feature = input.mul(mq_att) #b, l, d, q
    feature = torch.sum(feature, 1)
    
    return feature.view(feature.shape[0], -1) #b, d*q

def get_class(filename):
    with open(filename, mode='r') as f:
        data = f.readlines()
        res_dict = {}
        for line in data:
            line = line.strip('\n')
            user = line.split(':')[0]
            num = line.split(':')[1].split(',')
            for item in num:
                if item in res_dict:
                    res_dict[eval(item)] += 1
                else:
                    res_dict[eval(item)] = 1
                    #print('the user of %s has been done!', user)
        print('all of users have been done!')
    item_id = list(res_dict.keys())
    return len(item_id)