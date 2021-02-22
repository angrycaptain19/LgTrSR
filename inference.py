import os
import torch
from network.lgtrnet_v1 import Model
from dataset import Dataset
from util import load_model, get_args, get_device, set_env, get_class

@torch.no_grad()
def inference(args, dataloder, model, output_dir, DEVICE):

    with open(output_dir, 'w') as f:
        model = model.to(DEVICE)
        model.eval()
        state_h, state_c = model.init_state()
        state_h = state_h.to(DEVICE)
        state_c = state_h.to(DEVICE)

        i = 0
        for batch, (user_id, sequence) in enumerate(dataloder):
            sequence = sequence[:,1:].to(DEVICE)

            #y_pred, _ = model(sequence, (state_h, state_c)) # when use lgtrnet_v4 model, you need to use this line code since it doesn't have state_h and state_c.
            y_pred, (state_h, state_c) = model(sequence, (state_h, state_c))
            #y = int(torch.argmax(y_pred).data)
            #f.write('%s\n' % y)
            topk = torch.topk(y_pred, 10)[1].data[0].tolist()
            f.write('%s\n' % topk)

            i += 1
            #if i > 3 : break

if __name__ == '__main__':
    args = set_env(kind='zf')   #kind=['ml' or 'zf']
    DEVICE = get_device()

    data_dir = os.environ['SM_CHANNEL_EVAL'] #
    #model_dir = os.environ['SM_CHANNEL_MODEL']
    ##in case only inference
    model_dir = './model/' 
    output_dir = os.environ['SM_OUTPUT_DATA_DIR'] #'./output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    data_path = os.path.join(data_dir, 'test_seq_data.txt')
    output_path = os.path.join(output_dir, 'output.csv')

    dataset = Dataset(data_path, max_len=args.sequence_length)
    #max_item_count = 5382 #for data_ml
    max_item_count = 65427 #21078 #for data_zf
    model = Model(args, max_item_count, DEVICE)

    tr_dl = torch.utils.data.DataLoader(dataset, 1)

    model = load_model(model, model_dir)
    model = model.to(DEVICE)

    inference(args, tr_dl, model, output_path, DEVICE)
    print('finish!')
