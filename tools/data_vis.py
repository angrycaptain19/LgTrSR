import numpy as np
import matplotlib.pyplot as plt
import heapq

def list_max_len(seq, max_len):
    len_seq = len(seq)
    if len_seq >= max_len:
        return seq[-max_len:]

    ls_zero = [0 for i in range(max_len - len_seq)]
    ls_zero.extend(seq)
    return ls_zero

def user_item_inter(user_id, filename, max_len=64):
    with open(filename, mode='r') as f:
        seq = f.readlines()[user_id]
        print(seq)
        seq = seq.strip('\n')

        user_id = eval(seq.split(':')[0])
        items_id = [eval(item) for item in seq.split(':')[1].split(',')]
        items_id = list_max_len(items_id, max_len + 1)

    x = np.arange(max_len + 1) + 1
    print(x)
    y = items_id
    return x, y, user_id

def item_stat(filename):
    with open(filename, mode='r') as f:
        data = f.readlines()
        item_dict = {}
        for line in data:
            line = line.strip('\n')
            user = line.split(':')[0]
            seq = line.split(':')[1].split(',')
            for item_id in seq:
                if eval(item_id) in item_dict:
                    item_dict[eval(item_id)] += 1
                else:
                    item_dict[eval(item_id)] = 1
                    #print('the user of %s has been done!', user)
        print('all of users have been done!')
    x = sorted(list(item_dict.keys()))
    y = [item_dict[item_id] for item_id in x]
    return x, y

if __name__ == '__main__':
    data_ml = '../data-ml/train_data.txt'
    data_zf = '../data-zf/train_data.txt'
    flag = True
    if flag:
        x1, y1, id1 = user_item_inter(3, data_ml)
        x2, y2, id2 = user_item_inter(19, data_ml)
        plt.figure()
        plt.title('Data-ML: The interaction information between the user ID and item ID.')
        plt.plot(x1, y1, label='user_id='+str(id1))
        plt.scatter(x1, y1)

        plt.plot(x2, y2, label='user_id='+str(id2))
        plt.scatter(x2, y2)


        plt.axvline(x=64, c='g', label='64', lw=2)
        plt.axvline(x=65, c='r', ls='--', label='65', lw=2)
        plt.legend(loc='upper left')

    else:
        x, y = item_stat(data_zf)
        top_k = 10
        plt.figure()
        plt.title('Data-ZF: The statistics information of item ID for all users.')
        topk_x = heapq.nlargest(top_k, range(len(y)), y.__getitem__)
        topk_y = heapq.nlargest(top_k, y)
        plt.scatter(topk_x, topk_y, label='top_10_item', color='#FF0000')
        plt.legend(loc='upper left')

        plt.plot(x, y)
        for i in topk_x:
            y[i] = 0
        plt.scatter(x, y)
    plt.show()