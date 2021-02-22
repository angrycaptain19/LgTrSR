import sys
sys.path.append('..')

def get_seq_len(file_name):
    len_list = []
    with open(file_name, mode='r') as f:
        data = f.readlines()
        for line in data:
            line = line.strip('\n')
            seq = line.split(':')[1].split(',')
            len_list.append(len(seq))
        sorted_len_list = sorted(len_list)
        max_len = sorted_len_list[-1]
        min_len = sorted_len_list[0]

    return sorted_len_list[int(0.8*len(sorted_len_list))], max_len, min_len

if __name__ == '__main__':
    data_ml = '../data-ml/train_data.txt'
    data_zf = '../data-zf/train_data.txt'
    ml_choose_len, ml_max_len, ml_min_len = get_seq_len(data_ml)
    zf_choose_len, zf_max_len, zf_min_len = get_seq_len(data_zf)

    print('Data #', 'choose_len #', 'max_len #', 'min_len')
    print(' ml  #', '  ',ml_choose_len, '    # ', ml_max_len, '  #   ', ml_min_len)
    print(' zf  #', '  ',zf_choose_len, '     # ', zf_max_len, ' #   ', zf_min_len)