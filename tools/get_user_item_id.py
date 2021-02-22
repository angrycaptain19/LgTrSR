def get_user_item_id(filename):
    with open(filename, mode='r') as f:
        data = f.readlines()
        user_dict = {}
        item_dict = {}
        count = 0
        for line in data:
            line = line.strip('\n')
            user_id = line.split(':')[0]
            num = line.split(':')[1].split(',')

            if user_id in user_dict:
                user_dict[eval(user_id)] +=1
            else:
                user_dict[eval(user_id)] = 1

            for item_id in num:
                if item_id in item_dict:
                    item_dict[eval(item_id)] += 1
                else:
                    item_dict[eval(item_id)] = 1
            count += 1
        print('all of users have been done!')
    users_id = sorted(list(user_dict.keys()))
    items_id = sorted(list(item_dict.keys()))
    number = count
    train_test_length = [int(number * 0.8), number - int(number * 0.8)]
    return number, train_test_length, users_id, items_id

if __name__ == '__main__':
    data_ml = '../data-ml/train_data.txt'
    data_zf = '../data-zf/train_data.txt'
    ml_number, ml_train_test_length, ml_users_id, ml_items_id = get_user_item_id(data_ml)
    zf_number, zf_train_test_length, zf_users_id, zf_items_id = get_user_item_id(data_zf)

    print('Data #', 'number #', 'train #', ' test   #', 'user id #', 'item id')
    print(' ml  #',  ml_number, '  # ', ml_train_test_length[0], '# ', ml_train_test_length[1], '  #', len(ml_users_id), '   #', len(ml_items_id))
    print(' zf  #',  zf_number, ' #', zf_train_test_length[0], '# ', ml_train_test_length[1], '  #', len(zf_users_id), '  #', len(zf_items_id))