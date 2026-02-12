#! python

import argparse
import json
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f_map', help='file: fname mapping')
    parser.add_argument('-d_out', help='directory: output')
    parser.add_argument('-seed', help='parameter: random seed', type=int, default=0)
    args = parser.parse_args()

    print('** make list (harmonix) **')
    print(' files')
    print('  maps        : '+str(args.f_map))
    print(' directory')
    print('  output list : '+str(args.d_out))
    print(' random seed  : '+str(args.seed))

    random.seed(args.seed)

    #
    # random split
    #
    with open(args.f_map, 'r', encoding='utf-8') as f:
        a_fname_map = json.load(f)
    a_fname = []
    a_fname_duplicate = []
    for fname in a_fname_map:
        if a_fname_map[fname]['available'] is False:
            continue
        if ('info' in a_fname_map[fname]) and \
           ('duplicate' in a_fname_map[fname]['info']):
            if a_fname_map[fname]['info']['duplicate']['duration_order'] == 0:
                a_group = []
                for i in range(len(a_fname_map[fname]['info']['duplicate']['group'])):
                    a_group.append(a_fname_map[fname]['info']['duplicate']['group'][i]['name'])
                a_fname_duplicate.append(a_group)
        else:                
            a_fname.append(fname)
    n_normal = len(a_fname)
    n_duplicate = 0
    for i in range(len(a_fname_duplicate)):
        n_duplicate += len(a_fname_duplicate[i])
    n_all = n_normal + n_duplicate
    a_index = [i for i in range(n_normal)]
    random.shuffle(a_index)

    # 8-fold
    a_list_8fold = {}
    for n in range(8):
        a_list_8fold[n] = []
        for i in range(len(a_fname_duplicate)):
            if (i % 8) == n:
                for j in range(len(a_fname_duplicate[i])):
                    a_list_8fold[n].append(a_fname_duplicate[i][j])
    max_num = 0
    for n in range(8):
        if max_num < len(a_list_8fold[n]):
            max_num = len(a_list_8fold[n])

    offset = 0
    for n in range(8):
        diff = max_num - len(a_list_8fold[n])
        for i in range(diff):
            a_list_8fold[n].append(a_fname[a_index[i+offset]])
        offset += diff

    num_all = n_normal-offset
    a_num = [int(num_all/8) for i in range(8)]
    num_remains = num_all - sum(a_num)
    for i in range(num_remains):
        a_num[i] += 1
    for n in range(8):
        for i in range(a_num[n]):
            a_list_8fold[n].append(a_fname[a_index[i+offset]])
        offset += a_num[n]
        a_list_8fold[n].sort()

    with open(args.d_out.rstrip('/')+'/list_harmonix_8fold.json', 'w', encoding='utf-8') as f:
        json.dump(a_list_8fold, f, ensure_ascii=False, indent=4, sort_keys=False)

    # train/valid
    n_train = int(len(a_fname_duplicate)*0.85)
    n_valid = len(a_fname_duplicate) - n_train
    a_list_single = {'train': [], 'validation': []}
    for i in range(n_train):
        for j in range(len(a_fname_duplicate[i])):
            a_list_single['train'].append(a_fname_duplicate[i][j])
    for i in range(n_valid):
        for j in range(len(a_fname_duplicate[i+n_train])):
            a_list_single['validation'].append(a_fname_duplicate[i+n_train][j])

    n_train = int(n_all * 0.85 - len(a_list_single['train']))
    for i in range(n_train):
        a_list_single['train'].append(a_fname[a_index[i]])
    a_list_single['train'].sort()
    for i in range(n_train, n_normal):
        a_list_single['validation'].append(a_fname[a_index[i]])
    a_list_single['validation'].sort()

    with open(args.d_out.rstrip('/')+'/list_harmonix_single.json', 'w', encoding='utf-8') as f:
        json.dump(a_list_single, f, ensure_ascii=False, indent=4, sort_keys=False)

    print('** done **')
