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

    print('** make list (rwc) **')
    print(' files')
    print('  maps        : '+str(args.f_map))
    print(' directory')
    print('  output list : '+str(args.d_out))
    print(' random seed  : '+str(args.seed))

    random.seed(args.seed)

    with open(args.f_map, 'r', encoding='utf-8') as f:
        fname_map = json.load(f)
    a_fname = []
    for fname in fname_map:
        if fname_map[fname]['available'] is True:
            a_fname.append(fname)

    num_all = len(a_fname)
    a_index = [i for i in range(num_all)]
    random.shuffle(a_index)

    a_list_single = {'train': [], 'validation': []}
    n_train = int(num_all * 0.85)
    for i in range(n_train):
        a_list_single['train'].append(a_fname[a_index[i]])
    a_list_single['train'].sort()
    for i in range(n_train, num_all):
        a_list_single['validation'].append(a_fname[a_index[i]])
    a_list_single['validation'].sort()

    with open(args.d_out.rstrip('/')+'/list_rwc_single.json', 'w', encoding='utf-8') as f:
        json.dump(a_list_single, f, ensure_ascii=False, indent=4, sort_keys=False)

    print('** done **')
