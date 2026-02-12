#! python

import os
import json
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_result', help='directory: inference result')
    args = parser.parse_args()

    print('** calculate n-fold averaged result **')
    print(' result directory : '+str(args.d_result))
    result = {
        'Precision@0.5': {'mean': 0.0, 'std_dev': 0.0, 'scores': []},
        'Recall@0.5': {'mean': 0.0, 'std_dev': 0.0, 'scores': []},
        'F-measure@0.5': {'mean': 0.0, 'std_dev': 0.0, 'scores': []},
        'Precision@3.0': {'mean': 0.0, 'std_dev': 0.0, 'scores': []},
        'Recall@3.0': {'mean': 0.0, 'std_dev': 0.0, 'scores': []},
        'F-measure@3.0': {'mean': 0.0, 'std_dev': 0.0, 'scores': []},
        'Pairwise Precision': {'mean': 0.0, 'std_dev': 0.0, 'scores': []},
        'Pairwise Recall': {'mean': 0.0, 'std_dev': 0.0, 'scores': []},
        'Pairwise F-measure': {'mean': 0.0, 'std_dev': 0.0, 'scores': []},
        'accuracy': {'mean': 0.0, 'std_dev': 0.0, 'scores': []}
    }

    n = 0
    a_idx_fold = os.listdir(args.d_result.rstrip('/'))
    a_idx_fold.sort()
    for idx_fold in a_idx_fold:
        if idx_fold.isdigit() is False:
            continue
        with open(args.d_result.rstrip('/')+'/'+str(idx_fold)+'/result_test.json', 'r', encoding='utf-8') as f:
            result_tmp = json.load(f)
        for metric in result:
            result[metric]['scores'].append(result_tmp[metric])
        n += 1

    for metric in result:
        result[metric]['mean'] = np.mean(np.array(result[metric]['scores']))
        result[metric]['std_dev'] = np.std(np.array(result[metric]['scores']), ddof=1)

    with open(args.d_result.rstrip('/')+'/result_test.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4, sort_keys=False)

    print('** done **')

