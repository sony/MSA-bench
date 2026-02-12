#! python

import os
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_result', help='directory: result')
    parser.add_argument('-epoch_start', help='min epoch', type=int, default=0)
    parser.add_argument('-epoch_end', help='max epoch', type=int, default=100)
    args = parser.parse_args()

    print('** summary **')
    print(' result directory : '+str(args.d_result))
    print(' epoch            : '+str(args.epoch_start)+' - '+str(args.epoch_end))

    a_epoch = []
    a_result = []
    for epoch in range(args.epoch_start):
        a_epoch.append(str(epoch).zfill(3))
        a_result.append(None)
        
    max_epoch = args.epoch_start
    for epoch in range(args.epoch_start, args.epoch_end):
        a_epoch.append(str(epoch).zfill(3))
        if os.path.exists(args.d_result.rstrip('/')+'/'+str(epoch).zfill(3)+'/result_validation.json'):
            if max_epoch < epoch:
                max_epoch = epoch
            with open(args.d_result.rstrip('/')+'/'+str(epoch).zfill(3)+'/result_validation.json', 'r', encoding='utf-8') as f:
                a_result.append(json.load(f))
        else:
            print('epoch: '+str(epoch)+' missing!') 
            print(args.d_result.rstrip('/')+'/'+str(epoch).zfill(3)+'/result_validation.json')
            max_epoch = epoch-1
            break
    print('epoch: '+str(max_epoch))

    a_metrics = ['F-measure@0.5', 'F-measure@3.0', 'Pairwise F-measure', 'accuracy']
    max_epoch += 1
    a_sum_score = []
    best_score = 0
    best_epoch = 0
    for epoch in range(args.epoch_start):
        a_sum_score.append(0)
    for epoch in range(args.epoch_start, args.epoch_end):
        score = 0
        for metrics in a_metrics:
            score += a_result[epoch][metrics]
        if best_score < score:
            best_score = score
            best_epoch = epoch
        a_sum_score.append(score)

    with open(args.d_result.rstrip('/')+'/result_validation.tsv', 'w', encoding='utf-8') as f:
        f.write('epoch')
        f.write('\t'+str(best_epoch).zfill(3))
        for epoch in range(args.epoch_start, args.epoch_end):
            f.write('\t'+str(epoch).zfill(3))
        f.write('\n')

        ## sum
        f.write('all\t')
        f.write('sum')
        f.write('\t'+str(best_score))
        for i in range(args.epoch_start):
            f.write('\t')
        for i in range(args.epoch_start, max_epoch):
            f.write('\t'+str(a_sum_score[i]))
        for i in range(max_epoch, 100):
            f.write('\t')
        f.write('\n')

        for metrics in a_result[best_epoch]:
            f.write('each_metric\t')
            f.write(metrics)
            f.write('\t'+str(a_result[best_epoch][metrics]))
            for i in range(args.epoch_start):
                f.write('\t')
            for i in range(args.epoch_start, max_epoch):
                f.write('\t'+str(a_result[i][metrics]))
            for i in range(max_epoch, 100):
                f.write('\t')
            f.write('\n')

    print('best epoch : '+str(best_epoch))
    print('** done **')
