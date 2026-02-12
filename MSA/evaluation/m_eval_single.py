#! python

import os
import mir_eval
import collections
import json
import argparse
import torch

def evaluate(ref_intervals, ref_labels, est_intervals, est_labels, **kwargs):
    # Adjust timespan of estimations relative to ground truth
    ref_intervals, ref_labels = \
        mir_eval.util.adjust_intervals(ref_intervals, labels=ref_labels, t_min=0.0)

    est_intervals, est_labels = \
        mir_eval.util.adjust_intervals(est_intervals, labels=est_labels, t_min=0.0,
                              t_max=ref_intervals.max())

    # Now compute all the metrics
    scores = collections.OrderedDict()

    # Boundary detection
    # Force these values for window
    kwargs['window'] = .5
    scores['Precision@0.5'], scores['Recall@0.5'], scores['F-measure@0.5'] = \
        mir_eval.util.filter_kwargs(mir_eval.segment.detection, ref_intervals, est_intervals, **kwargs)

    kwargs['window'] = 3.0
    scores['Precision@3.0'], scores['Recall@3.0'], scores['F-measure@3.0'] = \
        mir_eval.util.filter_kwargs(mir_eval.segment.detection, ref_intervals, est_intervals, **kwargs)

    # Pairwise clustering
    (scores['Pairwise Precision'],
     scores['Pairwise Recall'],
     scores['Pairwise F-measure']) = mir_eval.util.filter_kwargs(mir_eval.segment.pairwise,
                                                                 ref_intervals,
                                                                 ref_labels,
                                                                 est_intervals,
                                                                 est_labels, **kwargs)
    '''
    # Boundary deviation
    scores['Ref-to-est deviation'], scores['Est-to-ref deviation'] = \
        mir_eval.util.filter_kwargs(deviation, ref_intervals, est_intervals, **kwargs)
    '''
    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f_list', help='file: list')
    parser.add_argument('-f_config', help='file: config')
    parser.add_argument('-d_label', help='directory: label')
    parser.add_argument('-d_reference', help='directory: reference')
    parser.add_argument('-d_result', help='directory: inference result')
    parser.add_argument('-split', help='parameter: validation or test')
    args = parser.parse_args()

    print('** MSA evaluation (single) **')
    print(' files')
    print('  file list  : '+str(args.f_list))
    print('  config     : '+str(args.f_config))
    print(' directories')
    print('  reference  : '+str(args.d_reference))
    print('  label      : '+str(args.d_label))
    print('  result     : '+str(args.d_result))
    print(' parameter')
    print('  split      : '+str(args.split))

    with open(args.f_config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    nframe_in_sec = 1.0 / config['label']['hop_sec']

    result = {
        'Precision@0.5': 0.0,
        'Recall@0.5': 0.0,
        'F-measure@0.5': 0.0,
        'Precision@3.0': 0.0,
        'Recall@3.0': 0.0,
        'F-measure@3.0': 0.0,
        'Pairwise Precision': 0.0,
        'Pairwise Recall': 0.0,
        'Pairwise F-measure': 0.0,
        'accuracy': 0.0
    }

    if args.split == 'test':
        with open(args.d_result.rstrip('/')+'/result_validation.tsv', 'r', encoding='utf-8') as f:
            a_tmp = f.readlines()
        best_epoch = str(a_tmp[0].rstrip('\n').split('\t')[1])
        result['best_epoch'] = best_epoch

    with open(args.f_list, 'r', encoding='utf-8') as f:
        a_list = json.load(f)
    a_fname = []
    if args.split == 'validation':
        for fname in a_list[args.split]:
            a_fname.append(fname)
    elif args.split == 'test':
        for split in a_list:
            for fname in a_list[split]:
                a_fname.append(fname)

    num_file = 0
    for fname in a_fname:
        print('['+fname+']')
        if args.split == 'validation':
            fname_est = args.d_result.rstrip('/')+'/'+fname+'.json'
            fname_est_lab = args.d_result.rstrip('/')+'/'+fname+'.lab'
            fname_result = args.d_result.rstrip('/')+'/'+fname+'_result.json'
        elif args.split == 'test':
            fname_est = args.d_result.rstrip('/')+'/'+best_epoch+'/'+fname+'.json'
            fname_est_lab = args.d_result.rstrip('/')+'/'+best_epoch+'/'+fname+'.lab'
            fname_result = args.d_result.rstrip('/')+'/'+best_epoch+'/'+fname+'_result.json'
        fname_ref_lab = args.d_reference.rstrip('/')+'/'+fname+'.lab'

        with open(fname_est, 'r', encoding='utf-8') as f:
            a_est = json.load(f)

        with open(args.d_reference.rstrip('/')+'/'+fname+'.lab', 'r', encoding='utf-8') as f:
            a_ref = f.readlines()
        duration = float(a_ref[len(a_ref)-1].rstrip('\n').split('\t')[1])

        # accuracy
        a_label = torch.load(args.d_label.rstrip('/')+'/'+fname+'.dat')
        nframe_label = len(a_label['structure']['name'])
        a_est_frame = torch.zeros(nframe_label, dtype=torch.int8)

        with open(fname_est_lab, 'w', encoding='utf-8') as f:
            for obj in a_est:
                if obj['offset'] - obj['onset'] <= 0:
                    print('error in '+fname+': '+str(obj))
                # to avoid the bad effect of "t_max=ref_intervals.max())" in mir_eval.util.adjust_intervals(est_intervals, labels=est_labels, t_min=0.0, t_max=ref_intervals.max())"
                if obj['onset'] >= duration:
                    continue
                f.write(str(obj['onset'])+'\t')
                f.write(str(obj['offset'])+'\t')
                f.write(obj['label_structure']+'\n')

                onset_frame = int(obj['onset'] * nframe_in_sec + 0.5)
                offset_frame = min(int(obj['offset'] * nframe_in_sec + 0.5), nframe_label)
                a_est_frame[onset_frame:offset_frame] = config['dictionary']['index'][obj['label_structure']]

        est_intervals, est_labels = mir_eval.io.load_labeled_intervals(fname_est_lab)
        ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(fname_ref_lab)
        scores = evaluate(ref_intervals, ref_labels, est_intervals, est_labels)

        correct = (a_label['structure']['name'] == a_est_frame).sum().item()
        scores['accuracy']  = correct / nframe_label

        with open(fname_result, 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False, indent=4, sort_keys=False)

        for metrics in scores:
            result[metrics] += scores[metrics]
        num_file += 1

    for metrics in scores:
        result[metrics] /= num_file
    with open(args.d_result.rstrip('/')+'/result_'+args.split+'.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4, sort_keys=False)
    print('** done **')
