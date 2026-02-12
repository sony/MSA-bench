#! python

import argparse
import math
import json
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f_map', help='file: fname mapping')
    parser.add_argument('-f_config', help='file: config')
    parser.add_argument('-d_label', help='directory: label')
    parser.add_argument('-smooth', help='parameter: smoothing', action='store_true')
    parser.add_argument('-tolerance', help='parameter: tolerance in msec', type=int, default=500)
    args = parser.parse_args()

    print('** convert label for training **')
    print(' directory')
    print('  label      : '+str(args.d_label))
    print(' file')
    print('  fname map  : '+str(args.f_map))
    print('  config     : '+str(args.f_config))
    print(' parameter')
    print('  smoothing  : '+str(args.smooth))
    print('  tolerance  : '+str(args.tolerance)+'[msec]')

    with open(args.f_map, 'r', encoding='utf-8') as f:
        fname_map = json.load(f)
    with open(args.f_config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    nframe_in_sec = 1.0 / config['label']['hop_sec']
    hop_ms = config['label']['hop_sec'] * 1000.0
    # tolerance: 500[ms]
    onset_sharpness = int(args.tolerance / hop_ms + 0.5) + 1

    for fname in fname_map:
        if fname_map[fname]['available'] is False:
            continue
        #print(fname)

        with open(args.d_label.rstrip('/')+'/'+fname+'_structure.json', 'r', encoding='utf-8') as f:
            a_label = json.load(f)

        nframe = int(fname_map[fname]['duration'] * nframe_in_sec + 0.5) + 1
        if args.smooth is False:
            label_out = {
                'structure': {
                    'name': torch.zeros(nframe, dtype=torch.int8),
                    'boundary': torch.zeros(nframe, dtype=torch.bool)
                }
            }
        else:
            label_out = {
                'structure': {
                    'name': torch.zeros(nframe, dtype=torch.int8),
                    'boundary': torch.zeros(nframe, dtype=torch.float32)
                }
            }

        # structure
        if args.smooth is False:
            for i in range(len(a_label)):
                if 'label_structure' in a_label[i]:
                    onset_frame = int(a_label[i]['onset'] * nframe_in_sec + 0.5)
                    label_out['structure']['boundary'][onset_frame] = 1
                    offset_frame = -1
                    for j in range(i+1, len(a_label)):
                        if 'label_structure' in a_label[j]:
                            offset_frame = int(a_label[j]['onset'] * nframe_in_sec + 0.5)
                            break
                    if (offset_frame > nframe) or \
                       (offset_frame < 0):
                        offset_frame = nframe
                    label_out['structure']['name'][onset_frame:offset_frame] = config['dictionary']['index'][a_label[i]['label_structure']]
        else:
            for i in range(len(a_label)):
                if 'label_structure' in a_label[i]:
                    onset_frame = int(a_label[i]['onset'] * nframe_in_sec + 0.5)
                    onset_ms = a_label[i]['onset'] * 1000.0
                    label_out['structure']['boundary'][onset_frame] = 1
                    for j in range(1, onset_sharpness):
                        if onset_frame - j >= 0:
                            label_out['structure']['boundary'][onset_frame - j] = max((onset_sharpness - j) / onset_sharpness, label_out['structure']['boundary'][onset_frame - j])
                        if onset_frame + j < nframe:
                            label_out['structure']['boundary'][onset_frame + j] = max((onset_sharpness - j) / onset_sharpness, label_out['structure']['boundary'][onset_frame + j])

                    '''
                    for j in range(0, onset_sharpness+1):
                        onset_ms_q = (onset_frame + j) * hop_ms
                        onset_ms_diff = onset_ms_q - onset_ms
                        onset_val = max(0.0, 1.0 - (abs(onset_ms_diff) / (onset_sharpness * hop_ms)))
                        if onset_frame+j < nframe:
                            label_out['structure']['boundary'][onset_frame+j] = max(label_out['structure']['boundary'][onset_frame+j], onset_val)

                    for j in range(1, onset_sharpness+1):
                        onset_ms_q = (onset_frame - j) * hop_ms
                        onset_ms_diff = onset_ms_q - onset_ms
                        onset_val = max(0.0, 1.0 - (abs(onset_ms_diff) / (onset_sharpness * hop_ms)))
                        if onset_frame-j >= 0:
                            label_out['structure']['boundary'][onset_frame-j] = max(label_out['structure']['boundary'][onset_frame-j], onset_val)
                    '''
                    offset_frame = -1
                    for j in range(i+1, len(a_label)):
                        if 'label_structure' in a_label[j]:
                            offset_frame = int(a_label[j]['onset'] * nframe_in_sec + 0.5)
                            break
                    if (offset_frame > nframe) or \
                       (offset_frame < 0):
                        offset_frame = nframe
                    label_out['structure']['name'][onset_frame:offset_frame] = config['dictionary']['index'][a_label[i]['label_structure']]

        '''
        for i in range(len(a_label)):
            if 'label_structure' in a_label[i]:
                onset_frame = int(a_label[i]['onset'] * nframe_in_sec + 0.5)
                label_out['structure']['boundary'][onset_frame] = 1
                offset_frame = -1
                for j in range(i+1, len(a_label)):
                    if 'label_structure' in a_label[j]:
                        offset_frame = int(a_label[j]['onset'] * nframe_in_sec + 0.5)
                        break
                if (offset_frame > nframe) or \
                   (offset_frame < 0):
                    offset_frame = nframe
                label_out['structure']['name'][onset_frame:offset_frame] = config['dictionary']['index'][a_label[i]['label_structure']]
        '''
        torch.save(label_out, args.d_label.rstrip('/')+'/'+fname+'.dat')

    print('** done **')
