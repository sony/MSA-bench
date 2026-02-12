#! python

import argparse
import torch
import math
import numpy as np
import pickle
import json
import sys
from scipy.signal import find_peaks
sys.path.append('..')

def func_boundary_detection(nframe_label, output, threshold):
    output_boundary = output['boundary'].numpy()
    a_boundary_detect = [nframe_label]
    #for i in range(nframe_label):
    nframe = min(nframe_label, len(output_boundary))
    for i in range(nframe):
        if output_boundary[i] >= threshold:
            left_flag = True
            for ii in range(i-1, -1, -1):
                if output_boundary[i] > output_boundary[ii]:
                    left_flag = True
                    break
                elif output_boundary[i] < output_boundary[ii]:
                    left_flag = False
                    break
            right_flag = True
            for ii in range(i+1, len(output_boundary)):
                if output_boundary[i] > output_boundary[ii]:
                    right_flag = True
                    break
                elif output_boundary[i] < output_boundary[ii]:
                    right_flag = False
                    break
            if (left_flag is True) and (right_flag is True):
                a_boundary_detect.append(i)
    a_boundary_detect = sorted(list(set(a_boundary_detect)))

    return a_boundary_detect


def func_boundary_detection_peak_picking(nframe_label, output, threshold):
    output_boundary = output['boundary'].numpy()
    nframe = min(nframe_label, len(output_boundary))

    a_boundary_tmp = []
    for i in range(nframe):
        # detection interval : -6sec~+6sec -> -12~+12
        idx_s_detect = max(i-12, 0)
        idx_e_detect = min(i+12, len(output_boundary))
        # average subtraction: -12sec~+6sec -> -24~+12
        idx_s_mean = max(i-24, 0)
        idx_e_mean = idx_e_detect

        mean = output_boundary[idx_s_mean:idx_e_mean].mean()
        a_val = output_boundary[idx_s_detect:idx_e_detect] - mean
        loc = (a_val.argmax() + idx_s_detect).item()

        #val = (a_val[a_val.argmax()]).item()
        val = output_boundary[loc]
        a_boundary_tmp.append(loc)

    a_boundary_tmp = sorted(list(set(a_boundary_tmp)))

    idx_s = 0
    idx_e = 0
    a_segment = []
    for i in range(len(a_boundary_tmp)):
        if a_boundary_tmp[i] == 0:
            continue
        if a_boundary_tmp[i] == idx_e+1:
            idx_e += 1
        else:
            a_segment.append({'s': idx_s, 'e': idx_e})
            idx_s = a_boundary_tmp[i]
            idx_e = a_boundary_tmp[i]
        if i == len(a_boundary_tmp) - 1:
            a_segment.append({'s': idx_s, 'e': idx_e})

    a_boundary = []
    for i, segment in enumerate(a_segment):
        if segment['s'] == segment['e']:
            if output_boundary[segment['s']] >= threshold:
                a_boundary.append(segment['s'])
        else:
            data = output_boundary[segment['s']:segment['e']+1]
            if segment['s'] == 0:
                data = np.insert(data, 0, 0)
            else:
                data = np.insert(data, 0, output_boundary[segment['s']-1])
            if segment['e'] == nframe - 1:
                data = np.append(data, 0)
            else:
                data = np.append(data, output_boundary[segment['e']+1])
            peaks, _ = find_peaks(data)
            for loc in peaks:
                if output_boundary[loc+segment['s']-1] >= threshold:
                    a_boundary.append((loc+segment['s']-1).item())

    # edge
    if (0 in a_boundary) is False:
        flag = False
        for i in range(len(a_boundary)):
            if a_boundary[i] == 1:
                a_boundary[i] = 0
                flag = True
        if flag is False:
            a_boundary.append(0)

    flag = False
    for i in range(len(a_boundary)):
        if a_boundary[i] == nframe - 1:
            a_boundary[i] = nframe
            flag = True
    if flag is False:
        a_boundary.append(nframe)
    a_boundary = sorted(list(set(a_boundary)))

    return a_boundary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f_list', help='file: list')
    parser.add_argument('-f_config', help='file: config')
    parser.add_argument('-f_parameter', help='file: parameter')
    parser.add_argument('-f_model', help='file: checkpoint')
    parser.add_argument('-d_model', help='directory: checkpoint')
    parser.add_argument('-d_label', help='directory: label')
    parser.add_argument('-d_embed', help='directory: embedding')
    parser.add_argument('-d_result', help='directory: output')
    parser.add_argument('-split', help='parameter: validation of test')
    parser.add_argument('-thred', help='parameter: threshold value for boundary detection', type=float, default=0.5)
    args = parser.parse_args()

    print('** MSA inference (8-fold) **')
    print(' checkpoint')
    print('  directory  : '+str(args.d_model))
    print('  model file : '+str(args.f_model))
    print(' file')
    print('  list       : '+str(args.f_list))
    print('  config     : '+str(args.f_config))
    print('  parameter  : '+str(args.f_parameter))
    print(' directories')
    print('  label      : '+str(args.d_label))
    print('  embeddings : '+str(args.d_embed))
    print('  result     : '+str(args.d_result))
    print(' parameter')
    print('  split      : '+str(args.split))
    print('  threshold  : '+str(args.thred))

    if args.split == 'test':
        with open(args.d_result.rstrip('/')+'/result_validation.tsv', 'r', encoding='utf-8') as f:
            a_tmp = f.readlines()
        best_epoch = str(a_tmp[0].rstrip('\n').split('\t')[1])

    # config
    with open(args.f_parameter, 'r', encoding='utf-8') as f:
        parameter = json.load(f)

    idx_fold = parameter['dataset']['idx_fold']
    method = parameter['model']['embed_method']
    model_type = parameter['model']['embed_model_type']

    with open(args.f_config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    index2label = {}
    for label in config['dictionary']['index']:
        index2label[config['dictionary']['index'][label]] = label

    # list file
    with open(args.f_list, 'r', encoding='utf-8') as f:
        a_list = json.load(f)
    n_fold = len(a_list)
    # fname for test
    a_fname = []
    if args.split == 'validation':
        for fold in a_list:
            if str(fold) == str(idx_fold):
                for fname in a_list[fold]:
                    a_fname.append(fname)
            else:
                continue
    elif args.split == 'test':
        for fold in a_list:
            if str(fold) == str((int(idx_fold)+1) % int(n_fold)):
                for fname in a_list[fold]:
                    a_fname.append(fname)
            else:
                continue

    # model
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if args.split == 'validation':
        with open(args.d_model.rstrip('/')+'/'+args.f_model, 'rb') as f:
            model = pickle.load(f)
    elif args.split == 'test':
        with open(args.d_model.rstrip('/')+'/model_'+best_epoch+'.pkl', 'rb') as f:
            model = pickle.load(f)
    model = model.to(device)
    model.eval()

    Ti = int(parameter['model_parameters']['Ti'])
    To = int(parameter['model_parameters']['To'])
    Z = parameter['model_parameters']['Z']
    hop_sec = parameter['config']['label']['hop_sec']

    # inference
    if model_type is not None:
        ext = '_' + model_type + '.dat'
    else:
        ext = '.dat'
    for fname in a_fname:
        print('['+str(fname)+']')

        # input embedding
        feature_embed_file = torch.load(args.d_embed+'/'+fname+ext)
        '''
        if 'n_layer' in parameter['model_parameters']:
            feature_embed_file = feature_embed_file[:, parameter['model_parameters']['n_layer'], :]
        '''
        nframe_embed = feature_embed_file.shape[0]
        n_proc_embed = math.ceil(nframe_embed / Ti)

        label_file = torch.load(args.d_label.rstrip('/')+'/'+fname+'.dat')
        nframe_label = label_file['structure']['name'].shape[0]
        n_proc_label = math.ceil(nframe_label / Ti)

        # output label
        output = {
            'name': torch.zeros((n_proc_embed*To, parameter['model_parameters']['structure']['name']), dtype=torch.float32),
            'boundary': torch.zeros((n_proc_embed*To), dtype=torch.float32)
        }

        # processing
        for i in range(n_proc_embed):
            idx_s = int(i * Ti)
            idx_e = min(idx_s+int(Ti), nframe_embed)
            input_emb = torch.zeros([int(Ti), Z], dtype=torch.float32)
            if idx_s < idx_e:
                input_emb[:(idx_e-idx_s)] = feature_embed_file[idx_s:idx_e]

            with torch.no_grad():
                output_structure_name, output_structure_boundary = model(input_emb.unsqueeze(0).to(device))

            output['name'][i*To:(i+1)*To] = output_structure_name.squeeze(0).to('cpu').detach()
            output['boundary'][i*To:(i+1)*To] = (torch.sigmoid(output_structure_boundary).squeeze(0)).to('cpu').detach()

        if n_proc_embed*To > nframe_label:
            output['name'] = output['name'][:nframe_label]
            output['boundary'] = output['boundary'][:nframe_label]
        if args.split == 'validation':
            torch.save(output, args.d_result.rstrip('/')+'/'+fname+'.dat')
        elif args.split == 'test':
            torch.save(output, args.d_result.rstrip('/')+'/'+best_epoch+'/'+fname+'.dat')

        # convert to structure
        # (1) boundary detection
        #a_boundary_detect = func_boundary_detection(nframe_label, output, args.thred)
        a_boundary_detect = func_boundary_detection_peak_picking(nframe_label, output, args.thred)

        # (2) label name
        a_structure = []
        for i in range(len(a_boundary_detect)-1):
            idx_s = a_boundary_detect[i]
            idx_e = a_boundary_detect[i+1]
            a_structure.append({'onset': idx_s * hop_sec,
                                'offset': idx_e * hop_sec,
                                'label_structure': index2label[output['name'][idx_s:idx_e].mean(dim=0).argmax().item()]})
                                
        if args.split == 'validation':
            with open(args.d_result.rstrip('/')+'/'+fname+'.json', 'w', encoding='utf-8') as f:
                json.dump(a_structure, f, ensure_ascii=False, indent=4, sort_keys=False)
        elif args.split == 'test':
            with open(args.d_result.rstrip('/')+'/'+best_epoch+'/'+fname+'.json', 'w', encoding='utf-8') as f:
                json.dump(a_structure, f, ensure_ascii=False, indent=4, sort_keys=False)

    print('** done **')
