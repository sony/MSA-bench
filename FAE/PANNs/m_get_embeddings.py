#! python

import json
import sys
import math
import csv
import torch
import torchaudio
import argparse

sys.path.append('audioset_tagging_cnn/pytorch')
from models import *

def get_config(f_config):
    # Load label
    with open(f_config, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)

    labels = []
    for i1 in range(1, len(lines)):
        label = lines[i1][2]
        labels.append(label)

    classes_num = len(labels)

    return classes_num, labels

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_audio', help='directory: input audio', required=True)
    parser.add_argument('-d_embed', help='directory: output embeddings', required=True)
    parser.add_argument('-f_list', help='file: file name list', required=True)
    parser.add_argument('-f_checkpoint', help='file: checkpoint', required=True)
    parser.add_argument('-model_type', help='parameter: model type(Cnn14|Cnn14_16k|Cnn14_DecisionLevelMax)', required=True)
    parser.add_argument('-chunk_len_msec', help='parameter: chunk length in msec', type=int, default=5000)
    parser.add_argument('-chunk_hop_msec', help='parameter: chunk hop in msec', type=int, default=500)
    parser.add_argument('-zero_padding', help='parameter: zero padding at top', action='store_true')
    parser.add_argument('-hop_mean', help='parameter: apply mean', action='store_true')
    args = parser.parse_args()

    print('** get embeddings (PANNs) **')
    print(' directory')
    print('  input audio       : '+str(args.d_audio))
    print('  output embeddings : '+str(args.d_embed))
    print(' file')
    print('  file list         : '+str(args.f_list))
    print('  checkpoint        : '+str(args.f_checkpoint))
    print(' parameter')
    print('  model type        : '+str(args.model_type))
    print('  chunk length      : '+str(args.chunk_len_msec)+' [msec]')
    print('  chunk hop         : '+str(args.chunk_hop_msec)+' [msec]')
    print('  zero padding      : '+str(args.zero_padding))
    print('  hop_mean          : '+str(args.hop_mean))

    # fixed value
    DEFAULT_FS = 44100
    MODEL_FS = 32000
    sample_rate = MODEL_FS
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 50
    fmax = 14000
    Z = 2048

    if args.chunk_len_msec > 0:
        N = int(args.chunk_len_msec / hop_size)
        chunk_sample = int(args.chunk_len_msec * (MODEL_FS/1000))
        chunk_hop_sample = int(args.chunk_hop_msec * (MODEL_FS/1000))
        print(' chunk processing')
        print('  chunk_sample      : '+str(chunk_sample))
        print('  chunk_hop_sample  : '+str(chunk_hop_sample))
        print('  N                 : '+str(N))

    classes_num, labels = get_config('audioset_tagging_cnn/metadata/class_labels_indices.csv')
    frames_per_second = sample_rate // hop_size
    #print('classes_num: '+str(classes_num))
    #print('labels     : '+str(labels))
    #print('frame/sec  : '+str(frames_per_second))

    # Model
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    Model = eval(args.model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    checkpoint = torch.load(args.f_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(' model')
    print('  num of parameters : '+str(count_parameters(model)))
    if 'cuda' in str(device):
        model.to(device)
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')
    model.eval()
    
    # fs converter
    sr = DEFAULT_FS
    tr_fsconv = torchaudio.transforms.Resample(sr, MODEL_FS)

    # get embeddings
    with open(args.f_list, 'r', encoding='utf-8') as f:
        a_fname_tmp = json.load(f)
    a_fname = []
    for split in a_fname_tmp:
        for fname in a_fname_tmp[split]:
            if fname.startswith('#'):
                continue
            a_fname.append(fname)
    del a_fname_tmp
    a_fname.sort()
    for fname in a_fname:
        print(fname)

        wave, sr = torchaudio.load(args.d_audio.rstrip('/')+'/'+fname+'.wav')
        wave_mono = torch.mean(wave, dim=0)
        del wave
        if sr != MODEL_FS:
            if sr != DEFAULT_FS:
                print(sr)
                tr_fsconv_tmp = torchaudio.transforms.Resample(sr, MODEL_FS)
                wave_mono = tr_fsconv_tmp(wave_mono)
            else:
                wave_mono = tr_fsconv(wave_mono)

        # to avoid missing data ending
        zero_tail = torch.zeros(int(MODEL_FS*0.32), dtype=wave_mono.dtype)
        wave_mono = torch.cat([wave_mono, zero_tail])

        if args.chunk_len_msec == 0:
            # 'Cnn14_DecisionLevelMax' only
            # original implementation
            wave_mono = wave_mono.unsqueeze(0).to(device)
            with torch.no_grad():
                batch_output_dict = model(wave_mono, None)
                # [B, T, Z]
            del wave_mono
 
            embeddings = batch_output_dict['embedding'].data.cpu().squeeze(0)
            torch.save(embeddings, args.d_embed.rstrip('/')+'/'+fname+'_'+args.model_type+'.dat')

        else:
            # w/ chunk processing

            # add half chunk to top
            if args.zero_padding:
                zero_top = torch.zeros((int(chunk_sample/2)), dtype=wave_mono.dtype)
                wave_mono = torch.cat([zero_top, wave_mono])

            n_chunk = math.ceil(wave_mono.shape[0] / chunk_hop_sample)
            if (args.model_type == 'Cnn14_DecisionLevelMax') and (args.hop_mean is False):
                a_embeddings = torch.zeros((n_chunk, N, Z), dtype=torch.float32)
            else:
                a_embeddings = torch.zeros((n_chunk, Z), dtype=torch.float32)
            for i in range(n_chunk):
                input_audio = torch.zeros((1, chunk_sample), dtype=torch.float32)
                idx_s = i * chunk_hop_sample
                idx_e = min(idx_s + chunk_sample, wave_mono.shape[0])
                input_audio[:, :(idx_e-idx_s)] = wave_mono[idx_s:idx_e]
                input_audio = input_audio.to(device)

                with torch.no_grad():
                    batch_output_dict = model(input_audio, None)
                    # Cnn14: [B, Z]
                    # Cnn14_DecisionLevelMax: [B, N, Z]
                del input_audio

                if (args.model_type == 'Cnn14_DecisionLevelMax') and (args.hop_mean):
                    a_embeddings[i] = batch_output_dict['embedding'].data.cpu().detach().squeeze(0).mean(dim=0)
                else:
                    a_embeddings[i] = batch_output_dict['embedding'].data.cpu().detach().squeeze(0)

            if (args.model_type == 'Cnn14_DecisionLevelMax') and (args.hop_mean is False):
                a_embeddings = a_embeddings.reshape(n_chunk*N, Z)

            torch.save(a_embeddings, args.d_embed.rstrip('/')+'/'+fname+'_'+args.model_type+'.dat')

    print('** done **')
