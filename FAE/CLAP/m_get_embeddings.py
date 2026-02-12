#! python

import json
import sys
import math
import csv
import torch
import torchaudio
import argparse

import numpy as np
import librosa
import torch
import laion_clap

#sys.path.append('CLAP')
#from models import *

# quantization
'''
def int16_to_float32(x):
    return (x / 32767.0).astype('float32')

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype('int16')
'''

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_audio', help='directory: input audio', required=True)
    parser.add_argument('-d_embed', help='directory: output embeddings', required=True)
    parser.add_argument('-f_list', help='file: file name list', required=True)
    parser.add_argument('-chunk_len_msec', help='parameter: chunk length in msec', type=int, default=5000, required=True)
    parser.add_argument('-chunk_hop_msec', help='parameter: chunk hop in msec', type=int, default=500, required=True)
    parser.add_argument('-model_type', help='parameter: model type(music_audioset|music_speech_audioset)', required=True)
    parser.add_argument('-zero_padding', help='parameter: zero padding at top', action='store_true')
    args = parser.parse_args()

    print('** get embeddings (LAION-CLAP) **')
    print(' directory')
    print('  input audio       : '+str(args.d_audio))
    print('  output embeddings : '+str(args.d_embed))
    print(' file')
    print('  file list         : '+str(args.f_list))
    print(' parameter')
    print('  chunk length      : '+str(args.chunk_len_msec)+' [msec]')
    print('  chunk hop         : '+str(args.chunk_hop_msec)+' [msec]')
    print('  model type        : '+str(args.model_type))
    print('  zero padding      : '+str(args.zero_padding))

    # fixed value
    DEFAULT_FS = 44100
    MODEL_FS = 48000
    Z = 512

    # chunk processing
    chunk_sample = int(args.chunk_len_msec * (MODEL_FS/1000))
    chunk_hop_sample = int(args.chunk_hop_msec * (MODEL_FS/1000))
    print(' chunk processing')
    print('  chunk_sample      : '+str(chunk_sample))
    print('  chunk_hop_sample  : '+str(chunk_hop_sample))

    # Model
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if args.model_type == 'music_audioset':
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        model.load_ckpt('checkpoint/music_audioset_epoch_15_esc_90.14.pt')
        ext = '_'+args.model_type
    elif args.model_type == 'music_speech_audioset':
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        model.load_ckpt('checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt')
        ext = '_'+args.model_type
    else:
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt() # download the default pretrained checkpoint.
        ext = ''
    model.to(device)
    model.eval()
    print(' model')
    print('  num of parameters : '+str(count_parameters(model)))

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

        # add half chunk to top
        if args.zero_padding:
            zero_top = torch.zeros((int(chunk_sample/2)), dtype=wave_mono.dtype)
            wave_mono = torch.cat([zero_top, wave_mono])

        n_chunk = math.ceil(wave_mono.shape[0] / chunk_hop_sample)
        a_embeddings = torch.zeros((n_chunk, Z), dtype=torch.float32)

        for i in range(n_chunk):
            input_audio = torch.zeros((1, chunk_sample), dtype=torch.float32)
            idx_s = i * chunk_hop_sample
            idx_e = min(idx_s + chunk_sample, wave_mono.shape[0])
            input_audio[:, :(idx_e-idx_s)] = wave_mono[idx_s:idx_e]
            input_audio = input_audio.to(device)

            with torch.no_grad():
                embeddings = model.get_audio_embedding_from_data(x = input_audio, use_tensor=True)
            del input_audio
            # embeddings: [1, 512]
            a_embeddings[i] = embeddings.detach().cpu().unsqueeze(0)

        torch.save(a_embeddings, args.d_embed.rstrip('/')+'/'+fname+ext+'.dat')

    print('** done **')
