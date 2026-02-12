#! python

import json
import sys
import torch
import torchaudio
import math
import argparse

sys.path.append('passt_hear21')
from hear21passt.base import load_model, get_scene_embeddings, get_timestamp_embeddings

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_audio', help='directory: input audio', required=True)
    parser.add_argument('-d_embed', help='directory: output embeddings', required=True)
    parser.add_argument('-f_list', help='file: file name list', required=True)
    parser.add_argument('-chunk_len_msec', help='parameter: chunk length in msec', type=int, default=5000, required=True)
    parser.add_argument('-chunk_hop_msec', help='parameter: chunk hop in msec', type=int, default=500, required=True)
    parser.add_argument('-zero_padding', help='parameter: zero padding at top', action='store_true')
    parser.add_argument('-hop_mean', help='parameter: apply mean', action='store_true')
    args = parser.parse_args()

    print('** get embeddings (PaSST) **')
    print(' directory')
    print('  input audio       : '+str(args.d_audio))
    print('  output embeddings : '+str(args.d_embed))
    print(' file')
    print('  file list         : '+str(args.f_list))
    print(' parameter')
    print('  chunk length      : '+str(args.chunk_len_msec)+' [msec]')
    print('  chunk hop         : '+str(args.chunk_hop_msec)+' [msec]')
    print('  zero padding      : '+str(args.zero_padding))
    print('  hop_mean          : '+str(args.hop_mean))

    # model
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(device)
    model = load_model().to(device)
    model.eval()
    print(' model')
    print('  num of parameters : '+str(count_parameters(model)))

    # fixed value
    DEFAULT_FS = 44100
    MODEL_FS = 32000
    Z = 768

    # fs converter
    sr = DEFAULT_FS
    tr_fsconv = torchaudio.transforms.Resample(sr, MODEL_FS)

    # chunk processing
    if args.chunk_len_msec > 0:
        #N = int(args.chunk_len_msec / 50)# 50ms
        N = int(args.chunk_len_msec / 50)+1# 50ms
        chunk_sample = int(args.chunk_len_msec * (MODEL_FS/1000))
        chunk_hop_sample = int(args.chunk_hop_msec * (MODEL_FS/1000))
        print(' chunk processing')
        print('  chunk_sample      : '+str(chunk_sample))
        print('  chunk_hop_sample  : '+str(chunk_hop_sample))
        print('  N                 : '+str(N))

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

        if args.chunk_len_msec == 0:
            # original implementation
            input_audio = torch.Tensor(wave_mono).type(torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                output_embed, time_stamps = get_timestamp_embeddings(input_audio, model)
                # output_embed: [B, N, H+Z] (1, ceil(input_audio.shape[0]/32000/0.05), 578+768)
                # time_stamps: [B, N] (1, ceil(input_audio.shape[0]/32000/0.05))

            torch.save(output_embed[:,:,-Z:].squeeze(0).detach().cpu(), args.d_embed.rstrip('/')+'/'+fname+'.dat')

        else:
            # w/ chunk processing

            # add half chunk to top
            if args.zero_padding:
                zero_top = torch.zeros((int(chunk_sample/2)), dtype=wave_mono.dtype)
                wave_mono = torch.cat([zero_top, wave_mono])

            n_chunk = math.ceil(wave_mono.shape[0] / chunk_hop_sample)
            if args.hop_mean:
                a_embeddings = torch.zeros((n_chunk, Z), dtype=torch.float32)
            else:
                a_embeddings = torch.zeros((n_chunk, N, Z), dtype=torch.float32)
            for i in range(n_chunk):
                input_audio = torch.zeros(chunk_sample, dtype=torch.float32)
                idx_s = i * chunk_hop_sample
                idx_e = min(idx_s + chunk_sample, wave_mono.shape[0])
                input_audio[:(idx_e-idx_s)] = wave_mono[idx_s:idx_e]
                with torch.no_grad():
                    output_embed, time_stamps = get_timestamp_embeddings(input_audio.unsqueeze(0).to(device), model)
                #print(output_embed.shape)#[B, N, H(527)+Z(768)]
                if args.hop_mean:
                    embeddings = output_embed[:,:,-Z:].squeeze(0).mean(dim=0).detach().cpu()
                else:
                    embeddings = output_embed[:,:,-Z:].squeeze(0).detach().cpu()
                a_embeddings[i] = embeddings
            if args.hop_mean is False:
                a_embeddings = a_embeddings.reshape(n_chunk*N, Z)

            torch.save(a_embeddings, args.d_embed.rstrip('/')+'/'+fname+'.dat')

    print('** done **')
