#! python

import os
import json
import sys
import torch
import torchaudio
import math
import argparse

CKPT_PATH='.'
from musicfm.model.musicfm_25hz import MusicFM25Hz

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_audio', help='directory: input audio', required=True)
    parser.add_argument('-d_embed', help='directory: output embeddings', required=True)
    parser.add_argument('-f_list', help='file: file name list', required=True)
    parser.add_argument('-flash_attention', help='parameter: use flash attention to reduce GPU memory', action='store_true')
    parser.add_argument('-half_precision', help='parameter: use half precision', action='store_true')
    parser.add_argument('-model_type', help='parameter: model type(msd|fma)', required=True)
    parser.add_argument('-chunk_len_msec', help='parameter: chunk length in msec', type=int, default=5000, required=True)
    parser.add_argument('-chunk_hop_msec', help='parameter: chunk hop in msec', type=int, default=500, required=True)
    parser.add_argument('-zero_padding', help='parameter: zero padding at top', action='store_true')
    parser.add_argument('-hop_mean', help='parameter: apply mean', action='store_true')
    parser.add_argument('-layer', help='parameter: layer to extract', default=7)
    args = parser.parse_args()

    print('** get embeddings (musicfm) **')
    print(' directory')
    print('  input audio       : '+str(args.d_audio))
    print('  output embeddings : '+str(args.d_embed))
    print(' file')
    print('  file list         : '+str(args.f_list))
    print(' parameter')
    print('  chunk length      : '+str(args.chunk_len_msec)+' [msec]')
    print('  chunk hop         : '+str(args.chunk_hop_msec)+' [msec]')
    print('  model type        : '+str(args.model_type))
    print('  flash attention   : '+str(args.flash_attention))
    print('  half precision    : '+str(args.half_precision))
    print('  zero padding      : '+str(args.zero_padding))
    print('  hop_mean          : '+str(args.hop_mean))
    print('  layer             : '+str(args.layer))

    # model
    if args.model_type == 'msd':
        model = MusicFM25Hz(
            is_flash=args.flash_attention,
            stat_path=os.path.join(CKPT_PATH, 'musicfm', 'data', 'msd_stats.json'),
            model_path=os.path.join(CKPT_PATH, 'musicfm', 'data', 'pretrained_msd.pt'),
        )
    elif args.model_type == 'fma':
        model = MusicFM25Hz(
            is_flash=args.flash_attention,
            stat_path=os.path.join(CKPT_PATH, 'musicfm', 'data', 'fma_stats.json'),
            model_path=os.path.join(CKPT_PATH, 'musicfm', 'data', 'pretrained_fma.pt'),
        )
    if args.half_precision:
        model = model.half()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(' GPU                : '+str(device))
    print(' model')
    print('  num of parameters : '+str(count_parameters(model)))
    model = model.to(device)
    model.eval()

    # fixed value
    DEFAULT_FS = 44100
    MODEL_FS = 24000
    Z = 1024
    L = 13
    hop_sec = 0.04

    # chunk processing
    if args.chunk_len_msec > 0:
        N = int(args.chunk_len_msec / (hop_sec*1000))
        chunk_sample = int(args.chunk_len_msec * (MODEL_FS/1000))
        chunk_hop_sample = int(args.chunk_hop_msec * (MODEL_FS/1000))
        print(' chunk processing')
        print('  chunk_sample      : '+str(chunk_sample))
        print('  chunk_hop_sample  : '+str(chunk_hop_sample))
        print('  N                 : '+str(N))

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

        if args.chunk_len_msec == 0:
            ## as-is -> out of memory
            # w/o chunk processing
            if args.half_precision:
                input_audio = wave_mono.to(device).half()
            else:
                input_audio = wave_mono.to(device)
            with torch.no_grad():
                if args.layer == 'full':
                    _, output_embeddings = model.get_predictions(input_audio.unsqueeze(0))
                else:
                    #output_embeddings = model.get_latent(input_audio.unsqueeze(0), layer_ix=7)
                    output_embeddings = model.get_latent(input_audio.unsqueeze(0), layer_ix=int(args.layer))
            if args.layer == 'full':
                output_embeddings = torch.stack(output_embeddings).squeeze(1).detach().cpu()
            else:
                output_embeddings = output_embeddings.squeeze(0).detach().cpu()
            torch.save(output_embeddings, args.d_embed.rstrip('/')+'/'+fname+'_'+args.model_type+'.dat')

        else:
            # w/ chunk processing

            # add half chunk to top
            if args.zero_padding:
                zero_top = torch.zeros((int(chunk_sample/2)), dtype=wave_mono.dtype)
                wave_mono = torch.cat([zero_top, wave_mono])

            if args.half_precision:
                wave_mono = wave_mono.half()

            n_chunk = math.ceil(wave_mono.shape[0] / chunk_hop_sample)
            if args.layer == 'full':
                if args.hop_mean:
                    a_embeddings = torch.zeros((n_chunk, L, Z), dtype=wave_mono.dtype)
                else:
                    a_embeddings = torch.zeros((n_chunk, N, L, Z), dtype=wave_mono.dtype)
            else:
                if args.hop_mean:
                    a_embeddings = torch.zeros((n_chunk, Z), dtype=wave_mono.dtype)
                else:
                    a_embeddings = torch.zeros((n_chunk, N, Z), dtype=wave_mono.dtype)
            for i in range(n_chunk):
                input_audio = torch.zeros((1, chunk_sample), dtype=wave_mono.dtype)
                idx_s = i * chunk_hop_sample
                idx_e = min(idx_s + chunk_sample, wave_mono.shape[0])
                input_audio[:, :(idx_e-idx_s)] = wave_mono[idx_s:idx_e]
                with torch.no_grad():
                    if args.layer == 'full':
                        _, embeddings = model.get_predictions(input_audio.to(device))
                        # embeddings: [L, B, N, Z]
                    else:
                        #embeddings = model.get_latent(input_audio.to(device), layer_ix=7)
                        embeddings = model.get_latent(input_audio.to(device), layer_ix=int(args.layer))
                        # embeddings: [B, N, Z]

                del input_audio
                if args.layer == 'full':
                    if args.hop_mean:
                        a_embeddings[i] = torch.stack(embeddings).detach().cpu().squeeze(1).mean(dim=1)
                    else:
                        a_embeddings[i] = torch.stack(embeddings).detach().cpu().squeeze(1).permute(1,0,2).contiguous()
                else:
                    if args.hop_mean:
                        a_embeddings[i] = embeddings.detach().cpu().squeeze(0).mean(dim=0)
                    else:
                        a_embeddings[i] = embeddings.detach().cpu().squeeze(0)

            if args.hop_mean is False:
                if args.layer == 'full':
                    a_embeddings = a_embeddings.reshape(n_chunk*N, L, Z)
                else:
                    a_embeddings = a_embeddings.reshape(n_chunk*N, Z)

            torch.save(a_embeddings, args.d_embed.rstrip('/')+'/'+fname+'_'+args.model_type+'.dat')

    print('** done **')
