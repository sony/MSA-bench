#! python

import os
import json
import sys
import torch
import torchaudio
import math
import argparse

from encodec import EncodecModel
from encodec.utils import convert_audio

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_audio', help='directory: input audio', required=True)
    parser.add_argument('-d_embed', help='directory: output embeddings', required=True)
    parser.add_argument('-f_list', help='file: file name list', required=True)
    parser.add_argument('-model_fs', help='parameter: model sampling frequency in kHz (24|48)', type=int, default=48)
    parser.add_argument('-model_bitrate', help='parameter: model bit rate in kpbs (3|6|12|24)', type=int, default=24)
    parser.add_argument('-chunk_len_msec', help='parameter: chunk length in msec', type=int, default=5000)
    parser.add_argument('-chunk_hop_msec', help='parameter: chunk hop in msec', type=int, default=500)
    parser.add_argument('-zero_padding', help='parameter: zero padding at top', action='store_true')
    parser.add_argument('-hop_mean', help='parameter: apply mean', action='store_true')
    parser.add_argument('-stereo', help='parameter: stereo PCM input', action='store_true')
    parser.add_argument('-gpu', help='parameter: use GPU', action='store_true')
    args = parser.parse_args()

    print('** get embeddings (EnCodec) **')
    print(' directory')
    print('  input audio       : '+str(args.d_audio))
    print('  output embeddings : '+str(args.d_embed))
    print(' file')
    print('  file list         : '+str(args.f_list))
    print(' parameter')
    print('  model fs          : '+str(args.model_fs)+'[kHz]')
    print('  model bitrate     : '+str(args.model_bitrate)+'[kbps]')
    print('  chunk length      : '+str(args.chunk_len_msec)+' [msec]')
    print('  chunk hop         : '+str(args.chunk_hop_msec)+' [msec]')
    print('  zero padding      : '+str(args.zero_padding))
    print('  hop_mean          : '+str(args.hop_mean))
    print('  stereo input      : '+str(args.stereo))
    print('  GPU               : '+str(args.gpu))

    # model
    if args.model_fs == 24:
        model = EncodecModel.encodec_model_24khz()
    elif args.model_fs == 48:
        model = EncodecModel.encodec_model_48khz()
    model.set_target_bandwidth(args.model_bitrate)
    print(' model')
    print('  sample_rate       : '+str(model.sample_rate))
    print('  channels          : '+str(model.channels))
    print('  frame_rate        : '+str(model.frame_rate))
    print('  bandwidth         : '+str(model.bandwidth))
    print('  num of parameters : '+str(count_parameters(model)))
    print('   encoder          : '+str(count_parameters(model.encoder)))
    print('   quantizer        : '+str(count_parameters(model.quantizer)))
    print('   decoder          : '+str(count_parameters(model.decoder)))
    # 24kHz 24kbps: sample_rate=24000, channels=1, frame_rate=75, bandwitdh=24
    # 24kHz 12kbps: sample_rate=24000, channels=1, frame_rate=75, bandwitdh=12
    # 24kHz  6kbps: sample_rate=24000, channels=1, frame_rate=75, bandwitdh=6
    # 24kHz  3kbps: sample_rate=24000, channels=1, frame_rate=75, bandwitdh=3
    # 48kHz 24kbps: sample_rate=48000, channels=2, frame_rate=150, bandwitdh=24
    # 48kHz 12kbps: sample_rate=48000, channels=2, frame_rate=150, bandwitdh=12
    # 48kHz  6kbps: sample_rate=48000, channels=2, frame_rate=150, bandwitdh=6
    # 48kHz  3kbps: sample_rate=48000, channels=2, frame_rate=150, bandwitdh=3
    if args.gpu:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = 'cpu'
    model = model.to(device)
    model.eval()

    # fixed value
    Z = 128

    # chunk processing
    if args.chunk_len_msec > 0:
        N = int(args.chunk_len_msec / 1000 * model.frame_rate)
        chunk_sample = int(args.chunk_len_msec * (model.sample_rate/1000))
        chunk_hop_sample = int(args.chunk_hop_msec * (model.sample_rate/1000))
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
        if args.stereo is False:
            wave = wave.mean(0, keepdim=True)

        if args.chunk_len_msec <= 0:
            # as-is
            wave = convert_audio(wave, sr, model.sample_rate, model.channels)
            # convert_audio: (downmix(wave.shape[0]->model.channels) and resample(sr->model.sample_rate)
            # wave: [ch, sample]

            with torch.no_grad():
                emb_enc = model.encoder(wave.unsqueeze(0).to(device))
                codes = model.quantizer.encode(emb_enc, model.frame_rate, model.bandwidth)
                emb_dec = model.quantizer.decode(codes)
                # emb_enc: [B, Z, T]
                # emb_dec: [B, Z, T]
                #encoded_frames = model.encode(wave.unsqueeze(0).to(device))
                #encoded_frames, scale = model._encode_frame(wave.unsqueeze(0).to(device))

            if args.stereo:
                torch.save(emb_dec.detach().cpu().squeeze(0).permute(1,0).contiguous(), args.d_embed.rstrip('/')+'/'+fname+'_'+str(args.model_fs)+'kHz_'+str(args.model_bitrate)+'kbps_stereo.dat')
            else:
                torch.save(emb_dec.detach().cpu().squeeze(0).permute(1,0).contiguous(), args.d_embed.rstrip('/')+'/'+fname+'_'+str(args.model_fs)+'kHz_'+str(args.model_bitrate)+'kbps.dat')
        else:
            # w/ chunk processing

            if args.zero_padding:
                zero_top = torch.zeros((wave.shape[0], int(chunk_sample/2)), dtype=wave.dtype)
                wave = torch.cat([zero_top, wave], dim=1)

            wave = convert_audio(wave, sr, model.sample_rate, model.channels)
            n_chunk = math.ceil(wave.shape[1] / chunk_hop_sample)
            if args.hop_mean:
                a_embeddings = torch.zeros((n_chunk, Z), dtype=wave.dtype)
            else:
                a_embeddings = torch.zeros((n_chunk, N, Z), dtype=wave.dtype)
            for i in range(n_chunk):
                input_audio = torch.zeros((model.channels, chunk_sample), dtype=wave.dtype)
                idx_s = i * chunk_hop_sample
                idx_e = min(idx_s + chunk_sample, wave.shape[1])
                input_audio[:, :(idx_e-idx_s)] = wave[:, idx_s:idx_e]
                with torch.no_grad():
                    emb_enc = model.encoder(input_audio.unsqueeze(0).to(device))
                    codes = model.quantizer.encode(emb_enc, model.frame_rate, model.bandwidth)
                    emb_dec = model.quantizer.decode(codes)
                    # emb_enc: [B, Z, N]
                    # emb_dec: [B, Z, N]
                del input_audio
                if args.hop_mean:
                    a_embeddings[i] = emb_dec.detach().cpu().squeeze(0).permute(1,0).contiguous().mean(dim=0)
                else:
                    a_embeddings[i] = emb_dec.detach().cpu().squeeze(0).permute(1,0).contiguous()

            if args.hop_mean is False:
                a_embeddings = a_embeddings.reshape(n_chunk*N, Z)

            if args.stereo:
                torch.save(a_embeddings, args.d_embed.rstrip('/')+'/'+fname+'_'+str(args.model_fs)+'kHz_'+str(args.model_bitrate)+'kbps_stereo.dat')
            else:
                torch.save(a_embeddings, args.d_embed.rstrip('/')+'/'+fname+'_'+str(args.model_fs)+'kHz_'+str(args.model_bitrate)+'kbps.dat')
    print('** done **')
