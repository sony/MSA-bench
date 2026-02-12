#! python

from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn

import os
import json
import math
import torchaudio
import argparse

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_audio', help='directory: input audio', required=True)
    parser.add_argument('-d_embed', help='directory: output embeddings', required=True)
    parser.add_argument('-d_embed_full', help='directory: output embeddings (full layer)')
    parser.add_argument('-layer', help='parameter: index of the embeddings layer', type=int, required=True)
    parser.add_argument('-f_list', help='file: file name list', required=True)
    parser.add_argument('-model_type', help='parameter: model type(95M|330M)', required=True)
    parser.add_argument('-chunk_len_msec', help='parameter: chunk length in msec', type=int, required=True)
    parser.add_argument('-chunk_hop_msec', help='parameter: chunk hop in msec', type=int, required=True)
    parser.add_argument('-zero_padding', help='parameter: zero padding at top', action='store_true')
    parser.add_argument('-gpu', help='parameter: use GPU', action='store_true')
    parser.add_argument('-hop_mean', help='parameter: apply mean', action='store_true')
    parser.add_argument('-n_thread', help='parameter: num thread', type=int, default=2)
    args = parser.parse_args()

    print('** get embeddings (MERT) **')
    print(' directory')
    print('  input audio       : '+str(args.d_audio))
    print('  output embeddings : '+str(args.d_embed))
    print('  output full layer : '+str(args.d_embed_full))
    print(' file')
    print('  file list         : '+str(args.f_list))
    print(' parameter')
    print('  index of layer    : '+str(args.layer))
    print('  chunk length      : '+str(args.chunk_len_msec)+' [msec]')
    print('  chunk hop         : '+str(args.chunk_hop_msec)+' [msec]')
    print('  model type        : '+str(args.model_type))
    print('  zero padding      : '+str(args.zero_padding))
    print('  hop_mean          : '+str(args.hop_mean))
    print('  GPU               : '+str(args.gpu))
    print('  num threads       : '+str(args.n_thread))

    if args.n_thread is not None:
        os.environ['OMP_NUM_THREADS'] = str(args.n_thread)

    # loading our model weights
    if args.model_type == '95M':
        model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)
        Z = 768
        L = 13
    else:
        model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True)
        Z = 1024
        L = 25
    print(' model')
    print('  num of parameters : '+str(count_parameters(model)))
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    model = model.to(device)
    model.eval()

    # fixed value
    DEFAULT_FS = 44100
    # to obtain 375 output per second
    OFFSET_SAMPLE = 80

    # resample_rate: 24000
    resample_rate = processor.sampling_rate

    # chunk processing
    if args.chunk_len_msec > 0:
        chunk_sample = int(args.chunk_len_msec * (resample_rate/1000)) + OFFSET_SAMPLE
        chunk_hop_sample = int(args.chunk_hop_msec * (resample_rate/1000))
        N = int(args.chunk_len_msec * (75/1000))#75Hz(13.3333ms)
        print(' chunk processing')
        print('  chunk_sample      : '+str(chunk_sample))
        print('  chunk_hop_sample  : '+str(chunk_hop_sample))
        print('  N                 : '+str(N))

    # fs converter
    sr = DEFAULT_FS
    tr_fsconv = torchaudio.transforms.Resample(sr, resample_rate)

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
        if sr != resample_rate:
            if sr != DEFAULT_FS:
                #print(sr)
                tr_fsconv_tmp = torchaudio.transforms.Resample(sr, resample_rate)
                wave_mono = tr_fsconv_tmp(wave_mono)
            else:
                wave_mono = tr_fsconv(wave_mono)

        if args.chunk_len_msec <= 0:
            # w/o chunk processing (as-is)
            inputs = processor(wave_mono.to(device), sampling_rate=resample_rate, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # take a look at the output shape, there are 13 layers of representation
            # each layer performs differently in different downstream tasks, you should choose empirically
            all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
            #print(all_layer_hidden_states.shape) # [13 layer, Time steps, 768 feature_dim]

            # for utterance level classification tasks, you can simply reduce the representation in time
            #time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
            #print(time_reduced_hidden_states.shape) # [13, 768]

            # you can even use a learnable weighted average representation
            #aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)
            #weighted_avg_hidden_states = aggregator(time_reduced_hidden_states.unsqueeze(0)).squeeze()
            #print(weighted_avg_hidden_states.shape) # [768]

            torch.save(all_layer_hidden_states.permute(1,0,2).contiguous().detach().cpu(), args.d_embed.rstrip('/')+'/'+fname+'_'+args.model_type+'.dat')

        else:
            # w/ chunk processing

            # add half chunk to top
            if args.zero_padding:
                zero_top = torch.zeros((int(chunk_sample/2)), dtype=wave_mono.dtype)
                wave_mono = torch.cat([zero_top, wave_mono])

            n_chunk = math.ceil(wave_mono.shape[0] / chunk_hop_sample)
            if args.hop_mean:
                a_embeddings = torch.zeros((n_chunk, L, Z), dtype=torch.float32)
            else:
                a_embeddings = torch.zeros((n_chunk, N, L, Z), dtype=torch.float32)
            for i in range(n_chunk):
                input_audio = torch.zeros(chunk_sample, dtype=torch.float32)
                idx_s = i * chunk_hop_sample
                idx_e = min(idx_s + chunk_sample, wave_mono.shape[0])
                input_audio[:(idx_e-idx_s)] = wave_mono[idx_s:idx_e]
                inputs = processor(input_audio.to(device), sampling_rate=resample_rate, return_tensors='pt').to(device)
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
                embeddings = all_layer_hidden_states.permute(1,0,2).contiguous().detach().cpu()
                # embeddings: [N, L, Z]
                if args.hop_mean:
                    a_embeddings[i] = embeddings.mean(dim=0)
                else:
                    a_embeddings[i] = embeddings

            if args.hop_mean is False:
                a_embeddings = a_embeddings.reshape(n_chunk*N, L, Z)

            if args.d_embed_full is not None:
                torch.save(a_embeddings, args.d_embed_full.rstrip('/')+'/'+fname+'_'+args.model_type+'.dat')
            torch.save(a_embeddings[:,args.layer,:], args.d_embed.rstrip('/')+'/'+fname+'_'+args.model_type+'.dat')

    print('** done **')
