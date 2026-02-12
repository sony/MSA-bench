#! python

import json
import argparse
import dac
import math
import torch
from audiotools import AudioSignal

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
    parser.add_argument('-gpu', help='parameter: GPU', action='store_true')
    args = parser.parse_args()

    print('** get embeddings (DAC) **')
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
    print('  GPU               : '+str(args.gpu))

    # Download a model
    model_path = dac.utils.download(model_type="44khz")
    model = dac.DAC.load(model_path)
    if args.gpu:
        model.to('cuda')
    else:
        model.to('cpu')
    model.eval()

    # fixed value
    Z = 1024
    # model numbers
    # delay: 7904, sample_rate: 44100, hop_length: 512
    print(' model')
    print('  delay             : '+str(model.delay))
    print('  sample_rate       : '+str(model.sample_rate))
    print('  hop_length        : '+str(model.hop_length))
    print('  num of parameters : '+str(count_parameters(model)))
    print('   encoder          : '+str(count_parameters(model.encoder)))
    print('   quantizer        : '+str(count_parameters(model.quantizer)))
    print('   decoder          : '+str(count_parameters(model.decoder)))

    # chunk processing
    if args.chunk_len_msec > 0:
        N = math.ceil(args.chunk_len_msec * (model.sample_rate / 1000) / model.hop_length)
        chunk_sample = N * model.hop_length
        M = math.ceil(model.delay / model.hop_length)
        margin_sample = M * model.hop_length
        chunk_hop_sample = int(args.chunk_hop_msec * (model.sample_rate / 1000))
        print(' chunk processing')
        print('  N                 : '+str(N))#431
        print('  M                 : '+str(M))#16
        print('  chunk_sample      : '+str(chunk_sample))#220672
        print('  margin_sample     : '+str(margin_sample))#8192
        print('  chunk_hop_sample  : '+str(chunk_hop_sample))#22050

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

        input_audio = AudioSignal(args.d_audio.rstrip('/')+'/'+fname+'.wav')
        if input_audio.audio_data.shape[1] > 1:
            input_audio.to_mono()

        if args.chunk_len_msec <= 0:
            # use prepared API: encode() -> out of memory
            input_audio.to(model.device)
            x = model.preprocess(input_audio.audio_data, input_audio.sample_rate)
            z, codes, latents, _, _ = model.encode(x)
            # z      : [1, 1024, n_frame]
            # codes  : [1,    9, n_frame]
            # latents: [1,   72, n_frame]
            torch.save(z.squeeze(0).permute(1,0).contiguous().cpu(), args.d_embed.rstrip('/')+'/'+fname+'.dat')

        elif args.chunk_len_msec == args.chunk_hop_msec:
            # chunk processing (w/o overlap)

            n_frame = math.ceil(input_audio.shape[-1] / model.hop_length)
            n_chunk = math.ceil(n_frame / N)

            a_embeddings = torch.zeros((n_chunk, N, Z), dtype=torch.float32)
            zeropad_top = margin_sample
            zeropad_tail = n_frame * model.hop_length - input_audio.shape[-1]
            zeropad_tail += (n_chunk * N - n_frame) * model.hop_length
            zeropad_tail += margin_sample
            input_audio.zero_pad(zeropad_top, zeropad_tail)

            for i in range(n_chunk):
                idx_s = (i * N) * model.hop_length
                idx_e = ((i+1) * N + 2*M) * model.hop_length

                x = input_audio[..., idx_s:idx_e]

                audio_data = x.audio_data.to(model.device)
                audio_data = model.preprocess(audio_data, model.sample_rate)

                z, _, _, _, _ = model.encode(audio_data)
                # z: [B, Z, M+N+M]

                a_embeddings[i] = z.squeeze(0).permute(1,0).contiguous()[M:-M].detach().cpu()
                del x, audio_data, z

            a_embeddings = a_embeddings.reshape(n_chunk*N, Z)
            torch.save(a_embeddings, args.d_embed.rstrip('/')+'/'+fname+'.dat')

        else:
            # chunk processing

            # add half chunk to top
            if args.zero_padding:
                input_audio.zero_pad(int(chunk_sample/2), 0)

            n_frame = math.ceil(input_audio.shape[-1] / model.hop_length)
            n_chunk = math.ceil(input_audio.shape[-1] / chunk_hop_sample)

            zeropad_top = margin_sample
            zeropad_tail = n_frame * model.hop_length - input_audio.shape[-1]
            zeropad_tail += (n_chunk * N - n_frame) * model.hop_length
            zeropad_tail += margin_sample
            input_audio.zero_pad(zeropad_top, zeropad_tail)

            if args.hop_mean:
                a_embeddings = torch.zeros((n_chunk, Z), dtype=torch.float32)
            else:
                a_embeddings = torch.zeros((n_chunk, N, Z), dtype=torch.float32)
            for i in range(n_chunk):
                idx_s = i * chunk_hop_sample
                idx_e = idx_s + chunk_sample + 2*M*model.hop_length

                x = input_audio[..., idx_s:idx_e]

                audio_data = x.audio_data.to(model.device)
                audio_data = model.preprocess(audio_data, model.sample_rate)

                z, _, _, _, _ = model.encode(audio_data)
                # z: [B, Z, M+N+M]

                if args.hop_mean:
                    a_embeddings[i] = z.squeeze(0).permute(1,0).contiguous()[M:-M].detach().cpu().mean(dim=0)
                else:
                    a_embeddings[i] = z.squeeze(0).permute(1,0).contiguous()[M:-M].detach().cpu()
                del x, audio_data, z

            if args.hop_mean is False:
                a_embeddings = a_embeddings.reshape(n_chunk*N, Z)
            torch.save(a_embeddings, args.d_embed.rstrip('/')+'/'+fname+'.dat')

    print('** done **')
