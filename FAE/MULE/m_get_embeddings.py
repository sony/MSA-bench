#! python

import json
import argparse
import numpy as np
import torch
import torchaudio
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_audio', help='directory: input audio', required=True)
    parser.add_argument('-d_embed', help='directory: output embeddings', required=True)
    parser.add_argument('-f_list', help='file: file name list', required=True)
    args = parser.parse_args()

    print('** get embeddings (MULE) **')
    print(' directory')
    print('  input audio       : '+str(args.d_audio))
    print('  output embeddings : '+str(args.d_embed))
    print(' file')
    print('  file list         : '+str(args.f_list))

    # fixed value
    hop_in_sec = 2.0
    length_trailing_margin_in_sec = 20.0

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
        duration = wave.shape[1] / sr
        nframe = int(np.ceil(duration / hop_in_sec))

        zero_tail = torch.zeros((wave.shape[0], int(sr*length_trailing_margin_in_sec)), dtype=wave.dtype)
        wave = torch.cat([wave, zero_tail], dim=1)
        torchaudio.save('tmp.wav', wave, sr)

        ret = subprocess.run(['mule', 'analyze', '--config', './supporting_data/configs/mule_embedding_timeline.yml', '-i', 'tmp.wav', '-o', args.d_embed.rstrip('/')+'/'+fname+'.npy'], stdout=subprocess.DEVNULL)
        embeddings = np.load(args.d_embed.rstrip('/')+'/'+fname+'.npy')

        # embeddings: [Z, T]
        torch.save(torch.from_numpy(embeddings.T)[:nframe], args.d_embed.rstrip('/')+'/'+fname+'.dat')
        ret = subprocess.run(['rm', args.d_embed.rstrip('/')+'/'+fname+'.npy'], stdout=subprocess.DEVNULL)

    print('** done **')
