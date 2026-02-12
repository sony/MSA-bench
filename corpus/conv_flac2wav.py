#! python

import argparse
import json
import soundfile as sf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_audio', help='directory: output audio data')
    parser.add_argument('-f_map', help='file: file mapping info')
    args = parser.parse_args()

    print('** convert flac to wav **')
    print(' directory')
    print('  output audio : '+str(args.d_audio))
    print(' file')
    print('  mapping      : '+str(args.f_map))

    with open(args.f_map, 'r', encoding='utf-8') as f:
        a_map = json.load(f)

    for fname in a_map:
        if a_map[fname]['available'] is False:
            continue
        wave, fs = sf.read(a_map[fname]['audio'])
        sf.write(args.d_audio.rstrip('/')+'/'+fname+'.wav', wave, fs, format='WAV', subtype='PCM_16')

    print('** done **')
