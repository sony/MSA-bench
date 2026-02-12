#! python

import argparse
import glob
import json
import soundfile as sf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_structure', help='directory: structure annotation')
    parser.add_argument('-d_audio', help='directory: (CD)audio data')
    parser.add_argument('-f_map', help='file: file mapping info')
    args = parser.parse_args()

    print('** make file map for RWC dataset **')
    print(' directory')
    print('  audio        : '+str(args.d_audio))
    print('  structure    : '+str(args.d_structure))
    print(' file')
    print('  mapping info : '+str(args.f_map))

    fname_map = {}

    # structure (*.txt)
    for i in range(100):
        fname_map['rwc'+str(i).zfill(3)] = {
            'available': True,
            'structure': args.d_structure.rstrip('/')+'/RM-P'+str(i+1).zfill(3)+'.CHORUS.TXT'
        }

    # audio (.flac)
    a_fname_audio = glob.glob(args.d_audio.rstrip('/')+'/RWC-MDB-P*/*.flac')
    a_fname_audio.sort()
    for i, fname_audio in enumerate(a_fname_audio):
        fname_map['rwc'+str(i).zfill(3)]['audio'] = fname_audio
        wave, fs = sf.read(fname_audio)
        fname_map['rwc'+str(i).zfill(3)]['duration'] = wave.shape[0] / fs

    with open(args.f_map, 'w', encoding='utf-8') as f:
        json.dump(fname_map, f, ensure_ascii=False, indent=4, sort_keys=True)

    print('** done **')
