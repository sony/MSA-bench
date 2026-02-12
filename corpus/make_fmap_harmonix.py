#! python

import argparse
import os
import json
import soundfile as sf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_structure', help='directory: structure annotation')
    parser.add_argument('-d_audio', help='directory: audio data')
    parser.add_argument('-f_info', help='file: harmonix info')
    parser.add_argument('-f_map', help='file: file mapping info')
    args = parser.parse_args()

    print('** make file map for harmonix dataset **')
    print(' directory')
    print('  audio         : '+str(args.d_audio))
    print('  structure     : '+str(args.d_structure))
    print(' file')
    print('  harmonix info : '+str(args.f_info))
    print('  mapping info  : '+str(args.f_map))

    fname_map = {}
    with open(args.f_info, 'r', encoding='utf-8') as f:
        a_info = json.load(f)

    a_fname_structure = os.listdir(args.d_structure)
    a_fname_audio = os.listdir(args.d_audio)

    n = 0
    for fname in a_info:
        fname_map[fname] = {
            'metadata': a_info[fname]['metadata'],
            'structure': None,
            'audio': None
        }
        fname_num = fname.split('_')[0]
        
        for fname_structure in a_fname_structure:
            if fname_structure.startswith(fname_num) and \
               fname_structure.endswith('.txt'):
                fname_map[fname]['structure'] = args.d_structure.rstrip('/')+'/'+fname_structure
                break
        for fname_audio in a_fname_audio:
            if fname_audio.startswith(fname_num) and \
               fname_audio.endswith('.wav'):
                fname_map[fname]['audio'] = args.d_audio.rstrip('/')+'/'+fname_audio
                wave, fs = sf.read(args.d_audio.rstrip('/')+'/'+fname_audio)
                fname_map[fname]['duration'] = wave.shape[0] / fs
                break
        if (fname_map[fname]['structure'] is None) or \
           (fname_map[fname]['audio'] is None):
            fname_map[fname]['available'] = False
            print(fname_map[fname])
        else:
            fname_map[fname]['available'] = True
            n += 1

    # re-check duplication
    a_duplication = {}
    for fname in fname_map:
        url = a_info[fname]['download']['url']
        if (url in a_duplication) is False:
            a_duplication[url] = []
        a_duplication[url].append({'name': fname, 'duration': float(fname_map[fname]['metadata']['Duration'])})
    for url in a_duplication:
        if len(a_duplication[url]) > 1:
            a_duplication[url] = sorted(a_duplication[url], key=lambda x: x['duration'], reverse=True)
            remark = ''
            for i in range(len(a_duplication[url])):
                if a_duplication[url][i]['name'].startswith('0644') or \
                   a_duplication[url][i]['name'].startswith('0769'):
                    remark = 'different song'
                elif a_duplication[url][i]['name'].startswith('0245') or \
                     a_duplication[url][i]['name'].startswith('0894') or \
                     a_duplication[url][i]['name'].startswith('0965') or \
                     a_duplication[url][i]['name'].startswith('0966'):
                    remark = 'different mix'

            for i in range(len(a_duplication[url])):
                fname_map[a_duplication[url][i]['name']]['info'] = {
                    'duplicate': {
                        'group': a_duplication[url],
                        'duration_order': i
                    }
                }
                if remark != '':
                    fname_map[a_duplication[url][i]['name']]['info']['duplicate']['remark'] = remark

    with open(args.f_map, 'w', encoding='utf-8') as f:
        json.dump(fname_map, f, ensure_ascii=False, indent=4, sort_keys=True)

    print(str(n)+' / '+str(len(a_info))+' data are available')
    print('** done **')
