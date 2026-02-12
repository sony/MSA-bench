#! python

import os
import argparse
import json
import csv
import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_audio', help='directory: audio data', required=True)
    parser.add_argument('-f_csv', help='file: csv', required=True)
    parser.add_argument('-f_meta', help='file: metadata', required=True)
    parser.add_argument('-f_info', help='file: info', required=True)
    args = parser.parse_args()

    print('** make harmonix information file **')
    print(' directory')
    print('  audio    : '+str(args.d_audio))
    print(' file')
    print('  info     : '+str(args.f_info))
    print('  csv      : '+str(args.f_csv))
    print('  metadata : '+str(args.f_meta))

    # csv2info
    a_info_tmp = {}
    with open(args.f_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            fname = row[0]
            url = row[1]
            fname_wav = url.split('?v=')[1]
            fname_number = fname.split('_')[0]
            a_info_tmp[fname] = {'download': {'url': url, 'wav': fname_wav, 'fname': fname}}

    a_fname_audio_tmp = os.listdir(args.d_audio.rstrip('/'))
    a_fname_audio = []
    for fname in a_fname_audio_tmp:
        if fname.endswith('.wav') and \
           ((len(fname.split('_')) > 1) and fname.split('_')[0].isdigit()):
            a_fname_audio.append(fname[:-4])
    a_fname_audio.sort()

    a_info = {}
    for fname in a_info_tmp:
        fname_number = fname.split('_')[0]
        if fname+'.wav' in a_fname_audio:
            a_info[fname] = copy.deepcopy(a_info_tmp[fname])
            a_info[fname]['available'] = True
        else:
            flag = False
            for fname_audio in a_fname_audio:
                if fname_audio.split('_')[0] == fname_number:
                    a_info[fname_audio] = copy.deepcopy(a_info_tmp[fname])
                    a_info[fname_audio]['available'] = True
                    flag = True
                    break
            if flag is False:
                a_info[fname] = copy.deepcopy(a_info_tmp[fname])
                a_info[fname]['available'] = False
    del a_info_tmp

    a_check = {}
    for fname in a_info:
        tmp = a_info[fname]['download']['wav'].split(' (')
        if (tmp[0] in a_check) is False:
            a_check[tmp[0]] = []
        a_check[tmp[0]].append(fname)
        if len(a_check[tmp[0]]) > 1:
            a_info[fname]['duplicate'] = a_check[tmp[0]][0]

        if len(tmp) > 1:
            a_info[fname]['remarks'] = tmp[1][:-1]

    a_meta = {}
    with open(args.f_meta, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            a_meta[row[0]] = {}
            for i, attr in enumerate(header):
                if attr == 'File':
                    continue
                a_meta[row[0]][attr] = row[i]

    a_out = {}
    for fname in a_meta:
        for fname_info in a_info:
            if a_info[fname_info]['download']['fname'] == fname:
                a_info[fname_info]['metadata'] = a_meta[fname]
                break

    with open(args.f_info, 'w', encoding='utf-8') as f:
        json.dump(a_info, f, ensure_ascii=False, indent=4, sort_keys=False)

    print('** done **')
