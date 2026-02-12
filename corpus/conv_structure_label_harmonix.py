#! python

import argparse
import json
import copy

def conv_structure_label(config, label_in):
    for s1 in config['dictionary']['mapping']:
        if s1 in label_in.lower():
            return config['dictionary']['mapping'][s1]
    return 'inst'

def conv_txt2obj_harmonix(fname_structure, duration, config):
    a_structure = []
    with open(fname_structure, 'r', encoding='utf-8') as f:
        a_lines = f.readlines()
    for j in range(len(a_lines)):
        if '\t' in a_lines[j]:
            a_data = a_lines[j].rstrip('\n').split('\t')
        elif ' ' in a_lines[j]:
            a_data = a_lines[j].rstrip('\n').split(' ')
        else:
            continue
        if len(a_data) < 2:
            continue
        start_sec = float(a_data[0])
        label_structure_org = a_data[1]
        label_structure = conv_structure_label(config, label_structure_org)

        #
        # heuristic rule
        #
        # first
        if j == 0:
            if start_sec > 3.0:
                if label_structure == 'silence':
                    start_sec = 0.0
                else:
                    a_structure.append({'onset': 0.0, 'label_structure': 'intro'})
            else:
                start_sec = 0.0
        # last
        if j == len(a_lines) - 1:
            if (label_structure_org == 'end') and \
               (start_sec != duration):
                start_sec = duration

        a_structure.append({'onset': start_sec, 'label_structure': label_structure})

    return a_structure


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_label', help='directory: converted label')
    parser.add_argument('-f_map', help='file: file mapping info')
    parser.add_argument('-f_config', help='file: config')
    args = parser.parse_args()

    print('** convert structure label **')
    print(' directory')
    print('  label              : '+str(args.d_label))
    print(' file')
    print('  fname mapping info : '+str(args.f_map))
    print('  config             : '+str(args.f_config))

    with open(args.f_config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    with open(args.f_map, 'r', encoding='utf-8') as f:
        fname_map = json.load(f)
    for fname in fname_map:
        a_error = {}
        if fname_map[fname]['available'] is False:
            continue

        a_structure = conv_txt2obj_harmonix(fname_map[fname]['structure'], fname_map[fname]['duration'], config)
        '''
        # add 'silence' on top
        if (a_structure[0]['onset'] > 0.0) and \
           (a_structure[0]['label_structure'] != 'silence'):
            a_structure.insert(0, {'onset': 0.0, 'offset': a_structure[0]['onset'], 'label_structure': 'silence'})
        '''
        with open(args.d_label.rstrip('/')+'/'+fname+'_structure.json', 'w', encoding='utf-8') as f:
            json.dump(a_structure, f, ensure_ascii=False, indent=4, sort_keys=False)

    print('** done **')
