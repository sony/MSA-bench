#! python

import argparse
import json
import copy

def conv_structure_label(config, label_in):
    for s1 in config['dictionary']['mapping']:
        if s1 in label_in.lower():
            return config['dictionary']['mapping'][s1]
    return 'inst'

def conv_txt2obj_rwc(fname_structure, duration, config):
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
        if len(a_data) < 3:
            continue
        start_sec = int(a_data[0]) * 0.01
        #end_sec = int(a_data[1]) * 0.01
        label_structure_org = a_data[2]
        label_structure = conv_structure_label(config, label_structure_org)

        #a_structure.append({'onset': start_sec, 'offset': end_sec, 'label_structure': label_structure})
        a_structure.append({'onset': start_sec, 'label_structure': label_structure})

    ##
    ## heuristic rule
    ##
    # first
    if (a_structure[0]['onset'] != 0) and (a_structure[0]['label_structure'] == 'intro'):
        a_structure[0]['onset'] = 0.0
    # last
    if a_structure[len(a_structure)-1]['onset'] < duration:
        a_structure.append({'onset': duration, 'label_structure': conv_structure_label(config, 'end')})
    '''
    elif a_structure[len(a_structure)-1]['onset'] > duration:
        # no data
        print(a_structure[len(a_structure)-1])
    '''
    return a_structure


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_label', help='directory: converted label')
    parser.add_argument('-f_map', help='file: file mapping info')
    parser.add_argument('-f_config', help='file: config')
    parser.add_argument('-f_error', help='file: error correction')
    args = parser.parse_args()

    print('** convert structure label **')
    print(' directory')
    print('  label              : '+str(args.d_label))
    print(' file')
    print('  fname mapping info : '+str(args.f_map))
    print('  config             : '+str(args.f_config))
    print('  error correction   : '+str(args.f_error))

    with open(args.f_config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    with open(args.f_error, 'r', encoding='utf-8') as f:
        a_error_correction = json.load(f)
    with open(args.f_map, 'r', encoding='utf-8') as f:
        fname_map = json.load(f)
    for fname in fname_map:
        a_error = {}
        if fname_map[fname]['available'] is False:
            continue

        # convert .txt to .json
        a_structure = conv_txt2obj_rwc(fname_map[fname]['structure'], fname_map[fname]['duration'], config)
        '''
        # add 'silence' on top
        if (a_structure[0]['onset'] > 0.0) and \
           (a_structure[0]['label_structure'] != 'silence'):
            a_structure.insert(0, {'onset': 0.0, 'offset': a_structure[0]['onset'], 'label_structure': 'silence'})
        '''
        # annotation error correction
        if (fname in a_error_correction) and ('structure' in a_error_correction[fname]):
            a_structure_tmp = copy.deepcopy(a_structure)
            del a_structure
            for error_structure in a_error_correction[fname]['structure']:
                if error_structure['org'] == None:
                    # add new label
                    a_structure_tmp.append({'onset': error_structure['fix']['onset'], 'label_structure': error_structure['fix']['label']})
                elif error_structure['fix'] == None:
                    for i in range(len(a_structure_tmp)):
                        if a_structure_tmp[i]['onset'] == error_structure['org']['onset']:
                            a_structure_tmp[i]['label_structure'] = 'remove'
                            break
                else:
                    if error_structure['org']['onset'] != error_structure['fix']['onset']:
                        for i in range(len(a_structure_tmp)):
                            if a_structure_tmp[i]['onset'] == error_structure['org']['onset']:
                                a_structure_tmp[i]['onset'] = error_structure['fix']['onset']
                                break
                    else:
                        for i in range(len(a_structure_tmp)):
                            if a_structure_tmp[i]['onset'] == error_structure['org']['onset']:
                                a_structure_tmp[i]['label_structure'] = error_structure['fix']['label']
                                break
            a_structure_tmp = sorted(a_structure_tmp, key=lambda x: x['onset'])
            a_structure = []
            for i in range(len(a_structure_tmp)):
                if a_structure_tmp[i]['label_structure'] != 'remove':
                    a_structure.append(a_structure_tmp[i])
            del a_structure_tmp

        with open(args.d_label.rstrip('/')+'/'+fname+'_structure.json', 'w', encoding='utf-8') as f:
            json.dump(a_structure, f, ensure_ascii=False, indent=4, sort_keys=False)

    print('** done **')
