#! python

import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f_map', help='file: fname mapping')
    parser.add_argument('-d_label', help='directory: label')
    parser.add_argument('-d_reference', help='directory: reference')
    args = parser.parse_args()

    print('** make reference data for evaluation **')
    print(' directory')
    print('  label     : '+str(args.d_label))
    print('  reference : '+str(args.d_reference))
    print(' file')
    print('  fname map  : '+str(args.f_map))

    with open(args.f_map, 'r', encoding='utf-8') as f:
        fname_map = json.load(f)

    for fname in fname_map:
        if fname_map[fname]['available'] is False:
            continue

        with open(args.d_label.rstrip('/')+'/'+fname+'_structure.json', 'r', encoding='utf-8') as f:
            a_label = json.load(f)

        a_out = []
        for label in a_label:
            if 'label_structure' in label:
                a_out.append({'onset': label['onset'], 'label': label['label_structure']})
        del a_label

        with open(args.d_reference.rstrip('/')+'/'+fname+'.lab', 'w', encoding='utf-8') as f:
            for i in range(len(a_out)-1):
                f.write(str(a_out[i]['onset'])+'\t')
                f.write(str(a_out[i+1]['onset'])+'\t')
                f.write(a_out[i]['label']+'\n')

            if fname_map[fname]['duration'] > a_out[len(a_out)-1]['onset']:
                f.write(str(a_out[len(a_out)-1]['onset'])+'\t')
                f.write(str(fname_map[fname]['duration'])+'\t')
                f.write(a_out[len(a_out)-1]['label']+'\n')

    print('** done **')
