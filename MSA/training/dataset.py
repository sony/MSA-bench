#! python

import math
import numpy as np
import torch

class dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            a_fname,
            d_embed,
            d_label,
            model_type,
            model_parameters):
        super().__init__()

        self.a_item = []

        if model_type is not None:
            ext = '_' + model_type + '.dat'
        else:
            ext = '.dat'
        for fname in a_fname:
            feature_embed_file = torch.load(d_embed+'/'+fname+ext)
            label_file = torch.load(d_label+'/'+fname+'.dat')
            # {'structure': 'name': [nframe_label], 'boundary': [nframe_label]}
            label_structure_name = label_file['structure']['name']
            label_structure_boundary = label_file['structure']['boundary']

            nframe_embed = feature_embed_file.shape[0]
            nframe_label = label_structure_name.shape[0]
            n_proc_embed = math.ceil(nframe_embed / model_parameters['Ti'])
            n_proc_label = math.ceil(nframe_label / model_parameters['Ti'])

            for i in range(n_proc_embed):
                item = {
                    'input': {
                        'embed' :torch.zeros([int(model_parameters['Ti']), model_parameters['Z']], dtype=torch.float32)
                    },
                    'label': {
                        'structure_name': torch.zeros([model_parameters['To']], dtype=torch.int8),
                        'structure_boundary': torch.zeros([model_parameters['To']], dtype=torch.bool)
                    },
                    'mask': torch.ones([model_parameters['To']], dtype=torch.bool)
                }

                # (1) input
                idx_s = int(i * model_parameters['Ti'])
                idx_e = min(idx_s+int(model_parameters['Ti']), nframe_embed)
                if idx_s < idx_e:
                    item['input']['embed'][:(idx_e-idx_s)] = feature_embed_file[idx_s:idx_e]

                # (2) label
                idx_s = i * model_parameters['To']
                idx_e = min((i+1) * model_parameters['To'], nframe_label)
                if idx_s < idx_e:
                    item['label']['structure_name'][:(idx_e-idx_s)] = label_structure_name[idx_s:idx_e]
                    item['label']['structure_boundary'][:(idx_e-idx_s)] = label_structure_boundary[idx_s:idx_e]

                # (3) mask
                if (i+1) * model_parameters['To'] >= nframe_label:
                    idx_s_mask = max(0, (nframe_label-1)-idx_s)
                    item['mask'][idx_s_mask:] = False

                # (4) collect items
                self.a_item.append(item)

        self.data_size = len(self.a_item)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.a_item[idx]


def count_data(a_fname, d_label):
    data_count = {
        'structure': {'boundary': 0, 'frame': 0}
    }
    for fname in a_fname:
        label_file = torch.load(d_label+'/'+fname+'.dat')
        data_count['structure']['frame'] += label_file['structure']['boundary'].shape[0]
        data_count['structure']['boundary'] += label_file['structure']['boundary'].sum().item()

    return data_count
