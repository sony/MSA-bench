#! python

import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import timm_scheduler

import numpy as np
import pickle
import json
import datetime
import random

import train
import dataset
sys.path.append('..')
from model.model import *

## model functions
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


## main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f_config', help='file: config')
    parser.add_argument('-d_out', help='directory: output')
    parser.add_argument('-idx_fold', help='parameter: index of n_fold', type=int, default=0)
    parser.add_argument('-single', help='parameter: single split', action='store_true')
    parser.add_argument('-d_embed', help='directory: input embeddings')
    parser.add_argument('-d_label', help='directory: input label')
    parser.add_argument('-embed_method', help='parameter: embed method')
    parser.add_argument('-embed_model_type', help='parameter: embed model type')
    parser.add_argument('-hop_type', help='parameter: hop type')
    parser.add_argument('-f_list', help='file: corpus list')
    parser.add_argument('-n_div_train', help='parameter: number of train dataset division(1)', type=int, default=1)
    parser.add_argument('-n_div_valid', help='parameter: number of valid dataset division(1)', type=int, default=1)
    parser.add_argument('-epoch', help='parameter: number of epochs(100)', type=int, default=100)
    parser.add_argument('-resume_epoch', help='parameter: number of epoch to resume(-1)', type=int, default=-1)
    parser.add_argument('-batch', help='parameter: batch size(8)', type=int, default=8)
    parser.add_argument('-lr', help='parameter: learning rate(1e-04)', type=float, default=1e-4)
    parser.add_argument('-lr_weight_decay', help='parameter: lr weight_decay(0.01)', type=float, default=0.01)
    parser.add_argument('-lr_warmup_epochs', help='parameter: lr warmup epoch(5)', type=int, default=5)
    parser.add_argument('-lr_warmup_init', help='parameter: lr warmup initial value(1e-05)', type=float, default=1e-5)
    parser.add_argument('-lr_min', help='parameter: lr minimum value(1e-05)', type=float, default=1e-5)
    parser.add_argument('-no_calc_valid', help='parameter: no calculation for validation', action='store_true')
    parser.add_argument('-seed', help='parameter: seed value(1234)', type=int, default=1234)
    parser.add_argument('-gpu', help='parameter: GPU number', type=int, default=0)
    parser.add_argument('-num_workers', help='parameter: number of workers for dataloader(8)', type=int, default=8)
    args = parser.parse_args()

    print('** MSA: training **')
    print(' config file        : '+str(args.f_config))
    print(' output directory   : '+str(args.d_out))
    print(' model')
    print('  embed method      : '+str(args.embed_method))
    print('  embed model type  : '+str(args.embed_model_type))
    print('  hop type          : '+str(args.hop_type))
    print(' dataset')
    print('  f_list            : '+str(args.f_list))
    print('  index of n_fold   : '+str(args.idx_fold))
    print('  single split      : '+str(args.single))
    print('  directory')
    print('   embed vector     : '+str(args.d_embed))
    print('   label            : '+str(args.d_label))
    print('  n_div(train)      : '+str(args.n_div_train))
    print('  n_div(valid)      : '+str(args.n_div_valid))
    print(' training parameter')
    print('  epoch             : '+str(args.epoch))
    print('  batch             : '+str(args.batch))
    print('  learning rate     : '+str(args.lr))
    print('   weight decay     : '+str(args.lr_weight_decay))
    print('   warmup epochs    : '+str(args.lr_warmup_epochs))
    print('   warmup initial   : '+str(args.lr_warmup_init))
    print('   minimum          : '+str(args.lr_min))
    print('  seed              : '+str(args.seed))
    print('  no calc valid     : '+str(args.no_calc_valid))
    print(' GPU                : '+str(args.gpu))
    print(' num_workers        : '+str(args.num_workers))

    # (1) read config file
    print('(1) read config file')
    with open(args.f_config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # (2) torch/seed settings
    print('(2) torch/seed settings')
    print(' torch version      : '+torch.__version__)
    print(' torch cuda         : '+str(torch.cuda.is_available()))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        device = 'cuda:'+str(args.gpu)
    else:
        device = 'cpu'
    #torch.cuda.set_device(device)
    print('  device            : '+str(device))

    # (3) network settings
    print('(3) network settings')
    if args.embed_model_type is not None:
        model_parameters = config['model_parameters'][args.hop_type][args.embed_method][args.embed_model_type]
    else:
        model_parameters = config['model_parameters'][args.hop_type][args.embed_method]
    if args.hop_type == 'pooling':
        model_parameters['Ti'] = config['model_parameters'][args.hop_type]['Ti']
    model_parameters['To'] = config['model_parameters'][args.hop_type]['To']
    model_parameters['structure'] = config['model_parameters']['structure']
    if model_parameters['Ti'] == model_parameters['To']:
        downsampling = False
        model = MODEL_A(model_parameters)
    else:
        downsampling = True
        model = MODEL_B(model_parameters)
    model = model.to(device)
    model.apply(initialize_weights);
    n_parameter = count_parameters(model)
    print(' The model has {} trainable parameters'.format(n_parameter))

    # (4) training settings
    print('(4) training settings')
    with open(args.f_list, 'r', encoding='utf-8') as f:
        a_list = json.load(f)
    a_fname_train_all = []
    a_fname_valid_all = []
    if args.single:
        for fname in a_list['train']:
            a_fname_train_all.append(fname)
        for fname in a_list['validation']:
            a_fname_valid_all.append(fname)
    else:
        n_fold = len(a_list)
        for fold in a_list:
            if str(fold) == str(args.idx_fold):
                # validation
                for fname in a_list[fold]:
                    a_fname_valid_all.append(fname)
            elif str(fold) == str((int(args.idx_fold)+1) % int(n_fold)):
                # test
                continue
            else:
                # training
                for fname in a_list[fold]:
                    a_fname_train_all.append(fname)

    print('num_file(train): '+str(len(a_fname_train_all)))
    print('num_file(valid): '+str(len(a_fname_valid_all)))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = timm_scheduler.CosineLRScheduler(
        optimizer,
        t_initial=args.epoch,
        lr_min=args.lr_min,
        warmup_t=args.lr_warmup_epochs,
        warmup_lr_init=args.lr_warmup_init,
        warmup_prefix=True
    )

    data_count = dataset.count_data(a_fname_train_all, args.d_label)
    pos_weight_structure_boundary = np.round((data_count['structure']['frame'] - data_count['structure']['boundary']) / data_count['structure']['boundary'])
    criterion_structure_name = nn.CrossEntropyLoss(reduction='none')
    criterion_structure_boundary = train.BinaryCrossEntropy_with_logits(pos_weight=pos_weight_structure_boundary, reduction='none')

    # (5) save parameters
    print('(5) save parameters')
    d_out = args.d_out.rstrip('/')
    if not os.path.exists(d_out):
        os.mkdir(d_out)
    parameters = {
        'config': config,
        'model_parameters': model_parameters,
        'downsampling': downsampling,
        'parameters': n_parameter,
        'd_output': args.d_out,
        'model': {
            'embed_method': args.embed_method,
            'embed_model_type': args.embed_model_type,
            'hop_type': args.hop_type
        },
        'dataset': {
            'f_list': args.f_list,
            'idx_fold': args.idx_fold,
            'single': args.single,
            'd_embed': args.d_embed,
            'd_label': args.d_label,
            'n_div_train': args.n_div_train,
            'n_div_valid': args.n_div_valid,
            'pos_weight': {
                'structure_boundary': pos_weight_structure_boundary,
                'data_count': data_count
            }
        },
        'training': {
            'epoch': args.epoch,
            'batch': args.batch,
            'lr': args.lr,
            'lr_weight_decay': args.lr_weight_decay,
            'lr_warmup_epochs': args.lr_warmup_epochs,
            'lr_warmup_init': args.lr_warmup_init,
            'lr_min': args.lr_min,
            'seed': args.seed,
            'resume_epoch': args.resume_epoch,
            'GPU': {
                'device': device,
                'gpu': args.gpu,
                'num_workers': args.num_workers,
            }
        },
        'validation': {
            'no_calculation': args.no_calc_valid
        }
    }
    with open(d_out+'/parameter.json', 'w', encoding='utf-8') as f:
        json.dump(parameters, f, ensure_ascii=False, indent=4, sort_keys=True)

    # (6) setting for resume
    epoch_start = 0
    a_performance = []
    if args.resume_epoch >= 0:
        print('(6) resume settings')
        print(' read checkpoint  : model_'+str(args.resume_epoch).zfill(3)+'.dat')
        checkpoint = torch.load(d_out+'/model_'+str(args.resume_epoch).zfill(3)+'.dat')
        model.load_state_dict(checkpoint['model_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_dict'])
        random.setstate(checkpoint['random']['random'])
        np.random.set_state(checkpoint['random']['np_random'])
        torch.set_rng_state(checkpoint['random']['torch'])
        torch.random.set_rng_state(checkpoint['random']['torch_random'])
        torch.cuda.set_rng_state(checkpoint['random']['cuda'])
        torch.cuda.torch.cuda.set_rng_state_all(checkpoint['random']['cuda_all'])

        epoch_start = args.resume_epoch + 1
        current_epoch = checkpoint['epoch']

        with open(d_out+'/fname_valid.json', 'r', encoding='utf-8') as f:
            a_fname_valid = json.load(f)
        with open(d_out+'/performance.json', 'r', encoding='utf-8') as f:
            a_performance = json.load(f)
        print(' resume     epoch: '+str(args.resume_epoch))
        print(' checkpoint epoch: '+str(current_epoch))
        del checkpoint
    else:
        a_fname_valid = {}
        num_file_valid = len(a_fname_valid_all)
        num_file_each = int(np.floor(num_file_valid/args.n_div_valid))
        a_num_file = [num_file_each for div_valid in range(args.n_div_valid)]
        num_file_remain = num_file_valid - num_file_each * args.n_div_valid
        for i in range(num_file_remain):
            a_num_file[i] += 1

        idx_offset = 0
        for div_valid in range(args.n_div_valid):
            a_fname_valid[str(div_valid)] = []
            for i in range(a_num_file[div_valid]):
                a_fname_valid[str(div_valid)].append(a_fname_valid_all[i+idx_offset])
            idx_offset += a_num_file[div_valid]
        with open(d_out+'/fname_valid.json', 'w', encoding='utf-8') as f:
            json.dump(a_fname_valid, f, ensure_ascii=False, indent=4, sort_keys=False)

    # (7) training
    print('(7) training')
    print(' epoch_start      : '+str(epoch_start))
    for epoch in range(epoch_start, args.epoch):
        # (7-1) training
        print('(7-1) training')
        a_fname_train = {}
        num_file_train = len(a_fname_train_all)
        num_file_each = int(np.floor(num_file_train/args.n_div_train))
        a_num_file = [num_file_each for div_train in range(args.n_div_train)]
        num_file_remain = num_file_train - num_file_each * args.n_div_train
        for i in range(num_file_remain):
            a_num_file[i] += 1

        a_idx = random.sample(range(num_file_train), k=num_file_train)
        idx_offset = 0
        for div_train in range(args.n_div_train):
            a_fname_train[str(div_train)] = []
            for i in range(a_num_file[div_train]):
                a_fname_train[str(div_train)].append(a_fname_train_all[a_idx[i+idx_offset]])
            idx_offset += a_num_file[div_train]

        loss_train = 0
        num_data_train = 0
        for div_train in range(args.n_div_train):
            print('[epoch: '+str(epoch).zfill(3)+' div_train: '+str(div_train).zfill(3)+']')
            dataset_train = dataset.dataset(a_fname_train[str(div_train)],
                                            args.d_embed.rstrip('/'),
                                            args.d_label.rstrip('/'),
                                            args.embed_model_type,
                                            model_parameters)
            dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                           num_workers=args.num_workers,
                                                           batch_size=args.batch,
                                                           shuffle=True,
                                                           pin_memory=True)
            nstep_train = len(dataloader_train)
            print('## nstep train: '+str(nstep_train)+', num file : '+str(len(a_fname_train[str(div_train)])))
            retval = train.train(model, dataloader_train, optimizer,
                                         criterion_structure_name, criterion_structure_boundary,
                                         device)
            loss_train += retval[0]
            num_data_train += retval[1]
            del dataset_train, dataloader_train
        # end (for div_train in range(args.n_div_train))
        loss_train /= num_data_train

        # (7-2) validation
        loss_valid = 0
        if args.no_calc_valid is False:
            print('(7-2) validation')
            num_data_valid = 0
            for div_valid in range(args.n_div_valid):
                dataset_valid = dataset.dataset(a_fname_valid[str(div_valid)],
                                                args.d_embed.rstrip('/'),
                                                args.d_label.rstrip('/'),
                                                args.embed_model_type,
                                                model_parameters)
                dataloader_valid = torch.utils.data.DataLoader(dataset_valid,
                                                               num_workers=args.num_workers,
                                                               batch_size=args.batch,
                                                               shuffle=False,
                                                               pin_memory=True)
                nstep_valid = len(dataloader_valid)
                print('## nstep valid: '+str(nstep_valid)+', num file : '+str(len(a_fname_valid[str(div_valid)])))
                retval = train.valid(model, dataloader_valid,
                                             criterion_structure_name, criterion_structure_boundary,
                                             device)
                loss_valid += retval[0]
                num_data_valid += retval[1]
                del dataset_valid, dataloader_valid
            # end (for div_valid in range(args.n_div_valid))
            loss_valid /= num_data_valid
        else:
            nstep_valid = 0

        print('[epoch: '+str(epoch).zfill(3)+']')
        print(' loss(train) :'+str(loss_train))
        if args.no_calc_valid is False:
            print(' loss(valid) :'+str(loss_valid))

        # (7-3) scheduler update
        scheduler.step(epoch+1)

        # (7-4) save model
        with open(d_out+'/model_'+str(epoch).zfill(3)+'.pkl', 'wb') as f:
            pickle.dump(model, f, protocol=4)
        checkpoint = {
            'epoch': epoch,
            'loss_train': loss_train,
            'loss_valid': loss_valid,
            'optimizer_dict': optimizer.state_dict(),
            'scheduler_dict': scheduler.state_dict(),
            'model_dict': model.state_dict(),
            'random': {
                'torch': torch.get_rng_state(),
                'torch_random': torch.random.get_rng_state(),
                'cuda': torch.cuda.get_rng_state(),
                'cuda_all': torch.cuda.get_rng_state_all(),
                'random': random.getstate(),
                'np_random': np.random.get_state()
            },
            'model': model
        }
        torch.save(checkpoint, d_out+'/model_'+str(epoch).zfill(3)+'.dat')

        # (7-5) save performance
        performance = {
            'epoch': epoch,
            'loss_train': loss_train,
            'loss_valid': loss_valid,
            'nstep_train': nstep_train,
            'nstep_valid': nstep_valid,
            'lr': optimizer.param_groups[0]['lr'],
            'datetime': datetime.datetime.now().isoformat()
        }
        a_performance.append(performance)
        with open(d_out+'/performance.json', 'w', encoding='utf-8') as f:
            json.dump(a_performance, f, ensure_ascii=False, indent=4, sort_keys=True)

    print('** done **')
