#! /bin/bash

METHOD=OpenL3

D_DATASET=../../dataset
mkdir -p $D_DATASET/embeddings
D_EMBED=$D_DATASET/embeddings/$METHOD
mkdir -p $D_EMBED

D_AUDIO_HARMONIX=$D_DATASET/original/harmonix/audio
F_LIST_HARMONIX=$D_DATASET/list/list_harmonix_single.json
D_AUDIO_RWC=$D_DATASET/original/rwc/audio
F_LIST_RWC=$D_DATASET/list/list_rwc_single.json

##
## w/o pooling
##
CHUNK_LEN_MSEC=0
CHUNK_HOP_MSEC=0
mkdir -p $D_EMBED/no_pooling

## harmonix
python3 m_get_embeddings.py -f_list $F_LIST_HARMONIX -d_audio $D_AUDIO_HARMONIX -d_embed $D_EMBED/no_pooling -chunk_len_msec $CHUNK_LEN_MSEC -chunk_hop_msec $CHUNK_HOP_MSEC

## RWC
python3 m_get_embeddings.py -f_list $F_LIST_RWC -d_audio $D_AUDIO_RWC -d_embed $D_EMBED/no_pooling -chunk_len_msec $CHUNK_LEN_MSEC -chunk_hop_msec $CHUNK_HOP_MSEC

##
## pooling
##
CHUNK_LEN_MSEC=5000
CHUNK_HOP_MSEC=500
mkdir -p $D_EMBED/pooling

## harmonix
python3 m_get_embeddings.py -f_list $F_LIST_HARMONIX -d_audio $D_AUDIO_HARMONIX -d_embed $D_EMBED/pooling -chunk_len_msec $CHUNK_LEN_MSEC -chunk_hop_msec $CHUNK_HOP_MSEC -zero_padding -hop_mean

## RWC
python3 m_get_embeddings.py -f_list $F_LIST_RWC -d_audio $D_AUDIO_RWC -d_embed $D_EMBED/pooling -chunk_len_msec $CHUNK_LEN_MSEC -chunk_hop_msec $CHUNK_HOP_MSEC -zero_padding -hop_mean
