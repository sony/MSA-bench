#! /bin/bash

METHOD=MetaAudioMAE

D_DATASET=../../dataset
mkdir -p $D_DATASET/embeddings
D_EMBED=$D_DATASET/embeddings/$METHOD
mkdir -p $D_EMBED

D_AUDIO_HARMONIX=$D_DATASET/original/harmonix/audio
F_LIST_HARMONIX=$D_DATASET/list/list_harmonix_single.json
D_AUDIO_RWC=$D_DATASET/original/rwc/audio
F_LIST_RWC=$D_DATASET/list/list_rwc_single.json

# replace some files
mv /usr/local/lib/python3.10/dist-packages/timm/models/layers/helpers.py /usr/local/lib/python3.10/dist-packages/timm/models/layers/helpers.py.org
cp -p helpers.py /usr/local/lib/python3.10/dist-packages/timm/models/layers/

mv AudioMAE/models_vit.py AudioMAE/models_vit.py.org
cp -p models_vit.py AudioMAE/

##
## w/o pooling
##
CHUNK_LEN_MSEC=0
CHUNK_HOP_MSEC=0
mkdir -p $D_EMBED/no_pooling

## harmonix
python3 m_get_embeddings.py -f_list $F_LIST_HARMONIX -d_audio $D_AUDIO_HARMONIX -d_embed $D_EMBED/no_pooling -layer 11 -chunk_len_msec $CHUNK_LEN_MSEC -chunk_hop_msec $CHUNK_HOP_MSEC -model_type pretrained
python3 m_get_embeddings.py -f_list $F_LIST_HARMONIX -d_audio $D_AUDIO_HARMONIX -d_embed $D_EMBED/no_pooling -layer 11 -chunk_len_msec $CHUNK_LEN_MSEC -chunk_hop_msec $CHUNK_HOP_MSEC -model_type finetuned

## RWC
python3 m_get_embeddings.py -f_list $F_LIST_RWC -d_audio $D_AUDIO_RWC -d_embed $D_EMBED/no_pooling -layer 11 -chunk_len_msec $CHUNK_LEN_MSEC -chunk_hop_msec $CHUNK_HOP_MSEC -model_type pretrained
python3 m_get_embeddings.py -f_list $F_LIST_RWC -d_audio $D_AUDIO_RWC -d_embed $D_EMBED/no_pooling -layer 11 -chunk_len_msec $CHUNK_LEN_MSEC -chunk_hop_msec $CHUNK_HOP_MSEC -model_type finetuned

##
## pooling
##
CHUNK_LEN_MSEC=10240
CHUNK_HOP_MSEC=500
mkdir -p $D_EMBED/pooling

## harmonix
python3 m_get_embeddings.py -f_list $F_LIST_HARMONIX -d_audio $D_AUDIO_HARMONIX -d_embed $D_EMBED/pooling -layer 11 -chunk_len_msec $CHUNK_LEN_MSEC -chunk_hop_msec $CHUNK_HOP_MSEC -model_type finetuned -zero_padding -hop_mean -halfB
python3 m_get_embeddings.py -f_list $F_LIST_HARMONIX -d_audio $D_AUDIO_HARMONIX -d_embed $D_EMBED/pooling -layer 11 -chunk_len_msec $CHUNK_LEN_MSEC -chunk_hop_msec $CHUNK_HOP_MSEC -model_type pretrained -zero_padding -hop_mean -halfC

## RWC
python3 m_get_embeddings.py -f_list $F_LIST_RWC -d_audio $D_AUDIO_RWC -d_embed $D_EMBED/pooling -layer 11 -chunk_len_msec $CHUNK_LEN_MSEC -chunk_hop_msec $CHUNK_HOP_MSEC -model_type finetuned -zero_padding -hop_mean -halfB
python3 m_get_embeddings.py -f_list $F_LIST_RWC -d_audio $D_AUDIO_RWC -d_embed $D_EMBED/pooling -layer 11 -chunk_len_msec $CHUNK_LEN_MSEC -chunk_hop_msec $CHUNK_HOP_MSEC -model_type pretrained -zero_padding -hop_mean -halfC
