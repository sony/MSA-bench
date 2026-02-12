#! /bin/bash

METHOD=PANNs

D_DATASET=../../dataset
mkdir -p $D_DATASET/embeddings
D_EMBED=$D_DATASET/embeddings/$METHOD
mkdir -p $D_EMBED

D_AUDIO_HARMONIX=$D_DATASET/original/harmonix/audio
F_LIST_HARMONIX=$D_DATASET/list/list_harmonix_single.json
D_AUDIO_RWC=$D_DATASET/original/rwc/audio
F_LIST_RWC=$D_DATASET/list/list_rwc_single.json

D_CP=./checkpoint
mkdir -p $D_CP

# replace 'models.py'
mv audioset_tagging_cnn/pytorch/models.py audioset_tagging_cnn/pytorch/models.py.org
cp models.py audioset_tagging_cnn/pytorch/

##
## w/o pooling (Cnn14_DecisionLevelMax)
##
CHUNK_LEN_MSEC=0
CHUNK_HOP_MSEC=0
mkdir -p $D_EMBED/no_pooling

## frame-wise model "Cnn14_DecisionLevelMax"
MODEL_TYPE_CNN14_DECISIONLEVELMAX="Cnn14_DecisionLevelMax"
CP_PATH_CNN14_DECISIONLEVELMAX=$D_CP/Cnn14_DecisionLevelMax_mAP=0.385.pth
#wget -O $CP_PATH_CNN14_DECISIONLEVELMAX https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1

## harmonix
python3 m_get_embeddings.py -f_list $F_LIST_HARMONIX -d_audio $D_AUDIO_HARMONIX -d_embed $D_EMBED/no_pooling -model_type $MODEL_TYPE_CNN14_DECISIONLEVELMAX -f_checkpoint $CP_PATH_CNN14_DECISIONLEVELMAX -chunk_len_msec $CHUNK_LEN_MSEC -chunk_hop_msec $CHUNK_HOP_MSEC

## RWC
python3 m_get_embeddings.py -f_list $F_LIST_RWC -d_audio $D_AUDIO_RWC -d_embed $D_EMBED/no_pooling -model_type $MODEL_TYPE_CNN14_DECISIONLEVELMAX -f_checkpoint $CP_PATH_CNN14_DECISIONLEVELMAX -chunk_len_msec $CHUNK_LEN_MSEC -chunk_hop_msec $CHUNK_HOP_MSEC


##
## pooling
##
CHUNK_LEN_MSEC=5000
CHUNK_HOP_MSEC=500
mkdir -p $D_EMBED/pooling

## chunk processing using "Cnn14"
MODEL_TYPE_CNN14="Cnn14"
CP_PATH_CNN14=$D_CP/Cnn14_mAP=0.431.pth
#wget -O $CP_PATH_CNN14 https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1

## harmonix
python3 m_get_embeddings.py -f_list $F_LIST_HARMONIX -d_audio $D_AUDIO_HARMONIX -d_embed $D_EMBED/pooling -model_type $MODEL_TYPE_CNN14 -f_checkpoint $CP_PATH_CNN14 -chunk_len_msec $CHUNK_LEN_MSEC -chunk_hop_msec $CHUNK_HOP_MSEC -zero_padding
python3 m_get_embeddings.py -f_list $F_LIST_HARMONIX -d_audio $D_AUDIO_HARMONIX -d_embed $D_EMBED/pooling -model_type $MODEL_TYPE_CNN14_DECISIONLEVELMAX -f_checkpoint $CP_PATH_CNN14_DECISIONLEVELMAX -chunk_len_msec $CHUNK_LEN_MSEC -chunk_hop_msec $CHUNK_HOP_MSEC -zero_padding -hop_mean

## RWC
python3 m_get_embeddings.py -f_list $F_LIST_RWC -d_audio $D_AUDIO_RWC -d_embed $D_EMBED/pooling -model_type $MODEL_TYPE_CNN14 -f_checkpoint $CP_PATH_CNN14 -chunk_len_msec $CHUNK_LEN_MSEC -chunk_hop_msec $CHUNK_HOP_MSEC -zero_padding
python3 m_get_embeddings.py -f_list $F_LIST_RWC -d_audio $D_AUDIO_RWC -d_embed $D_EMBED/pooling -model_type $MODEL_TYPE_CNN14_DECISIONLEVELMAX -f_checkpoint $CP_PATH_CNN14_DECISIONLEVELMAX -chunk_len_msec $CHUNK_LEN_MSEC -chunk_hop_msec $CHUNK_HOP_MSEC -zero_padding -hop_mean
