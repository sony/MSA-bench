#! /bin/bash

D_DATASET=../dataset
D_AUDIO=$D_DATASET/original/harmonix/audio
D_EMBED=$D_DATASET/embeddings
D_LABEL=$D_DATASET/label
F_CONFIG=../config/config.json

DATA=0074_djgotusfallininlove
METHOD=UMAP

python3 m_visualize.py -d_audio $D_AUDIO -d_embed $D_EMBED -d_label $D_LABEL -f_config $F_CONFIG -data $DATA -method $METHOD -fontsize 8
