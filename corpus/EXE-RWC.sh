#! /bin/bash

# output corpus
D_CORPUS=../dataset
D_LIST=$D_CORPUS/list
D_LABEL=$D_CORPUS/label
D_REFERENCE=$D_CORPUS/reference
F_MAP=$D_CORPUS/fmap_rwc.json

# original dataset
D_RWC=$D_CORPUS/original/rwc
D_RWC_LABEL=$D_RWC/annotation
D_RWC_CD=$D_RWC/CD
D_RWC_AUDIO=$D_RWC/audio

# files
F_CONFIG=../config/config.json
F_ERROR=annotation_fix/label_error_rwc.json

# (0) prepare RWC dataset
mkdir -p $D_CORPUS
mkdir -p $D_RWC
## 1. copy CHORUS annotation data to $D_RWC_LABEL
## 2. rip CD data in flac format, then copy to $D_RWC_CD

# (1) make file map
python3 make_fmap_rwc.py -d_structure $D_RWC_LABEL -d_audio $D_RWC_CD -f_map $F_MAP

# (2) convert flac to wav (RWC)
mkdir -p $D_RWC_AUDIO
python3 conv_flac2wav.py -d_audio $D_RWC_AUDIO -f_map $F_MAP

# (3) make list
mkdir -p $D_LIST
python3 make_list_rwc.py -d_out $D_LIST -f_map $F_MAP

# (4) convert structure label
mkdir -p $D_LABEL
python3 conv_structure_label_rwc.py -d_label $D_LABEL -f_map $F_MAP -f_config $F_CONFIG -f_error $F_ERROR

# (5) make label for training/evaluation
python3 conv_label.py -d_label $D_LABEL -f_map $F_MAP -f_config $F_CONFIG

# (6) make reference for evaluation
mkdir -p $D_REFERENCE
python3 make_reference.py -d_label $D_LABEL -d_reference $D_REFERENCE -f_map $F_MAP
