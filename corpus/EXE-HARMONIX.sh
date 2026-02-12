#! /bin/bash

# output corpus
D_CORPUS=../dataset
D_LIST=$D_CORPUS/list
D_LABEL=$D_CORPUS/label
D_REFERENCE=$D_CORPUS/reference
F_MAP=$D_CORPUS/fmap_harmonix.json

# original dataset
D_HARMONIX=$D_CORPUS/original/harmonix
D_HARMONIX_LABEL=$D_HARMONIX/harmonixset/dataset/segments
D_HARMONIX_AUDIO=$D_HARMONIX/audio

# files
F_CONFIG=../config/config.json

# (0) prepare harmonix dataset
mkdir -p $D_CORPUS
mkdir -p $D_HARMONIX
cd $D_HARMONIX
git clone https://github.com/urinieto/harmonixset
cd -
## 1. download audio files by yourself into $D_HARMONIX_AUDIO.
## 2. If the downloaded files are in mp3 format, convert wav
## 3. The file name should be the first column in youtube_urls.csv, such as 0001_12step.wav instead of iBHNgV6_znU.wav

# (1) make file map
mkdir -p tmp
python3 make_info_harmonix.py -f_csv $D_HARMONIX/harmonixset/dataset/youtube_urls.csv -f_meta $D_HARMONIX/harmonixset/dataset/metadata.csv -d_audio $D_HARMONIX_AUDIO -f_info tmp/info_harmonix.json
python3 make_fmap_harmonix.py -d_structure $D_HARMONIX_LABEL -d_audio $D_HARMONIX_AUDIO -f_map $F_MAP -f_info tmp/info_harmonix.json

# (2) make list
mkdir -p $D_LIST
python3 make_list_harmonix.py -d_out $D_LIST -f_map $F_MAP

# (3) convert structure label
mkdir -p $D_LABEL
python3 conv_structure_label_harmonix.py -d_label $D_LABEL -f_map $F_MAP -f_config $F_CONFIG

# (4) make label for training/evaluation
python3 conv_label.py -d_label $D_LABEL -f_map $F_MAP -f_config $F_CONFIG

# (5) make reference for evaluation
mkdir -p $D_REFERENCE
python3 make_reference.py -d_label $D_LABEL -d_reference $D_REFERENCE -f_map $F_MAP
