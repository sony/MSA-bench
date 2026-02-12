#! /bin/bash

D_OUT=../output
mkdir -p $D_OUT

D_DATASET=../../dataset
D_LABEL=$D_DATASET/label
D_EMBED=$D_DATASET/embeddings
D_LIST=$D_DATASET/list
F_CONFIG=../../config/config.json
N_EPOCH=100
dataset_name=(harmonix_single rwc_single)

## MusicFM
embed_method=MusicFM
hop_type=(no_pooling pooling)
model_type=(fma msd)
mkdir -p ${D_OUT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    for m in ${model_type[@]}; do
	mkdir -p ${D_OUT}/${embed_method}/$h/$m
	for f in ${dataset_name[@]}; do
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    python3 m_train.py -f_config $F_CONFIG -d_out ${D_OUT}/${embed_method}/$h/$m/$f -d_embed ${D_EMBED}/${embed_method}/$h -d_label $D_LABEL -f_list ${D_LIST}/list_${f}.json -n_div_train 1 -n_div_valid 1 -epoch $N_EPOCH -batch 8 -embed_method ${embed_method} -embed_model_type $m -hop_type $h -single
	done
    done
done

## MERT
embed_method=MERT
hop_type=(no_pooling pooling)
model_type=(95M 330M)
mkdir -p ${D_OUT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    for m in ${model_type[@]}; do
	mkdir -p ${D_OUT}/${embed_method}/$h/$m
	for f in ${dataset_name[@]}; do
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    python3 m_train.py -f_config $F_CONFIG -d_out ${D_OUT}/${embed_method}/$h/$m/$f -d_embed ${D_EMBED}/${embed_method}/$h -d_label $D_LABEL -f_list ${D_LIST}/list_${f}.json -n_div_train 1 -n_div_valid 1 -epoch $N_EPOCH -batch 8 -embed_method ${embed_method} -embed_model_type $m -hop_type $h -single
	done
    done
done

## AudioMAE (Huang)
embed_method=MetaAudioMAE
hop_type=(no_pooling pooling)
model_type=(pretrained finetuned)
mkdir -p ${D_OUT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    for m in ${model_type[@]}; do
	mkdir -p ${D_OUT}/${embed_method}/$h/$m
	for f in ${dataset_name[@]}; do
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    python3 m_train.py -f_config $F_CONFIG -d_out ${D_OUT}/${embed_method}/$h/$m/$f -d_embed ${D_EMBED}/${embed_method}/$h -d_label $D_LABEL -f_list ${D_LIST}/list_${f}.json -n_div_train 1 -n_div_valid 1 -epoch $N_EPOCH -batch 8 -embed_method ${embed_method} -embed_model_type $m -hop_type $h -single
	done
    done
done

## MULE
embed_method=MULE
hop_type=(no_pooling)
mkdir -p ${D_OUT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    for f in ${dataset_name[@]}; do
	mkdir -p ${D_OUT}/${embed_method}/$h/$f
	python3 m_train.py -f_config $F_CONFIG -d_out ${D_OUT}/${embed_method}/$h/$f -d_embed ${D_EMBED}/${embed_method}/$h -d_label $D_LABEL -f_list ${D_LIST}/list_${f}.json -n_div_train 1 -n_div_valid 1 -epoch $N_EPOCH -batch 8 -embed_method ${embed_method} -hop_type $h -single
    done
done

## EnCodec
embed_method=EnCodec
hop_type=(no_pooling pooling)
model_type=(24kHz_3kbps 24kHz_6kbps 24kHz_12kbps 24kHz_24kbps 48kHz_3kbps_stereo 48kHz_6kbps_stereo 48kHz_12kbps_stereo 48kHz_24kbps_stereo)
mkdir -p ${D_OUT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    for m in ${model_type[@]}; do
	mkdir -p ${D_OUT}/${embed_method}/$h/$m
	for f in ${dataset_name[@]}; do
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    python3 m_train.py -f_config $F_CONFIG -d_out ${D_OUT}/${embed_method}/$h/$m/$f -d_embed ${D_EMBED}/${embed_method}/$h -d_label $D_LABEL -f_list ${D_LIST}/list_${f}.json -n_div_train 1 -n_div_valid 1 -epoch $N_EPOCH -batch 8 -embed_method ${embed_method} -embed_model_type $m -hop_type $h -single
	done
    done
done

## DAC
embed_method=DAC
hop_type=(no_pooling pooling)
mkdir -p ${D_OUT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    for f in ${dataset_name[@]}; do
	mkdir -p ${D_OUT}/${embed_method}/$h/$f
	python3 m_train.py -f_config $F_CONFIG -d_out ${D_OUT}/${embed_method}/$h/$f -d_embed ${D_EMBED}/${embed_method}/$h -d_label $D_LABEL -f_list ${D_LIST}/list_${f}.json -n_div_train 1 -n_div_valid 1 -epoch $N_EPOCH -batch 8 -embed_method ${embed_method} -hop_type $h -single
    done
done

## PANNs
embed_method=PANNs
hop_type=(pooling)
model_type=(Cnn14)
mkdir -p ${D_OUT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    for m in ${model_type[@]}; do
	mkdir -p ${D_OUT}/${embed_method}/$h/$m
	for f in ${dataset_name[@]}; do
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    python3 m_train.py -f_config $F_CONFIG -d_out ${D_OUT}/${embed_method}/$h/$m/$f -d_embed ${D_EMBED}/${embed_method}/$h -d_label $D_LABEL -f_list ${D_LIST}/list_${f}.json -n_div_train 1 -n_div_valid 1 -epoch $N_EPOCH -batch 8 -embed_method ${embed_method} -embed_model_type $m -hop_type $h -single
	done
    done
done
hop_type=(no_pooling pooling)
model_type=(Cnn14_DecisionLevelMax)
mkdir -p ${D_OUT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    for m in ${model_type[@]}; do
	mkdir -p ${D_OUT}/${embed_method}/$h/$m
	for f in ${dataset_name[@]}; do
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    python3 m_train.py -f_config $F_CONFIG -d_out ${D_OUT}/${embed_method}/$h/$m/$f -d_embed ${D_EMBED}/${embed_method}/$h -d_label $D_LABEL -f_list ${D_LIST}/list_${f}.json -n_div_train 1 -n_div_valid 1 -epoch $N_EPOCH -batch 8 -embed_method ${embed_method} -embed_model_type $m -hop_type $h -single
	done
    done
done

## PaSST
embed_method=PaSST
hop_type=(no_pooling pooling)
mkdir -p ${D_OUT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    for f in ${dataset_name[@]}; do
	mkdir -p ${D_OUT}/${embed_method}/$h/$f
	python3 m_train.py -f_config $F_CONFIG -d_out ${D_OUT}/${embed_method}/$h/$f -d_embed ${D_EMBED}/${embed_method}/$h -d_label $D_LABEL -f_list ${D_LIST}/list_${f}.json -n_div_train 1 -n_div_valid 1 -epoch $N_EPOCH -batch 8 -embed_method ${embed_method} -hop_type $h -single
    done
done

## CLAP
embed_method=CLAP
hop_type=(pooling)
model_type=(music_audioset music_speech_audioset)
mkdir -p ${D_OUT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    for m in ${model_type[@]}; do
	mkdir -p ${D_OUT}/${embed_method}/$h/$m
	for f in ${dataset_name[@]}; do
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    python3 m_train.py -f_config $F_CONFIG -d_out ${D_OUT}/${embed_method}/$h/$m/$f -d_embed ${D_EMBED}/${embed_method}/$h -d_label $D_LABEL -f_list ${D_LIST}/list_${f}.json -n_div_train 1 -n_div_valid 1 -epoch $N_EPOCH -batch 8 -embed_method ${embed_method} -embed_model_type $m -hop_type $h -single
	done
    done
done

## OpenL3
embed_method=OpenL3
hop_type=(no_pooling pooling)
mkdir -p ${D_OUT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    for f in ${dataset_name[@]}; do
	mkdir -p ${D_OUT}/${embed_method}/$h/$f
	python3 m_train.py -f_config $F_CONFIG -d_out ${D_OUT}/${embed_method}/$h/$f -d_embed ${D_EMBED}/${embed_method}/$h -d_label $D_LABEL -f_list ${D_LIST}/list_${f}.json -n_div_train 1 -n_div_valid 1 -epoch $N_EPOCH -batch 8 -embed_method ${embed_method} -hop_type $h -single
    done
done
