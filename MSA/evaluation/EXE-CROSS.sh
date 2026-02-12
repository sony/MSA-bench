#! /bin/bash

D_OUT=../output
D_RESULT=../result
mkdir -p $D_RESULT

D_DATASET=../../dataset
D_LABEL=$D_DATASET/label
D_EMBED=$D_DATASET/embeddings
D_REFERENCE=$D_DATASET/reference
D_LIST=$D_DATASET/list
F_CONFIG=../../config/config.json

dataset_name=(harmonix_single rwc_single)
num=`seq -w 000 099`

## MusicFM
embed_method=MusicFM
hop_type=(no_pooling pooling)
model_type=(msd fma)
mkdir -p ${D_RESULT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    mkdir -p ${D_RESULT}/${embed_method}/$h
    echo $h
    for m in ${model_type[@]}; do
	mkdir -p ${D_OUT}/${embed_method}/$h/$m
	mkdir -p ${D_RESULT}/${embed_method}/$h/$m
	echo $h/$m
	for f in ${dataset_name[@]}; do
	    echo $h/$m/$f
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f

	    # evaluation
	    for n in ${num[@]}; do
		mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f/$n
		# inference
		python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list ${D_LIST}/list_${f}.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/$n -split validation

		# mir_eval
		python3 m_eval_single.py -f_list ${D_LIST}/list_${f}.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/"$n" -split validation -f_config $F_CONFIG -d_label $D_LABEL
	    done
	    python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$m/$f
	done

	## test: RWC, trained model: harmonix
	# inference
	python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/harmonix_single/parameter.json -f_list ${D_LIST}/list_rwc_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/harmonix_single -d_result ${D_RESULT}/${embed_method}/$h/$m/harmonix_single -split test
	# mir_eval
	python3 m_eval_single.py -f_list ${D_LIST}/list_rwc_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/harmonix_single -split test -f_config $F_CONFIG -d_label $D_LABEL

	## test: harmonix, trained model: RWC
	# inference
	python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/rwc_single/parameter.json -f_list ${D_LIST}/list_harmonix_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/rwc_single -d_result ${D_RESULT}/${embed_method}/$h/$m/rwc_single -split test
	# mir_eval
	python3 m_eval_single.py -f_list ${D_LIST}/list_harmonix_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/rwc_single -split test -f_config $F_CONFIG -d_label $D_LABEL
    done
done

## MERT
embed_method=MERT
hop_type=(no_pooling pooling)
model_type=(95M 330M)
mkdir -p ${D_RESULT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    mkdir -p ${D_RESULT}/${embed_method}/$h
    echo $h
    for m in ${model_type[@]}; do
	mkdir -p ${D_OUT}/${embed_method}/$h/$m
	mkdir -p ${D_RESULT}/${embed_method}/$h/$m
	echo $h/$m
	for f in ${dataset_name[@]}; do
	    echo $h/$m/$f
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f

	    # evaluation
	    for n in ${num[@]}; do
		mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f/$n
		# inference
		python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list ${D_LIST}/list_${f}.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/$n -split validation

		# mir_eval
		python3 m_eval_single.py -f_list ${D_LIST}/list_${f}.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/"$n" -split validation -f_config $F_CONFIG -d_label $D_LABEL
	    done
	    python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$m/$f
	done

	## test: RWC, trained model: harmonix
	# inference
	python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/harmonix_single/parameter.json -f_list ${D_LIST}/list_rwc_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/harmonix_single -d_result ${D_RESULT}/${embed_method}/$h/$m/harmonix_single -split test
	# mir_eval
	python3 m_eval_single.py -f_list ${D_LIST}/list_rwc_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/harmonix_single -split test -f_config $F_CONFIG -d_label $D_LABEL

	## test: harmonix, trained model: RWC
	# inference
	python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/rwc_single/parameter.json -f_list ${D_LIST}/list_harmonix_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/rwc_single -d_result ${D_RESULT}/${embed_method}/$h/$m/rwc_single -split test
	# mir_eval
	python3 m_eval_single.py -f_list ${D_LIST}/list_harmonix_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/rwc_single -split test -f_config $F_CONFIG -d_label $D_LABEL
    done
done

## AudioMAE (Huang)
embed_method=MetaAudioMAE
hop_type=(no_pooling pooling)
model_type=(pretrained finetuned)
mkdir -p ${D_RESULT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    mkdir -p ${D_RESULT}/${embed_method}/$h
    echo $h
    for m in ${model_type[@]}; do
	mkdir -p ${D_OUT}/${embed_method}/$h/$m
	mkdir -p ${D_RESULT}/${embed_method}/$h/$m
	echo $h/$m
	for f in ${dataset_name[@]}; do
	    echo $h/$m/$f
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f

	    # evaluation
	    for n in ${num[@]}; do
		mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f/$n
		# inference
		python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list ${D_LIST}/list_${f}.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/$n -split validation

		# mir_eval
		python3 m_eval_single.py -f_list ${D_LIST}/list_${f}.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/"$n" -split validation -f_config $F_CONFIG -d_label $D_LABEL
	    done
	    python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$m/$f
	done

	## test: RWC, trained model: harmonix
	# inference
	python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/harmonix_single/parameter.json -f_list ${D_LIST}/list_rwc_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/harmonix_single -d_result ${D_RESULT}/${embed_method}/$h/$m/harmonix_single -split test
	# mir_eval
	python3 m_eval_single.py -f_list ${D_LIST}/list_rwc_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/harmonix_single -split test -f_config $F_CONFIG -d_label $D_LABEL

	## test: harmonix, trained model: RWC
	# inference
	python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/rwc_single/parameter.json -f_list ${D_LIST}/list_harmonix_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/rwc_single -d_result ${D_RESULT}/${embed_method}/$h/$m/rwc_single -split test
	# mir_eval
	python3 m_eval_single.py -f_list ${D_LIST}/list_harmonix_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/rwc_single -split test -f_config $F_CONFIG -d_label $D_LABEL
    done
done

## MULE
embed_method=MULE
hop_type=(no_pooling)
mkdir -p ${D_RESULT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    mkdir -p ${D_RESULT}/${embed_method}/$h
    echo $h

    for f in ${dataset_name[@]}; do
	echo $h/$f
	mkdir -p ${D_OUT}/${embed_method}/$h/$f
	mkdir -p ${D_RESULT}/${embed_method}/$h/$f

	# evaluation
	for n in ${num[@]}; do
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$f/$n
	    # inference
	    python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$f/parameter.json -f_list ${D_LIST}/list_${f}.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$f/$n -split validation

	    # mir_eval
	    python3 m_eval_single.py -f_list ${D_LIST}/list_${f}.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$f/"$n" -split validation -f_config $F_CONFIG -d_label $D_LABEL
	done
	python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$f
    done

    ## test: RWC, trained model: harmonix
    # inference
    python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/harmonix_single/parameter.json -f_list ${D_LIST}/list_rwc_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/harmonix_single -d_result ${D_RESULT}/${embed_method}/$h/harmonix_single -split test
    # mir_eval
    python3 m_eval_single.py -f_list ${D_LIST}/list_rwc_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/harmonix_single -split test -f_config $F_CONFIG -d_label $D_LABEL

    ## test: harmonix, trained model: RWC
    # inference
    python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/rwc_single/parameter.json -f_list ${D_LIST}/list_harmonix_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/rwc_single -d_result ${D_RESULT}/${embed_method}/$h/rwc_single -split test
    # mir_eval
    python3 m_eval_single.py -f_list ${D_LIST}/list_harmonix_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/rwc_single -split test -f_config $F_CONFIG -d_label $D_LABEL
done

## EnCodec
embed_method=EnCodec
hop_type=(no_pooling pooling)
model_type=(24kHz_3kbps 24kHz_6kbps 24kHz_12kbps 24kHz_24kbps 48kHz_3kbps_stereo 48kHz_6kbps_stereo 48kHz_12kbps_stereo 48kHz_24kbps_stereo)
mkdir -p ${D_RESULT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    mkdir -p ${D_RESULT}/${embed_method}/$h
    echo $h
    for m in ${model_type[@]}; do
	mkdir -p ${D_OUT}/${embed_method}/$h/$m
	mkdir -p ${D_RESULT}/${embed_method}/$h/$m
	echo $h/$m
	for f in ${dataset_name[@]}; do
	    echo $h/$m/$f
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f

	    # evaluation
	    for n in ${num[@]}; do
		mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f/$n
		# inference
		python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list ${D_LIST}/list_${f}.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/$n -split validation

		# mir_eval
		python3 m_eval_single.py -f_list ${D_LIST}/list_${f}.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/"$n" -split validation -f_config $F_CONFIG -d_label $D_LABEL
	    done
	    python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$m/$f
	done

	## test: RWC, trained model: harmonix
	# inference
	python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/harmonix_single/parameter.json -f_list ${D_LIST}/list_rwc_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/harmonix_single -d_result ${D_RESULT}/${embed_method}/$h/$m/harmonix_single -split test
	# mir_eval
	python3 m_eval_single.py -f_list ${D_LIST}/list_rwc_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/harmonix_single -split test -f_config $F_CONFIG -d_label $D_LABEL

	## test: harmonix, trained model: RWC
	# inference
	python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/rwc_single/parameter.json -f_list ${D_LIST}/list_harmonix_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/rwc_single -d_result ${D_RESULT}/${embed_method}/$h/$m/rwc_single -split test
	# mir_eval
	python3 m_eval_single.py -f_list ${D_LIST}/list_harmonix_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/rwc_single -split test -f_config $F_CONFIG -d_label $D_LABEL
    done
done

## DAC
embed_method=DAC
hop_type=(no_pooling pooling)
mkdir -p ${D_RESULT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    mkdir -p ${D_RESULT}/${embed_method}/$h
    echo $h

    for f in ${dataset_name[@]}; do
	echo $h/$f
	mkdir -p ${D_OUT}/${embed_method}/$h/$f
	mkdir -p ${D_RESULT}/${embed_method}/$h/$f

	# evaluation
	for n in ${num[@]}; do
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$f/$n
	    # inference
	    python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$f/parameter.json -f_list ${D_LIST}/list_${f}.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$f/$n -split validation

	    # mir_eval
	    python3 m_eval_single.py -f_list ${D_LIST}/list_${f}.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$f/"$n" -split validation -f_config $F_CONFIG -d_label $D_LABEL
	done
	python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$f
    done

    ## test: RWC, trained model: harmonix
    # inference
    python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/harmonix_single/parameter.json -f_list ${D_LIST}/list_rwc_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/harmonix_single -d_result ${D_RESULT}/${embed_method}/$h/harmonix_single -split test
    # mir_eval
    python3 m_eval_single.py -f_list ${D_LIST}/list_rwc_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/harmonix_single -split test -f_config $F_CONFIG -d_label $D_LABEL

    ## test: harmonix, trained model: RWC
    # inference
    python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/rwc_single/parameter.json -f_list ${D_LIST}/list_harmonix_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/rwc_single -d_result ${D_RESULT}/${embed_method}/$h/rwc_single -split test
    # mir_eval
    python3 m_eval_single.py -f_list ${D_LIST}/list_harmonix_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/rwc_single -split test -f_config $F_CONFIG -d_label $D_LABEL
done

## PANNs
embed_method=PANNs
hop_type=(pooling)
model_type=(Cnn14)
mkdir -p ${D_RESULT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    mkdir -p ${D_RESULT}/${embed_method}/$h
    echo $h
    for m in ${model_type[@]}; do
	mkdir -p ${D_OUT}/${embed_method}/$h/$m
	mkdir -p ${D_RESULT}/${embed_method}/$h/$m
	echo $h/$m
	for f in ${dataset_name[@]}; do
	    echo $h/$m/$f
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f

	    # evaluation
	    for n in ${num[@]}; do
		mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f/$n
		# inference
		python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list ${D_LIST}/list_${f}.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/$n -split validation

		# mir_eval
		python3 m_eval_single.py -f_list ${D_LIST}/list_${f}.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/"$n" -split validation -f_config $F_CONFIG -d_label $D_LABEL
	    done
	    python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$m/$f
	done

	## test: RWC, trained model: harmonix
	# inference
	python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/harmonix_single/parameter.json -f_list ${D_LIST}/list_rwc_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/harmonix_single -d_result ${D_RESULT}/${embed_method}/$h/$m/harmonix_single -split test
	# mir_eval
	python3 m_eval_single.py -f_list ${D_LIST}/list_rwc_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/harmonix_single -split test -f_config $F_CONFIG -d_label $D_LABEL

	## test: harmonix, trained model: RWC
	# inference
	python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/rwc_single/parameter.json -f_list ${D_LIST}/list_harmonix_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/rwc_single -d_result ${D_RESULT}/${embed_method}/$h/$m/rwc_single -split test
	# mir_eval
	python3 m_eval_single.py -f_list ${D_LIST}/list_harmonix_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/rwc_single -split test -f_config $F_CONFIG -d_label $D_LABEL
    done
done
hop_type=(no_pooling pooling)
model_type=(Cnn14_DecisionLevelMax)
mkdir -p ${D_RESULT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    mkdir -p ${D_RESULT}/${embed_method}/$h
    echo $h
    for m in ${model_type[@]}; do
	mkdir -p ${D_OUT}/${embed_method}/$h/$m
	mkdir -p ${D_RESULT}/${embed_method}/$h/$m
	echo $h/$m
	for f in ${dataset_name[@]}; do
	    echo $h/$m/$f
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f

	    # evaluation
	    for n in ${num[@]}; do
		mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f/$n
		# inference
		python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list ${D_LIST}/list_${f}.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/$n -split validation

		# mir_eval
		python3 m_eval_single.py -f_list ${D_LIST}/list_${f}.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/"$n" -split validation -f_config $F_CONFIG -d_label $D_LABEL
	    done
	    python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$m/$f
	done

	## test: RWC, trained model: harmonix
	# inference
	python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/harmonix_single/parameter.json -f_list ${D_LIST}/list_rwc_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/harmonix_single -d_result ${D_RESULT}/${embed_method}/$h/$m/harmonix_single -split test
	# mir_eval
	python3 m_eval_single.py -f_list ${D_LIST}/list_rwc_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/harmonix_single -split test -f_config $F_CONFIG -d_label $D_LABEL

	## test: harmonix, trained model: RWC
	# inference
	python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/rwc_single/parameter.json -f_list ${D_LIST}/list_harmonix_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/rwc_single -d_result ${D_RESULT}/${embed_method}/$h/$m/rwc_single -split test
	# mir_eval
	python3 m_eval_single.py -f_list ${D_LIST}/list_harmonix_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/rwc_single -split test -f_config $F_CONFIG -d_label $D_LABEL
    done
done

## PaSST
embed_method=PaSST
hop_type=(no_pooling pooling)
mkdir -p ${D_RESULT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    mkdir -p ${D_RESULT}/${embed_method}/$h
    echo $h

    for f in ${dataset_name[@]}; do
	echo $h/$f
	mkdir -p ${D_OUT}/${embed_method}/$h/$f
	mkdir -p ${D_RESULT}/${embed_method}/$h/$f

	# evaluation
	for n in ${num[@]}; do
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$f/$n
	    # inference
	    python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$f/parameter.json -f_list ${D_LIST}/list_${f}.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$f/$n -split validation

	    # mir_eval
	    python3 m_eval_single.py -f_list ${D_LIST}/list_${f}.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$f/"$n" -split validation -f_config $F_CONFIG -d_label $D_LABEL
	done
	python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$f
    done

    ## test: RWC, trained model: harmonix
    # inference
    python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/harmonix_single/parameter.json -f_list ${D_LIST}/list_rwc_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/harmonix_single -d_result ${D_RESULT}/${embed_method}/$h/harmonix_single -split test
    # mir_eval
    python3 m_eval_single.py -f_list ${D_LIST}/list_rwc_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/harmonix_single -split test -f_config $F_CONFIG -d_label $D_LABEL

    ## test: harmonix, trained model: RWC
    # inference
    python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/rwc_single/parameter.json -f_list ${D_LIST}/list_harmonix_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/rwc_single -d_result ${D_RESULT}/${embed_method}/$h/rwc_single -split test
    # mir_eval
    python3 m_eval_single.py -f_list ${D_LIST}/list_harmonix_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/rwc_single -split test -f_config $F_CONFIG -d_label $D_LABEL
done

## CLAP
embed_method=CLAP
hop_type=(pooling)
model_type=(music_audioset music_speech_audioset)
mkdir -p ${D_RESULT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    mkdir -p ${D_RESULT}/${embed_method}/$h
    echo $h
    for m in ${model_type[@]}; do
	mkdir -p ${D_OUT}/${embed_method}/$h/$m
	mkdir -p ${D_RESULT}/${embed_method}/$h/$m
	echo $h/$m
	for f in ${dataset_name[@]}; do
	    echo $h/$m/$f
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f

	    # evaluation
	    for n in ${num[@]}; do
		mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f/$n
		# inference
		python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list ${D_LIST}/list_${f}.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/$n -split validation

		# mir_eval
		python3 m_eval_single.py -f_list ${D_LIST}/list_${f}.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/"$n" -split validation -f_config $F_CONFIG -d_label $D_LABEL
	    done
	    python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$m/$f
	done

	## test: RWC, trained model: harmonix
	# inference
	python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/harmonix_single/parameter.json -f_list ${D_LIST}/list_rwc_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/harmonix_single -d_result ${D_RESULT}/${embed_method}/$h/$m/harmonix_single -split test
	# mir_eval
	python3 m_eval_single.py -f_list ${D_LIST}/list_rwc_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/harmonix_single -split test -f_config $F_CONFIG -d_label $D_LABEL

	## test: harmonix, trained model: RWC
	# inference
	python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/rwc_single/parameter.json -f_list ${D_LIST}/list_harmonix_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/rwc_single -d_result ${D_RESULT}/${embed_method}/$h/$m/rwc_single -split test
	# mir_eval
	python3 m_eval_single.py -f_list ${D_LIST}/list_harmonix_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/rwc_single -split test -f_config $F_CONFIG -d_label $D_LABEL
    done
done

## OpenL3
embed_method=OpenL3
hop_type=(no_pooling pooling)
mkdir -p ${D_RESULT}/${embed_method}
for h in ${hop_type[@]}; do
    mkdir -p ${D_OUT}/${embed_method}/$h
    mkdir -p ${D_RESULT}/${embed_method}/$h
    echo $h

    for f in ${dataset_name[@]}; do
	echo $h/$f
	mkdir -p ${D_OUT}/${embed_method}/$h/$f
	mkdir -p ${D_RESULT}/${embed_method}/$h/$f

	# evaluation
	for n in ${num[@]}; do
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$f/$n
	    # inference
	    python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/$f/parameter.json -f_list ${D_LIST}/list_${f}.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$f/$n -split validation

	    # mir_eval
	    python3 m_eval_single.py -f_list ${D_LIST}/list_${f}.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$f/"$n" -split validation -f_config $F_CONFIG -d_label $D_LABEL
	done
	python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$f
    done

    ## test: RWC, trained model: harmonix
    # inference
    python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/harmonix_single/parameter.json -f_list ${D_LIST}/list_rwc_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/harmonix_single -d_result ${D_RESULT}/${embed_method}/$h/harmonix_single -split test
    # mir_eval
    python3 m_eval_single.py -f_list ${D_LIST}/list_rwc_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/harmonix_single -split test -f_config $F_CONFIG -d_label $D_LABEL

    ## test: harmonix, trained model: RWC
    # inference
    python3 m_inference_single.py -f_parameter ${D_OUT}/${embed_method}/$h/rwc_single/parameter.json -f_list ${D_LIST}/list_harmonix_single.json -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/rwc_single -d_result ${D_RESULT}/${embed_method}/$h/rwc_single -split test
    # mir_eval
    python3 m_eval_single.py -f_list ${D_LIST}/list_harmonix_single.json -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/rwc_single -split test -f_config $F_CONFIG -d_label $D_LABEL
done
