#! /bin/bash

D_OUT=../output
D_RESULT=../result
mkdir -p $D_RESULT

D_DATASET=../../dataset
D_LABEL=$D_DATASET/label
D_EMBED=$D_DATASET/embeddings
D_REFERENCE=$D_DATASET/reference
D_LIST=$D_DATASET/list
F_LIST=$D_LIST/list_harmonix_8fold.json
F_CONFIG=../../config/config.json

idx_fold=`seq -w 0 7`
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
	for f in ${idx_fold[@]}; do
	    echo $h/$m/$f
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f

	    # evaluation
	    for n in ${num[@]}; do
		mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f/$n
		# inference
		python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/$n -split validation

		# mir_eval
		python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/"$n" -idx_fold $f -split validation -f_config $F_CONFIG -d_label $D_LABEL
	    done
	    python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$m/$f

	    # test
	    # inference
	    python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -d_result ${D_RESULT}/${embed_method}/$h/$m/$f -split test

	    # mir_eval
	    python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f -idx_fold $f -split test -f_config $F_CONFIG -d_label $D_LABEL
	done

	python3 m_summary_test.py -d_result ${D_RESULT}/${embed_method}/$h/$m
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
	for f in ${idx_fold[@]}; do
	    echo $h/$m/$f
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f

	    # evaluation
	    for n in ${num[@]}; do
		mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f/$n
		# inference
		python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/$n -split validation

		# mir_eval
		python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/"$n" -idx_fold $f -split validation -f_config $F_CONFIG -d_label $D_LABEL
	    done
	    python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$m/$f

	    # test
	    # inference
	    python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -d_result ${D_RESULT}/${embed_method}/$h/$m/$f -split test

	    # mir_eval
	    python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f -idx_fold $f -split test -f_config $F_CONFIG -d_label $D_LABEL
	done

	python3 m_summary_test.py -d_result ${D_RESULT}/${embed_method}/$h/$m
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
	for f in ${idx_fold[@]}; do
	    echo $h/$m/$f
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f

	    # evaluation
	    for n in ${num[@]}; do
		mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f/$n
		# inference
		python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/$n -split validation

		# mir_eval
		python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/"$n" -idx_fold $f -split validation -f_config $F_CONFIG -d_label $D_LABEL
	    done
	    python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$m/$f

	    # test
	    # inference
	    python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -d_result ${D_RESULT}/${embed_method}/$h/$m/$f -split test

	    # mir_eval
	    python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f -idx_fold $f -split test -f_config $F_CONFIG -d_label $D_LABEL
	done

	python3 m_summary_test.py -d_result ${D_RESULT}/${embed_method}/$h/$m
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
    for f in ${idx_fold[@]}; do
	echo $h/$f
	mkdir -p ${D_OUT}/${embed_method}/$h/$f
	mkdir -p ${D_RESULT}/${embed_method}/$h/$f

	# evaluation
	for n in ${num[@]}; do
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$f/$n
	    # inference
	    python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$f/$n -split validation

	    # mir_eval
	    python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$f/"$n" -idx_fold $f -split validation -f_config $F_CONFIG -d_label $D_LABEL
	done
	python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$f

	# test
	# inference
	python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$f -d_result ${D_RESULT}/${embed_method}/$h/$f -split test

	# mir_eval
	python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$f -idx_fold $f -split test -f_config $F_CONFIG -d_label $D_LABEL
    done
    python3 m_summary_test.py -d_result ${D_RESULT}/${embed_method}/$h
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
	for f in ${idx_fold[@]}; do
	    echo $h/$m/$f
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f

	    # evaluation
	    for n in ${num[@]}; do
		mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f/$n
		# inference
		python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/$n -split validation

		# mir_eval
		python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/"$n" -idx_fold $f -split validation -f_config $F_CONFIG -d_label $D_LABEL
	    done
	    python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$m/$f

	    # test
	    # inference
	    python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -d_result ${D_RESULT}/${embed_method}/$h/$m/$f -split test

	    # mir_eval
	    python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f -idx_fold $f -split test -f_config $F_CONFIG -d_label $D_LABEL
	done

	python3 m_summary_test.py -d_result ${D_RESULT}/${embed_method}/$h/$m
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
    for f in ${idx_fold[@]}; do
	echo $h/$f
	mkdir -p ${D_OUT}/${embed_method}/$h/$f
	mkdir -p ${D_RESULT}/${embed_method}/$h/$f

	# evaluation
	for n in ${num[@]}; do
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$f/$n
	    # inference
	    python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$f/$n -split validation

	    # mir_eval
	    python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$f/"$n" -idx_fold $f -split validation -f_config $F_CONFIG -d_label $D_LABEL
	done
	python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$f

	# test
	# inference
	python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$f -d_result ${D_RESULT}/${embed_method}/$h/$f -split test

	# mir_eval
	python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$f -idx_fold $f -split test -f_config $F_CONFIG -d_label $D_LABEL
    done
    python3 m_summary_test.py -d_result ${D_RESULT}/${embed_method}/$h
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
	for f in ${idx_fold[@]}; do
	    echo $h/$m/$f
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f

	    # evaluation
	    for n in ${num[@]}; do
		mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f/$n
		# inference
		python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/$n -split validation

		# mir_eval
		python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/"$n" -idx_fold $f -split validation -f_config $F_CONFIG -d_label $D_LABEL
	    done
	    python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$m/$f

	    # test
	    # inference
	    python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -d_result ${D_RESULT}/${embed_method}/$h/$m/$f -split test

	    # mir_eval
	    python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f -idx_fold $f -split test -f_config $F_CONFIG -d_label $D_LABEL
	done

	python3 m_summary_test.py -d_result ${D_RESULT}/${embed_method}/$h/$m
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
	for f in ${idx_fold[@]}; do
	    echo $h/$m/$f
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f

	    # evaluation
	    for n in ${num[@]}; do
		mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f/$n
		# inference
		python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/$n -split validation

		# mir_eval
		python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/"$n" -idx_fold $f -split validation -f_config $F_CONFIG -d_label $D_LABEL
	    done
	    python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$m/$f

	    # test
	    # inference
	    python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -d_result ${D_RESULT}/${embed_method}/$h/$m/$f -split test

	    # mir_eval
	    python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f -idx_fold $f -split test -f_config $F_CONFIG -d_label $D_LABEL
	done

	python3 m_summary_test.py -d_result ${D_RESULT}/${embed_method}/$h/$m
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
    for f in ${idx_fold[@]}; do
	echo $h/$f
	mkdir -p ${D_OUT}/${embed_method}/$h/$f
	mkdir -p ${D_RESULT}/${embed_method}/$h/$f

	# evaluation
	for n in ${num[@]}; do
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$f/$n
	    # inference
	    python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$f/$n -split validation

	    # mir_eval
	    python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$f/"$n" -idx_fold $f -split validation -f_config $F_CONFIG -d_label $D_LABEL
	done
	python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$f

	# test
	# inference
	python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$f -d_result ${D_RESULT}/${embed_method}/$h/$f -split test

	# mir_eval
	python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$f -idx_fold $f -split test -f_config $F_CONFIG -d_label $D_LABEL
    done
    python3 m_summary_test.py -d_result ${D_RESULT}/${embed_method}/$h
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
	for f in ${idx_fold[@]}; do
	    echo $h/$m/$f
	    mkdir -p ${D_OUT}/${embed_method}/$h/$m/$f
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f

	    # evaluation
	    for n in ${num[@]}; do
		mkdir -p ${D_RESULT}/${embed_method}/$h/$m/$f/$n
		# inference
		python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/$n -split validation

		# mir_eval
		python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f/"$n" -idx_fold $f -split validation -f_config $F_CONFIG -d_label $D_LABEL
	    done
	    python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$m/$f

	    # test
	    # inference
	    python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$m/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$m/$f -d_result ${D_RESULT}/${embed_method}/$h/$m/$f -split test

	    # mir_eval
	    python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$m/$f -idx_fold $f -split test -f_config $F_CONFIG -d_label $D_LABEL
	done

	python3 m_summary_test.py -d_result ${D_RESULT}/${embed_method}/$h/$m
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
    for f in ${idx_fold[@]}; do
	echo $h/$f
	mkdir -p ${D_OUT}/${embed_method}/$h/$f
	mkdir -p ${D_RESULT}/${embed_method}/$h/$f

	# evaluation
	for n in ${num[@]}; do
	    mkdir -p ${D_RESULT}/${embed_method}/$h/$f/$n
	    # inference
	    python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$f -f_model model_"$n".pkl -d_result ${D_RESULT}/${embed_method}/$h/$f/$n -split validation

	    # mir_eval
	    python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$f/"$n" -idx_fold $f -split validation -f_config $F_CONFIG -d_label $D_LABEL
	done
	python3 m_summary.py -d_result ${D_RESULT}/${embed_method}/$h/$f

	# test
	# inference
	python3 m_inference_8fold.py -f_parameter ${D_OUT}/${embed_method}/$h/$f/parameter.json -f_list $F_LIST -f_config $F_CONFIG -d_embed $D_EMBED/${embed_method}/$h -d_label $D_LABEL -d_model ${D_OUT}/${embed_method}/$h/$f -d_result ${D_RESULT}/${embed_method}/$h/$f -split test

	# mir_eval
	python3 m_eval_8fold.py -f_list $F_LIST -d_reference $D_REFERENCE -d_result ${D_RESULT}/${embed_method}/$h/$f -idx_fold $f -split test -f_config $F_CONFIG -d_label $D_LABEL
    done
    python3 m_summary_test.py -d_result ${D_RESULT}/${embed_method}/$h
done
