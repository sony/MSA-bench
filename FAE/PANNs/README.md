# PANNs

## Usage
(1) get PANNs repository and checkpoints
```
$ git clone https://github.com/qiuqiangkong/audioset_tagging_cnn
$ mkdir -p checkpoint
$ CHECKPOINT_PATH_CNN14="checkpoint/Cnn14_mAP=0.431.pth"
$ wget -O $CHECKPOINT_PATH_CNN14 https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1
$ CHECKPOINT_PATH_CNN14_DECISIONLEVELMAX="checkpoint/Cnn14_DecisionLevelMax_mAP=0.385.pth"
$ wget -O $CHECKPOINT_PATH_CNN14_DECISIONLEVELMAX https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1
```

(2) build/run Docker
```
$ docker build -t fae:panns .
$ docker run -v /home/<user>:/home/<user> --runtime=nvidia -it --name=panns fae:panns /bin/bash
```

(3) get embeddings
```
$ ./EXE-GET-EMBEDDINGS.sh
```

## Remarks
If you want to use frame-wised model "Cnn14_DecisionLevelMax", see the comment out in EXE-GET-EMBEDDINGS.sh
