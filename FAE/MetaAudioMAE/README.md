# Audio MAE (Huang et al.)

## Usage
(1) get AudioMAE repository
```
$ git clone https://github.com/facebookresearch/AudioMAE
```

(2) make "checkpoint" directory, then download checkpoint files to the directory

   - ViT-B, AS-2M [pretrained]
     + https://drive.google.com/file/d/1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu/view?usp=share_link

   - ViT-B, AS-2M pretrained+[finetuned]
     + https://drive.google.com/file/d/18EsFOyZYvBYHkJ7_n7JFFWbj6crz01gq/view?usp=share_link

(3) build/run Docker
```
$ docker build -t fae:meta_audiomae .
$ docker run -v /home/<user>:/home/<user> --runtime=nvidia -it --name=meta_audiomae fae:meta_audiomae /bin/bash
```

(4) copy prepared file
```
$ mv /usr/local/lib/python3.10/dist-packages/timm/models/layers/helpers.py /usr/local/lib/python3.10/dist-packages/timm/models/layers/helpers.py.org
$ cp -p helpers.py /usr/local/lib/python3.10/dist-packages/timm/models/layers/
$ mv AudioMae/models_vit.py AudioMae/models_vit.py.org
$ cp -p models_vit.py AudioMAE/
```

(5) get embeddings
```
$ ./EXE-GET-EMBEDDINGS.sh
```
