# CLAP

## Usage
(1) get LAION CLAP and checkpoints
```
$ git clone https://github.com/LAION-AI/CLAP
$ mkdir -p checkpoint
$ wget -O checkpoint/music_audioset_epoch_15_esc_90.14.pt https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt
$ wget -O checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt
```

(2) install LAION CLAP using Docker
```
$ docker build -t fae:clap .
$ docker run -v /home/<user>:/home/<user> --runtime=nvidia -it --name=clap fae:clap /bin/bash
```

(3) get embeddings
```
$ ./EXE-GET-EMBEDDINGS.sh
```
