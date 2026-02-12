# MusicFM

## Usage
(1) get MusicFM repository and download models
```
$ git clone https://github.com/minzwon/musicfm
$ wget -P ./musicfm/data/ https://huggingface.co/minzwon/MusicFM/resolve/main/fma_stats.json
$ wget -P ./musicfm/data/ https://huggingface.co/minzwon/MusicFM/resolve/main/pretrained_fma.pt
$ wget -P ./musicfm/data/ https://huggingface.co/minzwon/MusicFM/resolve/main/msd_stats.json
$ wget -P ./musicfm/data/ https://huggingface.co/minzwon/MusicFM/resolve/main/pretrained_msd.pt
```

(2) build/run Docker
```
$ docker build -t fae:musicfm .
$ docker run -v /home/<user>:/home/<user> --runtime=nvidia -it --name=musicfm fae:musicfm /bin/bash
```

(3) get embeddings
```
$ ./EXE-GET-EMBEDDINGS.sh
```
