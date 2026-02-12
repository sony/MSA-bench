# OpenL3

## Usage
(1) get OpenL3 repository
```
$ git clone https://github.com/torchopenl3/torchopenl3
```

(2) install OpenL3 using Docker
```
$ docker build -t audio_encoder:openl3
$ docker run -v /home/<user>:/home/<user> -v /mnt/hdd1:/mnt/hdd1 --runtime=nvidia -it --name=openl3 audio_encoder:openl3 /bin/bash
```

(3) get embeddings
```
$ ./EXE-GET-EMBEDDINGS.sh
```
