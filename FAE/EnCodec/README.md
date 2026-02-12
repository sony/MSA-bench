# EnCodec

## Usage
(1) get EnCodec repository
```
$ git clone https://github.com/facebookresearch/encodec
```

(2) install EnCodec using docker
```
$ docker build -t fae:encodec .
$ docker run -v /home/<user>:/home/<user> --runtime=nvidia -it --name=encodec fae:encodec /bin/bash
```

(3) get embeddings
```
$ ./EXE-GET-EMBEDDINGS.sh
```
