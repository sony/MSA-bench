# PaSST

## Usage
(1) get PaSST
```
$ git clone https://github.com/kkoutini/PaSST
$ git clone https://github.com/kkoutini/passt_hear21
```

(2) build/run Docker
```
$ docker build -t fae:passt .
$ docker run -v /home/<user>:/home/<user> --runtime=nvidia -it --name=passt fae:passt /bin/bash
```

(3) get embeddings
```
$ ./EXE-GET-EMBEDDINGS.sh
```
