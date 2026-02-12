# MERT

## Usage
(1) get MERT repository
```
$ git clone https://github.com/yizhilll/MERT
```

(2) build/run Docker
```
$ docker build -t fae:mert .
$ docker run -v /home/<user>:/home/<user> --runtime=nvidia -it --name=mert fae:mert /bin/bash
```

(3) get embeddings
```
$ ./EXE-GET-EMBEDDINGS.sh
```
