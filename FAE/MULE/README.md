# MULE

## Usage
(1) get MULE repository
```
$ git clone https://github.com/PandoraMedia/music-audio-representations
```

(2) install MULE using Docker
```
$ docker build -t fae:mule .
$ docker run -v /home/<user>:/home/<user> --runtime=nvidia -it --name=mule fae:mule /bin/bash
```

(3) get embeddings
```
$ ./EXE-GET-EMBEDDINGS.sh
```

## Remarks
It seems that GPU is not working properly with the Dockerfile.
