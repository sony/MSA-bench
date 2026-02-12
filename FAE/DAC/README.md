# DAC

## Usage
(1) get DAC repository
```
$ git clone https://github.com/descriptinc/descript-audio-codec
```

(2) install DAC using Docker
```
$ docker build -t fae:dac .
$ docker run -v /home/<user>:/home/<user> --runtime=nvidia -it --name=dac fae:dac /bin/bash
```

(3) get embeddings
```
$ ./EXE-GET-EMBEDDINGS.sh
```
