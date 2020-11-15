#!/bin/bash

if [ -z "$1" ]; then
    PORT=8888
else
    PORT=$1
fi

docker run \
    --rm \
    --gpus all \
    -v ${PWD}:/jupyter_temp/ \
    -w /jupyter_temp \
    -p PORT:PORT \
    doublethinklab/structured-embeddings:latest \
        jupyter notebook \
            --ip 0.0.0.0 \
            --port PORT \
            --no-browser \
            --allow-root
