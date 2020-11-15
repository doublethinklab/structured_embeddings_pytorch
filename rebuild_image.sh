#!/bin/bash

docker build \
    --no-cache \
    -f docker/structured-embeddings.Dockerfile \
    -t doublethinklab/structured-embeddings:latest \
    .
