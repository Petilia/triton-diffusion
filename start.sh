#!/bin/bash

docker run --gpus=all -it \
            --shm-size=256m --rm \
            -p8000:8000  \
            -p8001:8001 \
            -p8002:8002 \
            -v ${PWD}:/workspace/ \
            -v ${PWD}/models:/models \
            dummy_triton 