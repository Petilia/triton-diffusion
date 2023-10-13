#!/bin/bash

docker run --gpus=all -it \
            --shm-size=256m \
            -p 8000:8000  \
            -p 8001:8001 \
            -p 8002:8002 \
            --name dummy_triton \
            -v ${PWD}:/workspace/ \
            -v ${PWD}/models:/models \
            dummy_triton 