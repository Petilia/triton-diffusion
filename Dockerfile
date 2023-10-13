FROM nvcr.io/nvidia/tritonserver:23.04-py3  

WORKDIR /workspace

RUN apt-get update && apt-get install cmake -y

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade tensorrt

RUN git clone https://github.com/NVIDIA/TensorRT.git -b release/8.6 --single-branch 

ENV TRT_OSSPATH=/workspace/TensorRT
WORKDIR ${TRT_OSSPATH}

ENV PLUGIN_LIBS="${TRT_OSSPATH}/build/out/libnvinfer_plugin.so"

RUN cd demo/Diffusion/ \
    && mkdir -p onnx engine output

RUN pip3 install -r demo/Diffusion/requirements.txt

RUN pip install tritonclient[http]




