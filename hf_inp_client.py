import argparse
import base64
from io import BytesIO

import numpy as np
import requests
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import *


# helper functions to encode and decode images
def encode_image(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue())
    return img_str


def decode_image(img):
    buff = BytesIO(base64.b64decode(img.encode("utf8")))
    image = Image.open(buff)
    return image


def main(model_name):
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # Inputs
    image = Image.open("./test_images/2_trees.jpg")
    prompt = "robot"
    mask = Image.open("test_images/2_trees.png")

    image = encode_image(image).decode("utf8")
    mask = encode_image(mask).decode("utf8")

    # image = np.asarray([str.encode(image)])
    # mask = np.asarray([str.encode(mask)])

    image = np.asarray([image], dtype=object)
    mask = np.asarray([mask], dtype=object)
    prompt = np.asarray([prompt], dtype=object)

    # Set Inputs
    input_tensors = [
        httpclient.InferInput("image", [1], datatype="BYTES"),
        httpclient.InferInput("mask", [1], datatype="BYTES"),
        httpclient.InferInput("prompt", [1], datatype="BYTES"),
    ]

    input_tensors[0].set_data_from_numpy(image.reshape([1]))
    input_tensors[1].set_data_from_numpy(mask.reshape([1]))
    input_tensors[2].set_data_from_numpy(prompt.reshape([1]))

    # Set outputs
    outputs = [httpclient.InferRequestedOutput("generated_image")]

    # Query
    query_response = client.infer(
        model_name=model_name, inputs=input_tensors, outputs=outputs
    )

    # Output
    # generated_image = query_response.as_numpy("generated_image")
    # print(generated_image.shape)


if __name__ == "__main__":
    main("hf_inpaint")
