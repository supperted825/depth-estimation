"""
Script for extracting images and depths from the  original NYU Depth V2 .mat file.
"""

import warnings

import numpy as np
import pymatreader
from PIL import Image


def read_data():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = pymatreader.read_mat(
            filename="./data/nyu_depth_v2_labeled.mat",
            variable_names=["images", "depths"],
        )
    return data


def write_dataset(data):
    images = data["images"].transpose([3, 0, 1, 2])
    depths = data["depths"].transpose([2, 0, 1])
    for i in range(len(images)):
        im = Image.fromarray(images[i])
        im.save(f"./data/images/{i}.jpg")
        np.save(f"./data/depths/{i}.npy", depths[i])


if __name__ == "__main__":
    data = read_data()
    write_dataset(data)
