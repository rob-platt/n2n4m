import numpy as np
import os
from spectral.io import envi, spyfile

import crism_ml.io

NUM_BANDS = 438
    

def load_image(filename: str) -> tuple[np.ndarray, envi.SpyFile]:
    """
    Modified from https://github.com/Banus/crism_ml/blob/master/crism_ml/io.py load_image
    """
    fbase, _ = os.path.splitext(filename)
    try:
        img = envi.open(f"{fbase}.hdr")
    except spyfile.FileNotFoundError:
        crism_ml.io._generate_envi_header(f"{fbase}.lbl")
        img = envi.open(f"{fbase}.hdr")

    arr = img.load()
    return np.array(arr), img
