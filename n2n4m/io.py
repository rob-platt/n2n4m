import numpy as np
import os
from spectral.io import envi, spyfile
import spectral 


import crism_ml.io

NUM_BANDS = 438
    

def load_image(filename: str) -> tuple[np.ndarray, spectral.SpyFile]:
    """
    Modified from https://github.com/Banus/crism_ml/blob/master/crism_ml/io.py load_image
    """
    fbase, _ = os.path.splitext(filename)
    try:
        img = envi.open(f"{fbase}.hdr")
    except spyfile.FileNotFoundError:
        crism_ml.io._generate_envi_header(f"{fbase}.lbl")
        img = envi.open(f"{fbase}.hdr")
    if type(img) is not spectral.io.bilfile.BilFile:
        raise ValueError(f"Image at: {filename} is not a BIL file - suggests not a CRISM image.")
    arr = img.load()
    return np.array(arr), img


def read_lbl_file(filename: str) -> str:
    """
    Read the .lbl file and return the contents as a string.
    """
    with open(filename, "r") as f:
        lbl = f.read()
    f.close()
    return lbl
    

def modify_hdr_metadata(metadata: dict, filename: str) -> dict:
    """
    Modify the metadata of the header file to correct which .img file it is pointing to.
    """
    # If image has not had CAT ATP corrections applied, then this won't exist in the metadata, so can skip
    if "cat input files" in metadata:
        metadata["cat input files"] = filename
    return metadata


def modify_lbl_str(lbl: str, original_filename: str, new_filename: str) -> str:
    """
    Modify the .lbl str to correct the filename of the .img file it is pointing to.
    """
    mod_lbl = lbl.replace(original_filename.upper(), new_filename) # .lbl files are allcaps but original filename may not be
    return mod_lbl


def write_lbl_file(filename: str, lbl: str) -> None:
    """
    Write .lbl file to filename.
    """
    with open(filename, "w") as f:
        f.write(lbl)
    f.close()
    return None

def write_image(filename: str, data: np.ndarray, original_image: envi.SpyFile) -> None:
    """
    
    Parameters
    ----------
    filename : str
        The filename and path of the new image to be written.
        Must be a .hdr file. The .img and .lbl files will be written with the same name.
    original_image : envi.SpyFile
        The object of the original image loaded in, used to write the new image.
        MUST be the original image, as need access to the header information to be able to generate the new header and .lbl file for the new image
        in such a way for CRISM Analysis Toolkit of ENVI to read it and map project it correctly.
    """
    if not filename.endswith(".hdr"):
        raise ValueError("Filename must end .hdr")
    metadata = original_image.metadata
    metadata = modify_hdr_metadata(metadata, filename)
    original_lbl_path = original_image.filename.replace(".img", ".lbl")
    original_lbl = read_lbl_file(original_lbl_path)
    new_lbl = modify_lbl_str(original_lbl, original_image.filename, filename)
    write_lbl_file(filename.replace(".hdr", ".lbl"), new_lbl)
    envi.save_image(filename, data, metadata=metadata)
    return None

