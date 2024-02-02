# Module to run N2N denoising across an entire image. Should allow for CLI usage. Input trained model as instance of Noise2Noise1D class, with weights loaded, image, and output numpy array of denoised image, with extra bands inserted back from original image.
import numpy as np


from n2n4m.model_functions import predict
from n2n4m.model import Noise2Noise1D

def run_denoise(model: Noise2Noise1D, image_filepath: str) -> np.ndarray:
    
    return np.zeros((1, 1))  # Placeholder