import os
import numpy as np
import spectral

from n2n4m import io
from n2n4m.wavelengths import ALL_WAVELENGTHS, PLEBANI_WAVELENGTHS
from n2n4m.summary_parameters import IMPLEMENTED_SUMMARY_PARAMETERS
from n2n4m.postprocessing import calculate_pixel_blandness
from crism_ml.preprocessing import remove_spikes_column, ratio

from n2n4m.cotcat_denoise import cotcat_denoise


BAND_MASK = np.isin(ALL_WAVELENGTHS, PLEBANI_WAVELENGTHS)

class CRISMImage:
    """
    Class for reading, manipulation and visulisation of CRISM images.

    Attributes
    ----------
    filepath : str
        Filepath of the image.
    image_array : np.ndarray
        Array of the image.
        (n_rows, n_columns, n_wavelengths)
    spatial_dims : tuple
        Spatial dimensions of the image.
        (n_rows, n_columns)
    im_shape : tuple
        Shape of the image.
        (n_rows, n_columns, n_wavelengths)
    im_name : str
        Name of the image.
    num_bands : int
        Number of channels in the image.
    summary_parameters : dict
        Dictionary of summary parameters calculated for the image.
    ratioed_image : np.ndarray
        Image after being ratioed.
        Ratioing acheived using GMM from [1] to identify bland pixels.
        (n_rows, n_columns, n_wavelengths)

    References
    ----------
    1. Plebani E, Ehlmann BL, Leask EK, Fox VK, Dundar MM. 
    A machine learning toolkit for CRISM image analysis. 
    Icarus. 2022 Apr;376:114849. 

    """
    def __init__(self, filepath: str):

        self.filepath = filepath
        self.image_array, self.SPy = self._load_image(self.filepath)
        self.im_shape = self.image_array.shape
        self.spatial_dims = (self.im_shape[0], self.im_shape[1])
        self.num_bands = self.im_shape[2]

        self.im_name = self._get_im_name()
        self.summary_parameters = {}
        print("Image loaded successfully.")

        self.ratioed_image = None

    def _load_image(self, filepath: str) -> tuple[np.ndarray, spectral.SpyFile]:
        "Load the image from the filepath."
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found.")
        try:
            image = io.load_image(filepath)
        except:
            raise ValueError(f"Image at: {filepath} could not be loaded.")
        return image
    
    def _get_im_name(self) -> str:
        "Clip image acquisition type + hexadecimal number from the filepath."
        im_name = self.filepath.split("/")[-1]
        im_name = im_name.split("_")[0]
        return im_name

    def ratio_image(self, train_data_dir: str = "data") -> None:
        """Ratio the image using the Plebani bland pixel model.
        Uses the 350 bands in PLEBANI_WAVELENGTHS to determine pixel blandness. 
        ALL_WAVELENGTHS are ratioed.

        Parameters
        ----------
        train_data_dir : str, optional
            Directory containing the training data for the GMM.
            Training data must be called "CRISM_bland_unratioed.mat"
            Default dir "data". 
        """
        if self.ratioed_image is not None:
            print("Image has already been ratioed.")
            return
        flattened_image = self.image_array.reshape(-1, self.num_bands)
        flattened_image_clipped = flattened_image[:, BAND_MASK]
        pixel_blandness = calculate_pixel_blandness(flattened_image_clipped, self.spatial_dims, train_data_dir)
        despiked_image = remove_spikes_column(self.image_array.copy(), size=3, sigma=5)
        self.ratioed_image = ratio(despiked_image, pixel_blandness)
        return 

    def calculate_summary_parameter(self, parameter: str) -> None:
        """Calculate summary parameter for the image.
        Uses the ratioed image.

        Parameters
        ----------
        parameter : str
            Summary parameter to calculate.
            Must be in IMPLEMENTED_SUMMARY_PARAMETERS.       
        """
        if self.ratioed_image is None:
            raise ValueError("Image must be ratioed before summary parameters can be calculated.")
        if parameter in self.summary_parameters:
            print(f"{parameter} has already been calculated.")
            return
        if parameter not in IMPLEMENTED_SUMMARY_PARAMETERS:
            raise ValueError(f"Summary parameter {parameter} is not implemented.")
        flattened_image = self.ratioed_image.reshape(-1, self.num_bands)
        self.summary_parameters[parameter] = IMPLEMENTED_SUMMARY_PARAMETERS[parameter](flattened_image, ALL_WAVELENGTHS).reshape(self.spatial_dims)
        return
    
    def write_image(self, filepath: str, data: np.ndarray) -> None:
        """Write the image to a new file. 
        Uses the original file .hdr and .lbl files to generate the new files.

        Parameters
        ----------
        filepath : str
            Filepath to write the new image to.
            Must be a .hdr file. The .img and .lbl files will be written with the same name.
        data : np.ndarray
            Image to write.      
        """
        io.write_image(filepath, data, self.SPy)
        return None
    

class CRISMImageDenoise(CRISMImage):
    """
    Class for denoising CRISM images.

    Attributes
    ----------
    filepath : str
        Filepath of the image.
    image_array : np.ndarray
        Array of the image.
        (n_rows, n_columns, n_wavelengths)
    spatial_dims : tuple
        Spatial dimensions of the image.
        (n_rows, n_columns)
    im_shape : tuple
        Shape of the image.
        (n_rows, n_columns, n_wavelengths)
    im_name : str
        Name of the image.
    num_bands : int
        Number of channels in the image.
    summary_parameters : dict
        Dictionary of summary parameters calculated for the image.
    ratioed_image : np.ndarray
        Image after being ratioed.
        Ratioing acheived using GMM from [1] to identify bland pixels.
        (n_rows, n_columns, n_wavelengths)

    References
    ----------
    1. Plebani E, Ehlmann BL, Leask EK, Fox VK, Dundar MM. 
    A machine learning toolkit for CRISM image analysis. 
    Icarus. 2022 Apr;376:114849. 

    """
    def __init__(self, filepath: str):
        super().__init__(filepath)
    
        self.cotcat_denoised_image = None
        self.n2n4m_denoised_image = None

    def cotcat_denoise(self, wavelengths: tuple[float, ...] = ALL_WAVELENGTHS) -> None:
        """Apply CoTCAT denoising to the image.
        """
        if self.num_bands != len(wavelengths):
            raise ValueError(f"Number of bands in image: {self.num_bands} does not match number of wavelengths: {len(wavelengths)}.")
        if self.cotcat_denoised_image is not None:
            print("Image has already been denoised using CoTCAT.")
            return
        
        self.cotcat_denoised_image = cotcat_denoise(self.image_array, wavelengths)

    def n2n4m_denoise(self):
        pass




            
