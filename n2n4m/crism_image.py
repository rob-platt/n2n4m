import os
import matplotlib.pyplot as plt
import numpy as np
import spectral
import warnings
from ipywidgets import widgets, interactive, IntSlider

from n2n4m import io
from n2n4m.wavelengths import ALL_WAVELENGTHS, PLEBANI_WAVELENGTHS
from n2n4m.postprocessing import calculate_pixel_blandness
from n2n4m.summary_parameters import IMPLEMENTED_SUMMARY_PARAMETERS
from crism_ml.preprocessing import remove_spikes_column, ratio

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
    


class Visualiser():

    def __init__(self, image: CRISMImage):
        self.image = image
        self.false_colour_im = None
        self.bad_value_check_flag = False
        self.im_array_copy = self.image.image_array # If bad values are detected, a copy will be made and bad values replaced with np.nan.
        pass

    def replace_bad_values(self) -> None:
        """Replace bad values in the new array with np.nan."""
        self.im_array_copy[self.im_array_copy > 1000] = np.nan
        return None
    
    def detect_bad_values(self) -> bool:
        """Check for bad values in the image.
        Bad values are anything > 1000, so can be used for raw or ratioed imagery.
        Includes defined CRISM bad value of 65535.

        Returns
        -------
        bool
            True if bad values are present, False otherwise.
        """
        if np.any(self.image.image_array > 1000):
           return True

        self.bad_value_check_flag = True
        return False
        
    def bad_value_check(self) -> None:
        """Check for bad values in the image.
        Bad values are anything > 1000, so can be used for raw or ratioed imagery.
        Includes defined CRISM bad value of 65535.
        """
        if self.detect_bad_values():
            warnings.warn("Bad values detected in the image. A copy has been made, values > 1000 will be replaced with np.nan.")
            self.im_array_copy = self.image.image_array.copy()
            self.replace_bad_values()
        return None
    
    def clip_image(self, image: np.ndarray, percentile: float = 99.9) -> np.ndarray:
        """Function to clip summary parameters for visualisation."""
        image[image < 0] = 0
        image[image > np.nanpercentile(image, percentile)] = np.nanpercentile(image, percentile)
        return image

    def get_false_colours(self) -> None:
        pass

    def plot_spectrum(
        self,
        pixel_coords: tuple[int, int],
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> plt.Axes:
        """Plot spectrum of a pixel (1D)
        
        Parameters
        ----------
        pixel_coords : tuple
            (x, y) coordinates of the pixel.
        ax : plt.Axes, optional
            Axes object to plot the image on.
            If None, a new figure and axes will be created.
        title : str, optional
            Title of the plot.
            If None, the title will be the name of the image.

        Returns
        -------
        plt.Axes
            Axes object of the plot.    
        """
        if not self.bad_value_check_flag:
            self.bad_value_check()
        if pixel_coords[0] > self.image.spatial_dims[1] or pixel_coords[1] > self.image.spatial_dims[0]:
            raise ValueError("Pixel coordinates out of range.")             
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 2.5))
        else:
            fig = ax.get_figure()
        
        pixel = self.im_array_copy[pixel_coords[1], pixel_coords[0]]
        
        ax.plot(ALL_WAVELENGTHS, pixel)
        ax.set_yticks([])
        ax.set_ylabel("Reflectance (I/F)")
        ax.set_xlabel("Wavelength (μm)")
        if title is None:
            ax.set_title(self.image.im_name)
        else: ax.set_title(title)
        return fig, ax

    def plot_image(
        self,
        band_num: int,
        title: str | None = None,
        ax: plt.Axes | None = None,
    ) -> None:
        """Plot 2D slice of hyperspectral datacube.
        
        Parameters
        ----------
        band_num : int
            Band number to plot.
        title : str, optional
            Title of the plot.
            If None, the title will be the name of the image.
        ax : plt.Axes, optional
            Axes object to plot the image on.
            If None, a new figure and axes will be created.
        """
        if not self.bad_value_check_flag:
            self.bad_value_check()
        if band_num > self.image.num_bands or band_num < 0:
            raise ValueError("Band number out of range.")
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        image = self.im_array_copy[:, :, band_num]
        ax.imshow(image)
        if title is None:
            ax.set_title(self.image.im_name)
        else:
            ax.set_title(title)

        ax.set_axis_off()
        return fig, ax
    
    def plot_summary_parameter(self, parameter: str, ax: plt.Axes | None = None) -> plt.Axes:
        """Plot summary parameter for the image.
        Makes a copy of the summary parameter, clips at 0 and 99th percentile.
        
        Parameters
        ----------
        parameter : str
            Summary parameter to plot.
            Must be in IMPLEMENTED_SUMMARY_PARAMETERS.
        ax : plt.Axes, optional
            Axes object to plot the image on.
            If None, a new figure and axes will be created.

        Returns
        -------
        plt.Axes
            Axes object of the plot.    
        """
        if parameter not in self.image.summary_parameters:
            raise ValueError(f"Summary parameter {parameter} has not been calculated.")
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        parameter_image = self.image.summary_parameters[parameter].copy()
        parameter_image = self.clip_image(parameter_image)

        ax.imshow(parameter_image)
        ax.set_title(parameter)
        ax.set_axis_off()
        return fig, ax
    
    def interactive_plot(self) -> interactive:
        """Interactive plot of the image.
        """
        def update_plots(x, y, **kwargs):

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            if "band_num" in kwargs:
                if kwargs["band_num"] == "" or int(kwargs["band_num"]) > self.image.num_bands or int(kwargs["band_num"]) < 1:
                    band_num = 59
                else:
                    band_num = int(kwargs["band_num"])-1 # -1 as python is 0-indexed
                self.plot_image(band_num, ax=ax[0])
            
            ax[0].scatter(x, y, marker='x', color='red', label="Selected Pixel")

            self.plot_spectrum((x, y), ax=ax[1])

            plt.tight_layout()
            plt.show()
        x_slider = IntSlider(min=0, max=self.image.spatial_dims[1]-1, value=0, step=1, description='X:')
        y_slider = IntSlider(min=0, max=self.image.spatial_dims[0]-1, value=0, step=1, description='Y:')
        band_to_display = widgets.Text(value="60", placeholder="1-438", description="Band:", continuous_update=False)

        # Create interactive widget
        interactive_plot = interactive(update_plots, x=x_slider, y=y_slider, band_num=band_to_display)
        return interactive_plot
