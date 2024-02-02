import os
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import widgets, interactive, IntSlider

from n2n4m import io
from n2n4m.wavelengths import ALL_WAVELENGTHS, PLEBANI_WAVELENGTHS
from crism_ml.io import image_shape

BAND_MASK = [True if wavelength in PLEBANI_WAVELENGTHS else False for wavelength in ALL_WAVELENGTHS]

class CRISMImage:
    """
    Class for reading, manipulation and visulisation of CRISM images.

    Attributes
    ----------
    filepath : str
        Filepath of the image.
    image : np.ndarray
        Raw image data.
    spatial_dims : tuple
        Spatial dimensions of the image.
        (n_rows, n_columns)
    im_shape : tuple
        Shape of the image.
        (n_rows, n_columns, n_wavelengths)
    im_name : str
        Name of the image.
    summary_parameters : dict
        Dictionary of summary parameters calculated for the image.

    Methods
    -------

    """
    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found.")

        self.filepath = filepath
        try:
            self.image = io.load_image(filepath)
        except:
            raise ValueError(f"Image at: {filepath} could not be loaded.")

        self.spatial_dims = self.__get_spatial_dims()
        self.im_shape = (*self.spatial_dims, len(ALL_WAVELENGTHS))
        self.image = self.image["IF"].reshape(*self.im_shape)

        self.im_name = self.__get_im_name()
        self.summary_parameters = {}
        self.cotcat_denoised = None
        self.n2n4m_denoised = None
        print("Image loaded successfully.")

    def __get_im_name(self) -> str:
        im_name = self.filepath.split("/")[-1]
        im_name = im_name.split("_")[0]
        return im_name

    def __get_spatial_dims(self) -> tuple:
        im_shape = image_shape(self.image)
        return im_shape

    def calculate_summary_parameter(self, parameter: str) -> None:
        pass

    def cotcat_denoise(self):
        self.cotcat_denoised = None
        pass

    def n2n4m_denoise(self):
        self.n2n4m_denoised = None
        pass

    def get_false_colours(self) -> None:
        self.false_colour_im = None
        pass

    def plot_spectrum(
        self,
        pixel: np.ndarray,
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> None:
        """Plot spectrum of a pixel (1D)"""
        if len(pixel.shape) != 1:
            raise ValueError("Pixel must be 1D.")
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 2.5))
        else:
            fig = ax.get_figure()
        
        if len(pixel) == len(ALL_WAVELENGTHS):
            pixel = pixel[BAND_MASK]
        
        ax.plot(PLEBANI_WAVELENGTHS, pixel)
        ax.set_yticks([])
        ax.set_ylabel("Reflectance (I/F)")
        ax.set_xlabel("Wavelength (μm)")
        if title is None:
            ax.set_title(self.im_name)
        else: ax.set_title(title)
        return fig, ax

    def plot_image(
        self,
        image: np.ndarray,
        title: str | None = None,
        ax: plt.Axes | None = None,
    ) -> None:
        """Plot 2D representation of hyperspectral datacube.
        
        Parameters
        ----------
        image : np.ndarray
            2D representation of the image.
        title : str, optional
            Title of the plot.
            If None, the title will be the name of the image.
        ax : plt.Axes, optional
            Axes object to plot the image on.
            If None, a new figure and axes will be created.
        """
        if len(image.shape) != 2:
            raise ValueError("Image must be 2D.")
        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        ax.imshow(image)
        if title is None:
            ax.set_title(self.im_name)
        else:
            ax.set_title(title)

        ax.set_axis_off()
        return fig, ax
    
    def interactive_plot(self) -> None:
        """Interactive plot of the image.
        """
        def update_plots(x, y, band_num):

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            if band_num == "":
                band_num = 60
            self.plot_image(self.image[:, :, int(band_num)], ax=ax[0])
            
            ax[0].scatter(x, y, marker='x', color='red', label="Selected Pixel")

            self.plot_spectrum(self.image[y, x], ax=ax[1])

            plt.tight_layout()
            plt.show()
        x_slider = IntSlider(min=0, max=self.spatial_dims[1]-1, value=0, step=1, description='X:')
        y_slider = IntSlider(min=0, max=self.spatial_dims[0]-1, value=0, step=1, description='Y:')
        band_to_display = widgets.Text(value="60", placeholder="0-438", description="Band:", continuous_update=False)

        # Create interactive widget
        interactive_plot = interactive(update_plots, x=x_slider, y=y_slider, band_num=band_to_display)
        return interactive_plot
