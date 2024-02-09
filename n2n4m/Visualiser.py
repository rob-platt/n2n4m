import numpy as np
import warnings
import matplotlib.pyplot as plt

from n2n4m.crism_image import CRISMImage
from n2n4m.wavelengths import ALL_WAVELENGTHS
from ipywidgets import widgets, interactive, HBox, VBox


class Visualiser():

    def __init__(self, image: CRISMImage):
        self.image = image
        self.bad_value_check_flag = False
        self.im_array_copy = self.image.image_array # If bad values are detected, a copy will be made and bad values replaced with np.nan.

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

    def get_spectrum(self, pixel_coords: tuple[int, int]) -> np.ndarray:
        """Get spectrum of a pixel (1D)
        
        Parameters
        ----------
        pixel_coords : tuple
            (x, y) coordinates of the pixel.

        Returns
        -------
        np.ndarray
            Spectrum of the pixel.    
        """
        if not self.bad_value_check_flag:
            self.bad_value_check()
        if pixel_coords[0] > self.im_array_copy.shape[1] or pixel_coords[1] > self.im_array_copy.shape[0]:
            raise ValueError("Pixel coordinates out of range.")             
        pixel = self.im_array_copy[pixel_coords[1], pixel_coords[0]]
        return pixel
    
    def get_bands(self, bands: tuple[int, int]) -> tuple[float, ...]:
        """Get which bands to plot for the spectrum.
        """
        if bands[0] < 1 or bands[1] > self.image.num_bands or bands[0] > bands[1] or bands[0] == bands[1]:
            raise ValueError("Band numbers out of range.")
        return ALL_WAVELENGTHS[bands[0]-1:bands[1]+1] # -1 as python is 0-indexed, and +1 to be inclusive of stop. 

    def get_image(self, band_num: int) -> np.ndarray:
        """Get 2D slice of hyperspectral datacube.
        
        Parameters
        ----------
        band_num : int
            Band number to get.

        Returns
        -------
        np.ndarray
            2D slice of the hyperspectral datacube.    
        """
        if not self.bad_value_check_flag:
            self.bad_value_check()
        if band_num > self.image.num_bands or band_num < 0:
            raise ValueError("Band number out of range.")
        image = self.im_array_copy[:, :, band_num]
        return image
    
    def get_summary_parameter(self, parameter: str) -> np.ndarray:
        """Get summary parameter for the image. Returns clipped image.
        
        Parameters
        ----------
        parameter : str
            Summary parameter to get.
            Must be in IMPLEMENTED_SUMMARY_PARAMETERS.

        Returns
        -------
        summary_parameter : np.ndarray
            Summary parameter for the image.    
        """
        if parameter not in self.image.summary_parameters:
            raise ValueError(f"Summary parameter {parameter} has not been calculated.")
        summary_parameter = self.image.summary_parameters[parameter]
        summary_parameter = self.clip_image(summary_parameter)
        return summary_parameter

    def plot_spectrum(
        self,
        pixel: np.ndarray,
        bands: tuple[float, ...],
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> plt.Axes:
        """Plot spectrum of a pixel (1D)
        Pixel length must match bands length.

        Parameters
        ----------
        pixel : np.ndarray
            Spectrum of the pixel.
        bands : tuple
            Wavelengths of the bands.
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
        if pixel.ndim !=1 or len(pixel) != len(bands):
            raise ValueError("Pixel and bands must be 1D and of the same length.")
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 2.5))
        else:
            fig = ax.get_figure()
        
        ax.plot(bands, pixel)
        ax.set_yticks([])
        ax.set_ylabel("Reflectance (I/F)")
        ax.set_xlabel("Wavelength (μm)")
        if title is None:
            ax.set_title(f"{self.image.im_name}")
        else: ax.set_title(title)
        return fig, ax

    def plot_image(
        self,
        image: np.ndarray,
        title: str | None = None,
        ax: plt.Axes | None = None,
    ) -> None:
        """Plot 2D slice of hyperspectral datacube.
        
        Parameters
        ----------
        image : np.ndarray
            2D slice of the hyperspectral datacube.
        title : str, optional
            Title of the plot.
            If None, the title will be the name of the image.
        ax : plt.Axes, optional
            Axes object to plot the image on.
            If None, a new figure and axes will be created.
        """
        if image.ndim != 2:
            raise ValueError(f"Image must be 2D but is {image.ndim}D")
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        ax.imshow(image)
        if title is None:
            ax.set_title(self.image.im_name)
        else:
            ax.set_title(title)

        ax.set_axis_off()
        return fig, ax
    
    def interactive_plot(self) -> interactive | VBox | HBox:
        """Interactive plot of the image and spectrum.
        """
        def update_plots(x, y, spectrum_range, image_band, **kwargs):

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            # Image Plot
            if "dropdown" in kwargs and kwargs["dropdown"] != "Raw":
                image = self.get_summary_parameter(kwargs["dropdown"])
            else: # Default is must have at least passed a raw image.
                if image_band == "" or int(image_band) > self.image.num_bands or int(image_band) < 1:
                    band_num = 59
                else:
                    band_num = int(image_band)-1 # -1 as python is 0-indexed
                image = self.get_image(band_num)
            self.plot_image(image, ax=ax[0]) 
                  
            # Add scatter plot of selected pixel to image plot
            ax[0].scatter(x, y, marker='x', color='red', label="Selected Pixel")

            # Spectrum Plot
            # Options are ratioed or raw, if ratioed, must have a ratioed image.
            # If option is not ratioed, then would be same as if ratioed not in kwargs.
            pixel = self.get_spectrum((x, y)) 
            if "ratio" in kwargs and kwargs["ratio"] == "Ratioed":
                pixel = self.get_spectrum((x, y))
            band_range = format_spectrum_band_range(spectrum_range)
            bands = self.get_bands(band_range)
            pixel = pixel[band_range[0]-1:band_range[1]+1] # -1 as python is 0-indexed and +1 to be inclusive of stop.
            self.plot_spectrum(pixel, bands=bands, ax=ax[1], title=f"{self.image.im_name} - Spectrum at ({x}, {y})")

            plt.tight_layout()
            return fig, ax

        def enable_image_band_widget(change):
            """Summary parameters have no bands, so if dropdown is not raw, disable the band widget."""
            if change.new == "Raw":
                image_band_to_display.disabled = False
                image_band_to_display.value = "60"
            else:
                image_band_to_display.disabled = True
                image_band_to_display.value = "N/A"

        def format_spectrum_band_range(input_str: str):
            start_band, stop_band = input_str.split("-")
            return int(start_band), int(stop_band)
    
        style = {'description_width': 'initial'}
        x_slider = widgets.IntSlider(min=0, max=self.image.spatial_dims[1]-1, value=0, step=1, description='X:')
        y_slider = widgets.IntSlider(min=0, max=self.image.spatial_dims[0]-1, value=0, step=1, description='Y:')
        image_band_to_display = widgets.Text(value="60", placeholder="1-438", description="Image Band:", style=style, continuous_update=False)
        spectrum_band_range = widgets.Text(value="1-438", placeholder="1-438", description="Spectrum Band Range:", style=style, continuous_update=False)

        additional_spectrum_widgets = {}
        if self.image.ratioed_image is not None:
            ratio_button = widgets.ToggleButtons(options=["Raw", "Ratioed"], button_style="", value="Raw", description="Spectrum Type:", style=style)
            additional_spectrum_widgets["ratio"] = ratio_button

        additional_image_widgets = {}
        if self.image.summary_parameters:
            dropdown_options = list(self.image.summary_parameters.keys())
            dropdown_options.append("Raw")
            summary_parameter_dropdown = widgets.Dropdown(options=dropdown_options, value="Raw", description="Image Options:", style=style)
            summary_parameter_dropdown.observe(enable_image_band_widget, names='value')
            additional_image_widgets["dropdown"] = summary_parameter_dropdown

        interactive_plot = interactive(update_plots, x=x_slider, y=y_slider, spectrum_range=spectrum_band_range, image_band=image_band_to_display, **additional_spectrum_widgets, **additional_image_widgets)
        image_controls = VBox([x_slider, y_slider, image_band_to_display, *additional_image_widgets.values()])
        spectrum_controls = VBox([spectrum_band_range, *additional_spectrum_widgets.values()])
        all_controls = HBox([image_controls, spectrum_controls])
        output = interactive_plot.children[-1]
        return VBox([all_controls, output])