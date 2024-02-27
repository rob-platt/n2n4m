import numpy as np
import warnings
import matplotlib.pyplot as plt

from n2n4m.crism_image import CRISMImage, CRISMImageCotcat, CRISMImageN2N4M
from n2n4m.wavelengths import ALL_WAVELENGTHS
from ipywidgets import widgets, interactive, HBox, VBox


class Visualiser:
    """
    Class to create visualisations of CRISM images.

    Parameters
    ----------
    image : CRISMImage | CRISMImageCotcat | CRISMImageN2N4M
        CRISMImage object to visualise.

    Attributes
    ----------
    image : CRISMImage | CRISMImageCotcat | CRISMImageN2N4M
        CRISMImage object to visualise.
    raw_image_bad_value_check_flag : bool
        Flag to check if bad values have been detected in the raw image.
    raw_image_copy : np.ndarray
        Copy of the raw image used to plot. Allows for replacing bad values and clipping without changing the original image.
    ratioed_image_bad_value_check_flag : bool
        Flag to check if bad values have been detected in the ratioed image.
    ratioed_image_copy : np.ndarray
        Copy of the ratioed image. Allows for replacing bad values and clipping without changing the original image.

    Methods
    -------
    replace_bad_values(array: np.ndarray) -> None
        Replace bad values in the new array with np.nan.
    detect_bad_values(array: np.ndarray) -> bool
        Detect if any bad values are present in the image.
    bad_value_check(array: np.ndarray) -> np.ndarray
        Check for bad values in the image.
    bad_value_check_raw_image() -> None
        Check for bad values in the raw image.
    bad_value_check_ratioed_image() -> None
        Check for bad values in the ratioed image.
    clip_image(image: np.ndarray, percentile: float = 99.9) -> np.ndarray
        Clip summary parameters for visualisation.
    get_raw_spectrum(pixel_coords: tuple[int, int]) -> np.ndarray
        Get spectrum of a pixel (1D).
    get_ratioed_spectrum(pixel_coords: tuple[int, int]) -> np.ndarray
        Get spectrum of a pixel (1D).
    get_bands(bands: tuple[int, int]) -> tuple[float, ...]
        Get which bands to plot for the spectrum.
    get_image(band_num: int) -> np.ndarray
        Get 2D slice of hyperspectral datacube.
    get_summary_parameter(parameter: str) -> np.ndarray
        Get summary parameter for the image.
    plot_spectrum(pixel: np.ndarray, bands: tuple[float, ...], ax: plt.Axes | None = None, title: str | None = None) -> plt.Axes
        Plot spectrum of a pixel (1D).
    plot_image(image: np.ndarray, title: str | None = None, ax: plt.Axes | None = None) -> None
        Plot 2D slice of hyperspectral datacube.
    """

    def __init__(self, image: CRISMImage | CRISMImageCotcat | CRISMImageN2N4M):
        """Initialise the Visualiser object.

        Parameters
        ----------
        image : CRISMImage | CRISMImageCotcat | CRISMImageN2N4M
            CRISMImage object to visualise.
        """
        self.image = image
        self.raw_image_bad_value_check_flag = False
        self.raw_image_copy = (
            self.image.image_array
        )  # If bad values are detected, a copy will be made and bad values replaced with np.nan.
        if self.image.ratioed_image is not None:
            self.ratioed_image_bad_value_check_flag = False
            self.ratioed_image_copy = self.image.ratioed_image

    def replace_bad_values(self, array: np.ndarray) -> None:
        """Replace bad values in the array with np.nan."""
        array[array > 1000] = np.nan
        return None

    def detect_bad_values(self, array: np.ndarray) -> bool:
        """Check for bad values in the array.
        Bad values are anything > 1000, so can be used for raw or ratioed imagery.
        Includes defined CRISM bad value of 65535.

        Returns
        -------
        bool
            True if bad values are present, False otherwise.
        """
        if np.any(array > 1000):
            return True
        return False

    def bad_value_check(self, array: np.ndarray) -> np.ndarray:
        """Check for bad values in the array.
        If any exist, the array is copied, and bad values in the copy are replaced with np.nan.
        Bad values are anything > 1000, so can be used for raw or ratioed imagery.
        Includes defined CRISM bad value of 65535.
        """
        if self.detect_bad_values(array):
            warnings.warn(
                "Bad values detected in the image. A copy has been made, values > 1000 will be replaced with np.nan."
            )
            array_copy = array.copy()
            self.replace_bad_values(array_copy)
            return array_copy
        else:
            return array

    def bad_value_check_raw_image(self) -> None:
        """Check for bad values in the raw image."""
        self.raw_image_bad_value_check_flag = True
        self.raw_image_copy = self.bad_value_check(self.raw_image_copy)
        return None

    def bad_value_check_ratioed_image(self) -> None:
        """Check for bad values in the ratioed image."""
        self.ratioed_image_bad_value_check_flag = True
        self.ratioed_image_copy = self.bad_value_check(self.ratioed_image_copy)
        return None

    def clip_image(self, image: np.ndarray, percentile: float = 99.9) -> np.ndarray:
        """Function to clip summary parameters for visualisation.
        Any negative values are set to 0, and any values above the percentile are set to the percentile value.
        Operation is done in-place.

        Parameters
        ----------
        image : np.ndarray
            Summary parameter image to clip.
        percentile : float, optional
            Percentile to clip the image at.
            Default is 99.9.

        Returns
        -------
        np.ndarray
            Clipped image.
        """
        image[image < 0] = 0
        image[image > np.nanpercentile(image, percentile)] = np.nanpercentile(
            image, percentile
        )
        return image

    def get_raw_spectrum(self, pixel_coords: tuple[int, int]) -> np.ndarray:
        """Get spectrum of a pixel (1D) from the raw image.

        Parameters
        ----------
        pixel_coords : tuple
            (x, y) coordinates of the pixel.

        Returns
        -------
        np.ndarray
            Spectrum of the pixel.
        """
        if not self.raw_image_bad_value_check_flag:
            self.bad_value_check_raw_image()
        if (
            pixel_coords[0] > self.raw_image_copy.shape[1]
            or pixel_coords[1] > self.raw_image_copy.shape[0]
        ):
            raise ValueError("Pixel coordinates out of range.")
        pixel = self.raw_image_copy[pixel_coords[1], pixel_coords[0]]
        return pixel

    def get_ratioed_spectrum(self, pixel_coords: tuple[int, int]) -> np.ndarray:
        """Get spectrum of a pixel (1D) from the ratioed image.

        Parameters
        ----------
        pixel_coords : tuple
            (x, y) coordinates of the pixel.

        Returns
        -------
        np.ndarray
            Spectrum of the pixel.
        """
        if type(self.image.ratioed_image) != np.ndarray:
            raise ValueError("No ratioed image available.")
        if not self.ratioed_image_bad_value_check_flag:
            self.bad_value_check_ratioed_image()
        if (
            pixel_coords[0] > self.image.ratioed_image.shape[1]
            or pixel_coords[1] > self.image.ratioed_image.shape[0]
        ):
            raise ValueError("Pixel coordinates out of range.")
        pixel = self.image.ratioed_image[pixel_coords[1], pixel_coords[0]]
        return pixel

    def get_bands(self, bands: tuple[int, int]) -> tuple[float, ...]:
        """Get which bands to plot for the spectrum.
        Inclusive of start and stop bands

        Parameters
        ----------
        bands : tuple
            Minimum and maximum band indices to plot.

        Returns
        -------
        tuple
            Wavelengths in ALL_WAVELENGTHS between the start and stop bands.
        """
        if (
            bands[0] < 1
            or bands[1] > self.image.num_bands
            or bands[0] > bands[1]
            or bands[0] == bands[1]
        ):
            raise ValueError("Band numbers out of range.")
        return ALL_WAVELENGTHS[
            bands[0] - 1 : bands[1] + 1
        ]  # -1 as python is 0-indexed, and +1 to be inclusive of stop.

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
            Shape is (rows, cols, 1).
        """
        if not self.raw_image_bad_value_check_flag:
            self.bad_value_check_raw_image()
        if band_num > self.image.num_bands or band_num < 0:
            raise ValueError("Band number out of range.")
        image = self.raw_image_copy[:, :, band_num]
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
        if pixel.ndim != 1 or len(pixel) != len(bands):
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
        else:
            ax.set_title(title)
        return fig, ax

    def plot_image(
        self,
        image: np.ndarray,
        title: str | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
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

        Returns
        -------
        plt.Axes
            Axes object of the plot.
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


class DenoisedVisualiser(Visualiser):
    """Class to create visualisations of denoised CRISM images.
    Inherits from Visualiser, and adds methods to visualise the denoised image and spectra.
    DenoisedVisualiser only works with CRISMImageCotcat or CRISMImageN2N4M. For basic CRISMImage, use Visualiser.

    Parameters
    ----------
    image : CRISMImageCotcat | CRISMImageN2N4M
        CRISMImageCotcat or CRISMImageN2N4M object to visualise.

    Attributes
    ----------
    image : CRISMImageCotcat | CRISMImageN2N4M
        CRISMImageCotcat or CRISMImageN2N4M object to visualise.
    denoised_bad_value_check_flag : bool
        Flag to check if bad values have been detected in the denoised image.
    denoised_copy : np.ndarray
        Copy of the denoised image used to plot. Allows for replacing bad values and clipping without changing the original image.
    ratioed_denoised_bad_value_check_flag : bool
        Flag to check if bad values have been detected in the ratioed denoised image.
    ratioed_denoised_copy : np.ndarray
        Copy of the ratioed denoised image. Allows for replacing bad values and clipping without changing the original image.

    Methods
    -------
    bad_value_check_denoised() -> None
        Check for bad values in the denoised image.
    bad_value_check_ratioed_denoised() -> None
        Check for bad values in the ratioed denoised image.
    get_denoised_spectrum(pixel_coords: tuple[int, int]) -> np.ndarray
        Get spectrum of a denoised pixel (1D).
    get_ratioed_denoised_spectrum(pixel_coords: tuple[int, int]) -> np.ndarray
        Get spectrum of a denoised pixel (1D).
    get_denoised_image(band_num: int) -> np.ndarray
        Get 2D slice of denoised hyperspectral datacube.
    """

    def __init__(self, image: CRISMImageCotcat | CRISMImageN2N4M):
        """Initialise the DenoisedVisualiser object.

        Parameters
        ----------
        image : CRISMImageCotcat | CRISMImageN2N4M
            CRISMImageCotcat or CRISMImageN2N4M object to visualise.
        """
        if type(image) == CRISMImage:
            raise ValueError(
                "DenoisedVisualiser only works with CRISMImageCotcat or CRISMImageN2N4M. For basic CRISMImage, use Visualiser."
            )
        if image.denoised_image is None:
            raise ValueError("No denoised image available.")
        super().__init__(image)
        self.denoised_bad_value_check_flag = False
        self.denoised_copy = self.image.denoised_image
        if self.image.ratioed_denoised_image is not None:
            self.ratioed_denoised_bad_value_check_flag = False
            self.ratioed_denoised_copy = self.image.ratioed_denoised_image

    def bad_value_check_denoised(self) -> None:
        """Check for bad values in the denoised image."""
        self.denoised_bad_value_check_flag = True
        self.denoised_copy = self.bad_value_check(self.denoised_copy)
        return None

    def bad_value_check_ratioed_denoised(self) -> None:
        """Check for bad values in the ratioed denoised image."""
        self.ratioed_denoised_bad_value_check_flag = True
        self.ratioed_denoised_copy = self.bad_value_check(self.ratioed_denoised_copy)
        return None

    def get_denoised_spectrum(
        self,
        pixel_coords: tuple[int, int],
    ) -> np.ndarray:
        """Get spectrum of a pixel (1D) from denoised image.

        Parameters
        ----------
        pixel_coords : tuple
            (x, y) coordinates of the pixel.

        Returns
        -------
        np.ndarray
            Spectrum of the pixel.
        """
        if not self.denoised_bad_value_check_flag:
            self.bad_value_check_denoised()
        if (
            pixel_coords[0] > self.denoised_copy.shape[1]
            or pixel_coords[1] > self.denoised_copy.shape[0]
        ):
            raise ValueError("Pixel coordinates out of range.")
        pixel = self.denoised_copy[pixel_coords[1], pixel_coords[0]]
        return pixel

    def get_ratioed_denoised_spectrum(
        self,
        pixel_coords: tuple[int, int],
    ) -> np.ndarray:
        """Get spectrum of a pixel (1D) from ratioed denoised image.

        Parameters
        ----------
        pixel_coords : tuple
            (x, y) coordinates of the pixel.

        Returns
        -------
        np.ndarray
            Spectrum of the pixel.
        """
        if type(self.image.ratioed_denoised_image) != np.ndarray:
            raise ValueError("No ratioed image available.")
        if not self.ratioed_denoised_bad_value_check_flag:
            self.bad_value_check_ratioed_denoised()
        if (
            pixel_coords[0] > self.image.ratioed_denoised_image.shape[1]
            or pixel_coords[1] > self.image.ratioed_denoised_image.shape[0]
        ):
            raise ValueError("Pixel coordinates out of range.")
        pixel = self.image.ratioed_denoised_image[pixel_coords[1], pixel_coords[0]]
        return pixel

    def get_denoised_image(self, band_num: int) -> np.ndarray:
        """Get 2D slice of denoised hyperspectral datacube.

        Parameters
        ----------
        band_num : int
            Band number to get.

        Returns
        -------
        np.ndarray
            2D slice of the hyperspectral datacube.
            Shape is (rows, cols, 1).
        """
        if not self.denoised_bad_value_check_flag:
            self.bad_value_check_denoised()
        if band_num > self.image.num_bands or band_num < 0:
            raise ValueError("Band number out of range.")
        image = self.denoised_copy[:, :, band_num]
        return image


class InteractiveVisualiser(Visualiser):
    """Class to create interactive visualisations of CRISM images in Jupyter notebooks.
    Creates a visualisation of the image and spectra, with sliders and text boxes to control the pixel and band selection.
    Inherits from Visualiser, and adds methods to create interactive visualisations of the image and spectra.

    Parameters
    ----------
    image : CRISMImage | CRISMImageCotcat | CRISMImageN2N4M
        CRISMImage object to visualise.

    Attributes
    ----------
    image : CRISMImage | CRISMImageCotcat | CRISMImageN2N4M
        CRISMImage object to visualise.
    x_slider : widgets.IntSlider
        Slider to select the x-coordinate of the pixel.
    y_slider : widgets.IntSlider
        Slider to select the y-coordinate of the pixel.
    image_band_to_display : widgets.Text
        Text box to input the band number to display.
    spectrum_band_range : widgets.Text
        Text box to input the band range to display.
    extra_image_widgets : dict
        Dictionary of extra widgets for control of the image plot.
    extra_spectrum_widgets : dict
        Dictionary of extra widgets for control of the spectrum plot.
    style : dict
        Dictionary of style settings for the widgets.
    """

    def __init__(self, image: CRISMImage):
        """Initialise the InteractiveVisualiser object."""
        super().__init__(image)
        self.style = {"description_width": "initial"}
        (
            self.x_slider,
            self.y_slider,
            self.image_band_to_display,
            self.spectrum_band_range,
        ) = self.create_base_widgets()
        self.extra_image_widgets = {}
        self.extra_spectrum_widgets = {}
        if self.image.summary_parameters is not None:
            self.create_summary_parameter_widget()
        if self.image.ratioed_image is not None:
            self.create_ratio_widget()

    def create_base_widgets(
        self,
    ) -> tuple[widgets.IntSlider, widgets.IntSlider, widgets.Text, widgets.Text]:
        """Create base widgets for the interactive plot.
        Returns
        -------
        tuple
            x_slider : widgets.IntSlider
                Slider to select the x-coordinate of the pixel.
            y_slider : widgets.IntSlider
                Slider to select the y-coordinate of the pixel.
            image_band_to_display : widgets.Text
                Text box to input the band number to display.
            spectrum_band_range : widgets.Text
                Text box to input the band range to display.
        """
        x_slider = widgets.IntSlider(
            min=0, max=self.image.spatial_dims[1] - 1, value=0, step=1, description="X:"
        )
        y_slider = widgets.IntSlider(
            min=0, max=self.image.spatial_dims[0] - 1, value=0, step=1, description="Y:"
        )
        image_band_to_display = widgets.Text(
            value="60",
            placeholder=f"1-{self.image.num_bands}",
            description="Image Band:",
            continuous_update=False,
        )
        spectrum_band_range = widgets.Text(
            value=f"1-{self.image.num_bands}",
            placeholder=f"1-{self.image.num_bands}",
            description="Spectrum Band Range:",
            continuous_update=False,
            style=self.style,
        )
        return x_slider, y_slider, image_band_to_display, spectrum_band_range

    def create_summary_parameter_widget(self) -> None:
        """Create widget to allow dropdown selection of summary parameters for the image plot.
        Summary parameters have no bands, so if dropdown is not raw, disable the band widget.
        """
        dropdown_options = list(self.image.summary_parameters.keys())
        dropdown_options.append("Raw")  # Always have the option to plot the raw image.
        summary_parameter_dropdown = widgets.Dropdown(
            options=dropdown_options,
            value="Raw",
            description="Image Options:",
            style=self.style,
        )
        summary_parameter_dropdown.observe(
            self.enable_image_band_widget, names="value"
        )  # If dropdown is not raw, disable the band widget.
        self.extra_image_widgets["dropdown"] = summary_parameter_dropdown

    def create_ratio_widget(self) -> None:
        """Create widget to allow selection of raw or ratioed spectrum for the spectrum plot."""
        ratio_button = widgets.ToggleButtons(
            options=["Raw", "Ratioed"],
            button_style="",
            value="Raw",
            description="Spectrum Type:",
            style=self.style,
        )
        self.extra_spectrum_widgets["ratio"] = ratio_button

    def enable_image_band_widget(self, change):
        """Summary parameters have no bands, so if dropdown is not raw, disable the band widget."""
        if change.new == "Raw":
            self.image_band_to_display.disabled = False
            self.image_band_to_display.value = "60"
        else:
            self.image_band_to_display.disabled = True
            self.image_band_to_display.value = "N/A"

    def format_spectrum_band_range(self, input_str: str):
        """Format the input string for the spectrum band range."""
        start_band, stop_band = input_str.split("-")
        return int(start_band), int(stop_band)

    def box_image_controls(self) -> VBox:
        """Create a VBox of the image controls."""
        return VBox(
            [
                self.x_slider,
                self.y_slider,
                self.image_band_to_display,
                *self.extra_image_widgets.values(),
            ]
        )

    def box_spectrum_controls(self) -> VBox:
        """Create a VBox of the spectrum controls."""
        return VBox([self.spectrum_band_range, *self.extra_spectrum_widgets.values()])

    def get_image_update(
        self,
        x: int,
        y: int,
        image_band: str,
        **kwargs,
    ) -> np.ndarray:
        """Get the next image to plot.

        Parameters
        ----------
        x : int
            x-coordinate of the pixel of spectrum to plot.
        y : int
            y-coordinate of the pixel of spectrum to plot.
        image_band : str
            Band number to display.
        kwargs : dict
            Extra widgets for control of the image plot.

        Returns
        -------
        np.ndarray
            Image to plot.
        """
        if "dropdown" in kwargs and kwargs["dropdown"] != "Raw":
            image = self.get_summary_parameter(kwargs["dropdown"])
        else:  # Default is must have at least passed a raw image.
            if (
                image_band == ""
                or int(image_band) > self.image.num_bands
                or int(image_band) < 1
            ):
                band_num = 59
            else:
                band_num = int(image_band) - 1  # -1 as python is 0-indexed
            image = self.get_image(band_num)
        return image

    def get_spectrum_update(
        self,
        x: int,
        y: int,
        spectrum_range: str,
        **kwargs,
    ) -> tuple[np.ndarray, tuple[float, ...]]:
        """Get the next spectrum to plot.

        Parameters
        ----------
        x : int
            x-coordinate of the pixel of spectrum to plot.
        y : int
            y-coordinate of the pixel of spectrum to plot.
        spectrum_range : str
            Band range to display.
        kwargs : dict
            Extra widgets for control of the spectrum plot.

        Returns
        -------
        tuple
            pixel : np.ndarray
                Spectrum of the pixel.
            bands : tuple
                Wavelengths of the spectrum.
        """
        pixel = self.get_raw_spectrum((x, y))
        if "ratio" in kwargs and kwargs["ratio"] == "Ratioed":
            pixel = self.get_ratioed_spectrum((x, y))
        band_range = self.format_spectrum_band_range(spectrum_range)
        bands = self.get_bands(band_range)
        pixel = pixel[band_range[0] - 1 : band_range[1] + 1]
        return pixel, bands

    def update_plots(
        self,
        x: int,
        y: int,
        spectrum_range: str,
        image_band: str,
        **kwargs,
    ) -> plt.Figure:
        """Redraw the image and spectrum plots.

        Parameters
        ----------
        x : int
            x-coordinate of the pixel of spectrum to plot.
        y : int
            y-coordinate of the pixel of spectrum to plot.
        spectrum_range : str
            Band range to display for spectrum.
        image_band : str
            Band number to display for image.
        kwargs : dict
            Extra widgets for control of both the image and spectrum plots.

        Returns
        -------
        plt.Figure
            Figure object of the plots.
        """
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        # Image Plot
        image = self.get_image_update(x, y, image_band, **kwargs)
        self.plot_image(image, ax=ax[0])
        ax[0].scatter(x, y, marker="x", color="red", label="Selected Pixel")

        # Spectrum Plot
        pixel, bands = self.get_spectrum_update(x, y, spectrum_range, **kwargs)
        self.plot_spectrum(
            pixel,
            bands=bands,
            ax=ax[1],
            title=f"{self.image.im_name} - Spectrum at ({x}, {y})",
        )
        plt.tight_layout()
        return fig, ax

    def interactive_plot(self) -> interactive | VBox | HBox:
        """Create an interactive plot of the image and its spectra.
        Wraps update_plots in an interactive widget, and adds the image and spectrum controls.

        Returns
        -------
        interactive | VBox | HBox
            Interactive plot of the image and its spectra.
        """
        interactive_plot = interactive(
            self.update_plots,
            x=self.x_slider,
            y=self.y_slider,
            spectrum_range=self.spectrum_band_range,
            image_band=self.image_band_to_display,
            **self.extra_spectrum_widgets,
            **self.extra_image_widgets,
        )
        image_controls = self.box_image_controls()
        spectrum_controls = self.box_spectrum_controls()
        all_controls = HBox([image_controls, spectrum_controls])
        output = interactive_plot.children[-1]
        return VBox([all_controls, output])


class DenoisedInteractiveVisualiser(DenoisedVisualiser, InteractiveVisualiser):
    """Class to create interactive visualisations of denoised CRISM images.
    Inherits from DenoisedVisualiser and InteractiveVisualiser, and adds methods to create interactive visualisations of the denoised image and spectra.
    Overwrites __init__ to only work with CRISMImageCotcat or CRISMImageN2N4M, and add extra widgets for control of plotting the denoised spectra in addition to the raw spectra.
    Overwrites update_plots to include the extra widgets for control of plotting the denoised spectra.

    Parameters
    ----------
    image : CRISMImageCotcat | CRISMImageN2N4M
        CRISMImageCotcat or CRISMImageN2N4M object to visualise.

    Attributes
    ----------
    image : CRISMImageCotcat | CRISMImageN2N4M
        CRISMImageCotcat or CRISMImageN2N4M object to visualise.
    Others as per DenoisedVisualiser and InteractiveVisualiser.

    Methods
    -------
    create_spectrum_plot_options_widgets() -> None
        Create widgets for picking raw or denoised spectra.
    get_denoised_spectrum_update(x: int, y: int, spectrum_range: str, image_band: str, **kwargs) -> tuple[np.ndarray, tuple[float, ...]]
        Get the next denoised spectrum to plot.
    update_plots(x: int, y: int, spectrum_range: str, image_band: str, **kwargs) -> plt.Figure
        Redraw the image and spectrum plots.
    Other methods as per DenoisedVisualiser and InteractiveVisualiser.
    """

    def __init__(self, image: CRISMImageCotcat | CRISMImageN2N4M):
        """Initialise the DenoisedInteractiveVisualiser object."""
        if type(image) == CRISMImage:
            raise ValueError(
                "DenoisedVisualiser only works with CRISMImageCotcat or CRISMImageN2N4M. For basic CRISMImage, use Visualiser."
            )
        if image.denoised_image is None:
            raise ValueError(
                "No denoised image available. Please denoise the image first or use InteractiveVisualiser."
            )
        super().__init__(image)
        self.style = {"description_width": "initial"}
        (
            self.x_slider,
            self.y_slider,
            self.image_band_to_display,
            self.spectrum_band_range,
        ) = self.create_base_widgets()
        self.extra_image_widgets = {}
        self.extra_spectrum_widgets = {}
        self.create_spectrum_plot_options_widgets()
        # In theory could have the denoised image but not the ratioed denoised image, and dynamically make the ratio toggle available based on which spectra are selected to plot, but for now, just raise an error.
        if (
            type(image.ratioed_image) == np.ndarray
            and image.ratioed_denoised_image is None
        ):
            raise ValueError(
                "No denoised ratioed image available. If you wish to plot any ratioed spectra, please ratio the denoised image."
            )
        else:
            self.create_ratio_widget()
        if self.image.summary_parameters is not None:
            self.create_summary_parameter_widget()

    def create_spectrum_plot_options_widgets(self) -> None:
        """Create widgets for the spectrum plot options.
        Create checkboxes for the raw and denoised spectra.
        Raw spectrum is checked by default, denoised spectrum is not.
        """
        self.extra_spectrum_widgets["Original Spectrum"] = widgets.Checkbox(
            value=True,
            description="Original Spectrum",
            style=self.style,
        )
        self.extra_spectrum_widgets["Denoised Spectrum"] = widgets.Checkbox(
            value=False,
            description="Denoised Spectrum",
            style=self.style,
        )

    def get_denoised_spectrum_update(
        self,
        x: int,
        y: int,
        spectrum_range: str,
        **kwargs,
    ) -> tuple[np.ndarray, tuple[float, ...]]:
        """Get the next denoised spectrum to plot.

        Parameters
        ----------
        x : int
            x-coordinate of the pixel of spectrum to plot.
        y : int
            y-coordinate of the pixel of spectrum to plot.
        spectrum_range : str
            Band range to display for spectrum.
        kwargs : dict
            Extra widgets for control of the spectrum plot.
        """
        pixel = self.get_denoised_spectrum((x, y))
        if "ratio" in kwargs and kwargs["ratio"] == "Ratioed":
            pixel = self.get_ratioed_denoised_spectrum((x, y))
        band_range = self.format_spectrum_band_range(spectrum_range)
        bands = self.get_bands(band_range)
        pixel = pixel[band_range[0] - 1 : band_range[1] + 1]
        return pixel, bands

    def update_plots(
        self,
        x: int,
        y: int,
        spectrum_range: str,
        image_band: str,
        **kwargs,
    ) -> plt.Figure:
        """Redraw the image and spectrum plots.

        Parameters
        ----------
        x : int
            x-coordinate of the pixel of spectrum to plot.
        y : int
            y-coordinate of the pixel of spectrum to plot.
        spectrum_range : str
            Band range to display for spectrum.
        image_band : str
            Band number to display for image.
        kwargs : dict
            Extra widgets for control of both the image and spectrum plots.

        Returns
        -------
        plt.Figure
            Figure object of the plots.
        """
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        # Image Plot
        image = self.get_image_update(x, y, image_band, **kwargs)
        self.plot_image(image, ax=ax[0])
        ax[0].scatter(x, y, marker="x", color="red", label="Selected Pixel")

        # Spectrum Plot
        # Plot whichever checkbox options are selected on the same axes.
        legend = []
        if self.extra_spectrum_widgets["Original Spectrum"].value:
            pixel, bands = self.get_spectrum_update(x, y, spectrum_range, **kwargs)
            self.plot_spectrum(
                pixel,
                bands=bands,
                ax=ax[1],
                title=f"{self.image.im_name} - Spectrum at ({x}, {y})",
            )
            legend.append("Original Spectrum")
        if self.extra_spectrum_widgets["Denoised Spectrum"].value:
            pixel, bands = self.get_denoised_spectrum_update(
                x, y, spectrum_range, **kwargs
            )
            self.plot_spectrum(
                pixel,
                bands=bands,
                ax=ax[1],
                title=f"{self.image.im_name} - Denoised Spectrum at ({x}, {y})",
            )
            legend.append("Denoised Spectrum")
        ax[1].legend(legend)
        plt.tight_layout()
        return fig, ax
