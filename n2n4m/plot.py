import numpy as np
import warnings
import matplotlib.pyplot as plt
from n2n4m.crism_image import CRISMImage # , CRISMImageCotcat, CRISMImageN2N4M
from n2n4m.wavelengths import ALL_WAVELENGTHS
# from ipywidgets import widgets, interactive, HBox, VBox


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

    def __init__(self, image: CRISMImage):# | CRISMImageCotcat | CRISMImageN2N4M):
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
        pixel = self.ratioed_image_copy[pixel_coords[1], pixel_coords[0]]
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
