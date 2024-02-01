import os
import matplotlib.pyplot as plt


from n2n4m import io
from n2n4m.wavelengths import ALL_WAVELENGTHS, PLEBANI_WAVELENGTHS
from crism_ml.io import image_shape


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

    def single_band_plot(self, ax: plt.Axes = None, band: int = 0) -> None:
        pass

    def false_colour_plot(self, ax: plt.Axes = None) -> None:
        pass

    def plot_spectrum(
        self,
        pixel: tuple,
        ax: plt.Axes = None,
        range: tuple | None = None,
        bands: tuple = ALL_WAVELENGTHS,
    ) -> None:
        pass

    def plot_summary_parameter(self, parameter: str, ax: plt.Axes = None) -> None:
        pass
