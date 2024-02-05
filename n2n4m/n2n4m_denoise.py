# Module to run N2N denoising across an entire image. Should allow for CLI usage. Input trained model as instance of Noise2Noise1D class, with weights loaded, image, and output numpy array of denoised image, with extra bands inserted back from original image.
import numpy as np
import pickle
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator
from torch import load as load_model
from torch import device, Tensor
from torch.utils.data import TensorDataset, DataLoader
import pkg_resources


from n2n4m.model_functions import predict, check_available_device
from n2n4m.model import Noise2Noise1D
from n2n4m.io import load_image
from n2n4m.wavelengths import ALL_WAVELENGTHS, PLEBANI_WAVELENGTHS
from crism_ml.io import image_shape

DEFAULT_MODEL_FILEPATH = pkg_resources.resource_filename(
    "n2n4m", "data/trained_model_weights.pt"
)
DEFAULT_SCALER_FILEPATH = pkg_resources.resource_filename(
    "n2n4m", "data/input_standardiser.pkl"
)


def band_index_mask(
    bands_to_keep: tuple = PLEBANI_WAVELENGTHS,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a mask for the indices of bands to keep out of all CRISM L sensor bands.

    Parameters
    ----------
    bands_to_keep : tuple, optional
        The wavelengths of bands to keep.
        Default PLEBANI_WAVELENGTHS

    Returns
    -------
    include_bands_indices : np.ndarray
        The indices of the bands to keep.
    exclude_bands_indices : np.ndarray
        The indices of the bands to exclude.
    """
    band_mask = np.isin(ALL_WAVELENGTHS, bands_to_keep)
    include_bands_indices = np.where(band_mask)[0]
    exclude_bands_indices = np.where(~band_mask)[0]
    return include_bands_indices, exclude_bands_indices


def clip_bands(
    spectra: np.ndarray, bands_to_keep: tuple = PLEBANI_WAVELENGTHS
) -> tuple[np.ndarray, np.ndarray]:
    """Clip the bands of the spectra to the bands specified in bands_to_keep. Returns the interior bands and the exterior bands.

    Parameters
    ----------
    spectra : np.ndarray
        The spectra to clip.
        Shape (n_samples, n_bands=ALL_WAVELENGTHS)
    bands_to_keep : tuple, optional
        The bands to keep.
        Default PLEBANI_WAVELENGTHS

    Returns
    -------
    interior_bands : np.ndarray
        The spectra with the bands specified in bands_to_keep.
        Shape (n_samples, n_bands)
    exterior_bands : np.ndarray
        The spectra with the inverse bands specified in bands_to_keep.
        Shape (n_samples, n_bands)
    """
    include_bands_indices, exclude_bands_indices = band_index_mask(bands_to_keep)
    interior_bands = spectra[:, include_bands_indices]
    exterior_bands = spectra[:, exclude_bands_indices]
    return interior_bands, exterior_bands


def combine_bands(
    clipped_data: np.ndarray,
    extra_data: np.ndarray,
    bands_to_keep: tuple = PLEBANI_WAVELENGTHS,
) -> np.ndarray:
    """Combine the interior and exterior bands back into the original shape of the spectra.

    Parameters
    ----------
    clipped_data : np.ndarray
        The spectra with the bands specified in bands_to_keep.
        Shape (n_samples, n_bands)
    extra_data : np.ndarray
        The spectra with the inverse bands specified in bands_to_keep.
        Shape (n_samples, n_bands)
    bands_to_keep : tuple, optional
        The bands corresposnding to the clipped_data.
        Default PLEBANI_WAVELENGTHS

    Returns
    -------
    spectra : np.ndarray
        The spectra with the bands combined.
        Shape (n_samples, n_bands=ALL_WAVELENGTHS)
    """
    spectra = np.zeros((clipped_data.shape[0], len(ALL_WAVELENGTHS)))
    include_bands_indices, exclude_bands_indices = band_index_mask(bands_to_keep)
    spectra[:, include_bands_indices] = clipped_data
    spectra[:, exclude_bands_indices] = extra_data
    return spectra


def load_scaler(filepath: str) -> BaseEstimator:
    """Load a fitted sklearn scaler object from a pickle file.

    Parameters
    ----------
    filepath : str
        The filepath to the pickle file.

    Returns
    -------
    scaler : object
        The loaded scaler object.
    """
    with open(filepath, "rb") as f:
        scaler = pickle.load(f)

    # check if the loaded object is a scaler and if it is fitted
    try:
        check_is_fitted(scaler)
    except NotFittedError:
        raise NotFittedError(f"The scaler loaded from: {filepath} is not fitted.")
    return scaler


def instantiate_default_model(filepath: str) -> Noise2Noise1D:
    """Load a trained Noise2Noise1D model from a file.

    Parameters
    ----------
    filepath : str
        The filepath to the model weights.

    Returns
    -------
    model : Noise2Noise1D
        The loaded model.
    """
    model = Noise2Noise1D(kernel_size=5, depth=3, num_blocks=4, num_input_features=350)
    model_state_dict = load_model(
        filepath, map_location=device(check_available_device())
    )
    new_model_state_dict = {
        k.replace("module.", ""): v for k, v in model_state_dict.items()
    }  # Model was trained on multiple GPUs, so need to remove "module." from keys
    model.load_state_dict(new_model_state_dict)
    model.eval()
    return model


def create_dataloader(spectra: np.ndarray, batch_size: int = 1000) -> DataLoader:
    """Create a DataLoader from the spectra.

    Parameters
    ----------
    spectra : np.ndarray
        The spectra to create a DataLoader from.
        Shape (n_samples, n_bands)
    batch_size : int, optional
        The batch size for the DataLoader.
        Default 1000

    Returns
    -------
    data_loader : DataLoader
        The DataLoader for the spectra.
    """
    dataset = TensorDataset(Tensor(spectra))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


def denoise_image(
    image_filepath: str,
    scaler_filepath: str = DEFAULT_SCALER_FILEPATH,
    model: Noise2Noise1D | None = None,
    batch_size: int = 1000,
) -> np.ndarray:
    """Denoise an image using a trained N2N model.

    Parameters
    ----------
    image_filepath : str
        The filepath to the image to denoise.
        Must be a CRISM L sensor TRR3 .img file.
    scaler_filepath : str, optional
        The filepath to the fitted sklearn scaler object pickle used to scale the spectra.
        Default None, the default scaler (Robust) will be applied.
    model : Noise2Noise1D, optional
        The trained Noise2Noise1D model.
        Default None, the default model will be loaded and used.
    batch_size : int, optional
        The batch size for the DataLoader.
        Default 1000
    """

    image = load_image(image_filepath)
    im_shape = image_shape(image)
    spectra = image["IF"]
    bands_to_denoise, additional_bands = clip_bands(spectra)

    scaler = load_scaler(scaler_filepath)
    bands_to_denoise = scaler.transform(bands_to_denoise)

    if model is None:
        model = instantiate_default_model(DEFAULT_MODEL_FILEPATH)

    spectra_dataloader = create_dataloader(bands_to_denoise, batch_size=batch_size)

    denoised_spectra = predict(
        model, spectra_dataloader, device(check_available_device())
    )
    if check_available_device() == "cuda":
        denoised_spectra = denoised_spectra.cpu().numpy()
    else:
        denoised_spectra = denoised_spectra.numpy()
    denoised_spectra = combine_bands(denoised_spectra, additional_bands)
    return denoised_spectra
