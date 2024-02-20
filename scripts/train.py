#!/home/rp1818/miniconda3/envs/CRISM_env_3/bin/python3
# This is a script to train a Noise2Noise model on the CRISM data.
# Requires the training data to be consolidated as JSON files in the data/extracted_mineral_pixel_data and data/extracted_bland_pixel_data directories.
# You can use the bland_dataset_collation.py and mineral_dataset_collation.py scripts to generate these files.
# Tuneable parameters are at the top of the script.
# Model weights are saved under data/{MODEL_NAME}_weights.pt
# Training curves are saved under data/{MODEL_NAME}_training_curve.png

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import n2n4m.preprocessing as preprocessing
from n2n4m.wavelengths import PLEBANI_WAVELENGTHS
from n2n4m.model_functions import check_available_device, train, validate
from n2n4m.model import Noise2Noise1D

PARENT_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(PARENT_DIR, "data")

BATCH_SIZE = 256  # Number of pixels to process at once.
N_EPOCHS = 100  # Number of times to loop through the training data.
LR = 0.001  # Learning rate for the model.
PATIENCE = 10  # Number of epochs to wait before stopping training if the validation loss does not improve.

KERNEL_SIZE = 5  # Size of the kernels used in convolutional layers.
DEPTH = 3  # Number of convolutional layers per upsample or downsample block.
NUM_BLOCKS = 4  # Number of upsample and downsample blocks in the model.
NUM_BLAND_PIXELS = 150_000  # Number of bland pixels added to the dataset.

MODEL_NAME = "N2N4M"  # Name of the model to be saved.


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False  # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
    torch.backends.cudnn.enabled = False

    return True


def run_training(
    model_name: str,
    kernel_size: int,
    depth: int,
    num_blocks: int,
    num_bland_pixels: int,
):
    """
    Wrapper around the model training process.
    """

    weight_filename = model_name + "_weights.pt"
    weights_filepath = os.path.join(DATA_DIR, weight_filename)
    training_curve_filename = model_name + "_training_curve.png"
    training_curve_filepath = os.path.join(DATA_DIR, training_curve_filename)

    set_seed(42)  # Fix seed for reproduceability.
    # Read the data.
    mineral_dataset_path = os.path.join(
        DATA_DIR, "extracted_mineral_pixel_data", "mineral_pixel_dataset.json"
    )
    bland_dataset_path = os.path.join(
        DATA_DIR, "extracted_bland_pixel_data", "bland_pixel_dataset.json"
    )
    mineral_dataset = preprocessing.load_dataset(mineral_dataset_path)
    bland_dataset = preprocessing.load_dataset(bland_dataset_path)

    # Get as many bland pixels from the bland pixel set as desired.
    # Sample equally from each image of bland pixels.
    num_bland_images = bland_dataset["Image_Name"].nunique()
    samples_per_image = num_bland_pixels // num_bland_images
    bland_dataset_sample = (
        bland_dataset.groupby("Image_Name")
        .apply(lambda x: x.sample(min(len(x), samples_per_image), random_state=42))
        .reset_index(drop=True)
    )

    # Combine the bland and mineral datasets, then apply all preprocessing steps.
    dataset = pd.concat(
        [mineral_dataset, bland_dataset_sample], ignore_index=True
    ).reset_index(drop=True)
    dataset = preprocessing.expand_dataset(dataset)
    dataset = preprocessing.drop_bad_bands(dataset, bands_to_keep=PLEBANI_WAVELENGTHS)
    dataset = preprocessing.impute_bad_values(dataset, threshold=1)
    dataset = preprocessing.impute_atmospheric_artefacts(
        dataset, wavelengths=PLEBANI_WAVELENGTHS
    )
    noise_dataset = preprocessing.generate_noisy_pixels(
        dataset.iloc[:, 3:], random_seed=42
    )
    input_target_dataset = pd.concat([dataset, noise_dataset], axis=1)
    train_set, test_set = preprocessing.train_test_split(
        input_target_dataset, bland_pixels=True
    )
    train_set, validation_set = preprocessing.train_validation_split(
        train_set, bland_pixels=True
    )

    # Split the training, validation, and testing sets.
    X_train, y_train, ancillary_train = preprocessing.split_features_targets_anciliary(
        train_set
    )
    X_test, y_test, ancillary_test = preprocessing.split_features_targets_anciliary(
        test_set
    )
    X_validation, y_validation, ancillary_validation = (
        preprocessing.split_features_targets_anciliary(validation_set)
    )

    # Fit a scaler to the training data, and then apply it to the validation and test data.
    X_train, feature_scaler = preprocessing.standardise(X_train, method="RobustScaler")
    X_test, _ = preprocessing.standardise(
        X_test, method="RobustScaler", scaler=feature_scaler
    )
    X_validation, _ = preprocessing.standardise(
        X_validation, method="RobustScaler", scaler=feature_scaler
    )

    # Convert the data to torch TensorDatasets
    X_train_tensor = torch.from_numpy(X_train.values).float()
    y_train_tensor = torch.from_numpy(y_train.values).float()
    X_test_tensor = torch.from_numpy(X_test.values).float()
    y_test_tensor = torch.from_numpy(y_test.values).float()
    X_validation_tensor = torch.from_numpy(X_validation.values).float()
    y_validation_tensor = torch.from_numpy(y_validation.values).float()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # Instantiate a model
    model = Noise2Noise1D(
        kernel_size=kernel_size,
        depth=depth,
        num_blocks=num_blocks,
        num_input_features=len(PLEBANI_WAVELENGTHS),
    )
    # Send model to device
    device = check_available_device()
    if device == "cuda":
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    model.to(device)

    # Create loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Create loss trackers
    best_loss = np.inf
    last_5_losses = [np.inf, np.inf, np.inf, np.inf, np.inf]
    training_losses = []
    validation_losses = []
    current_patience = PATIENCE
    # Loop training epochs
    for epoch in range(N_EPOCHS):
        print(f"Running epoch {epoch+1}", flush=True)
        train_loss = train(model, optimizer, criterion, train_loader, device)
        val_loss = validate(model, criterion, validation_loader, device)
        # Weights are saved if the validation loss is the best so far.
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), weights_filepath)
            current_patience = PATIENCE
        # Early stopping
        if val_loss >= np.mean(last_5_losses):
            current_patience -= 1
            if current_patience == 0:
                print("Early stopping!", flush=True)
                break
        # Update loss trackers
        last_5_losses.pop(0)
        last_5_losses.append(val_loss)

        training_losses.append(train_loss)
        validation_losses.append(val_loss)

        # Save the training curve
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(training_losses, label="Training loss")
        ax.plot(validation_losses, label="Validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        plt.savefig(training_curve_filepath)
        plt.close()

    print(
        f"{model_name} training complete! With kernel size {kernel_size}, depth {depth}, and num_blocks {num_blocks}",
        flush=True,
    )
    print(
        f"{model_name} best validation loss: {min(validation_losses)}, with training loss: {training_losses[validation_losses.index(min(validation_losses))]}",
        flush=True,
    )


run_training(
    model_name=MODEL_NAME,
    kernel_size=KERNEL_SIZE,
    depth=DEPTH,
    num_blocks=NUM_BLOCKS,
    num_bland_pixels=NUM_BLAND_PIXELS,
)
