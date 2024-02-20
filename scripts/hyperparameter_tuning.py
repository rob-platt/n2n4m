import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import TensorDataset, DataLoader
import ray
from ray import air, tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.air.checkpoint import Checkpoint

import n2n4m.preprocessing as preprocessing
from n2n4m.model import Noise2Noise1D
from n2n4m.model_functions import train, validate, check_available_device
from n2n4m.wavelengths import PLEBANI_WAVELENGTHS

# Get the data directory
PARENT_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(PARENT_DIR, "data")
CHECKPOINT_DIR = os.path.join(PARENT_DIR, "model_checkpoints")
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# Fixed hyperparameters
N_EPOCHS=100
NUM_BLAND_PIXELS=150_000

def train_N2N4M(config: dict):
    """
    Wrapper function to train the Noise2Noise4Minerals model with the given configuration. 
    Allows for parallel training of multiple models to speed hyperparameter tuning.
    Model checkpoints saved to disk are used to restore training if the process is interrupted.

    Parameters
    ----------
    config : dict
        Dictionary containing the hyperparameters to be used for training the model. The keys are:
        - "lr" : float
            The learning rate for the model.
        - "batchsize" : int
            The batch size to use for training.
        - "kernel_size" : int
            The size of the kernel to use in the convolutional layers of the model.
        - "depth" : int
            The number of convolutional layers in each upsampling and downsampling block of the model.
        - "num_blocks" : int
            The number of upsampling and downsampling blocks to use in the model.
    
        
    """
    lr = config["lr"]
    batch_size = config["batchsize"] 

    mineral_dataset_path = os.path.join(DATA_DIR, "extracted_mineral_pixel_data", "mineral_pixel_dataset.json")
    bland_dataset_path = os.path.join(DATA_DIR, "extracted_bland_pixel_data", "bland_pixel_dataset.json")

    mineral_dataset = preprocessing.load_dataset(mineral_dataset_path)
    bland_dataset = preprocessing.load_dataset(bland_dataset_path)

    # Get as many bland pixels from the bland pixel set as desired.
    # Sample equally from each image of bland pixels.
    num_bland_images = bland_dataset["Image_Name"].nunique()
    samples_per_image = NUM_BLAND_PIXELS // num_bland_images
    bland_dataset_sample = bland_dataset.groupby("Image_Name").apply(lambda x: x.sample(min(len(x), samples_per_image), random_state=42)).reset_index(drop=True)

    # Combine the bland and mineral datasets, then apply all preprocessing steps.
    dataset = pd.concat([mineral_dataset, bland_dataset_sample], ignore_index=True).reset_index(drop=True)
    dataset = preprocessing.expand_dataset(dataset)
    dataset = preprocessing.drop_bad_bands(dataset, bands_to_keep=PLEBANI_WAVELENGTHS)
    dataset = preprocessing.impute_bad_values(dataset, threshold=1)
    dataset = preprocessing.impute_atmospheric_artefacts(dataset, wavelengths=PLEBANI_WAVELENGTHS) 
    noise_dataset = preprocessing.generate_noisy_pixels(dataset.iloc[:,3:], random_seed=42)
    input_target_dataset = pd.concat([dataset, noise_dataset], axis=1)
    train_set, test_set = preprocessing.train_test_split(input_target_dataset, bland_pixels=True)
    train_set, validation_set = preprocessing.train_validation_split(train_set, bland_pixels=True)

    # Split the training, validation, and testing sets. 
    X_train, y_train, ancillary_train = preprocessing.split_features_targets_anciliary(train_set)
    X_test, y_test, ancillary_test = preprocessing.split_features_targets_anciliary(test_set)
    X_validation, y_validation, ancillary_validation = preprocessing.split_features_targets_anciliary(validation_set)

    # Fit a scaler to the training data, and then apply it to the validation and test data.
    X_train, feature_scaler = preprocessing.standardise(X_train, method="RobustScaler")
    X_test, _ = preprocessing.standardise(X_test, method="RobustScaler", scaler=feature_scaler)
    X_validation, _ = preprocessing.standardise(X_validation, method="RobustScaler", scaler=feature_scaler)

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    model = Noise2Noise1D(kernel_size=config["kernel_size"], depth=config["depth"], num_input_features=350, num_blocks=config["num_blocks"])

    device = check_available_device()
    if device == "cuda":
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    model.to(device)

    # Restore from checkpoint if exists.
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
            model.load_state_dict(model_state)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    best_loss = np.inf
    for epoch in range(N_EPOCHS):
        train_loss = train(model, optimizer, criterion, train_loader, device)
        val_loss = validate(model, criterion, validation_loader, device)       
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "checkpoint.pt"))
            checkpoint = Checkpoint.from_directory(CHECKPOINT_DIR)
        session.report({"val_loss": val_loss, "train_loss": train_loss}, checkpoint=checkpoint)

    return

# Hyperparameter search space
search_space = {
    "feature_scaler": tune.grid_search(["StandardScaler", "MinMaxScaler", "RobustScaler"]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batchsize": tune.choice([32, 64, 128, 256, 512]),
    "kernel_size": tune.choice([3, 5, 7, 9, 11]),
    "depth": tune.choice([1, 2, 3, 4, 5]),
    "num_blocks": tune.choice([1, 2, 3, 4, 5])
}

# Tuning with 0.5 GPU and 5 CPU cores per trial, using Asynchronous HyperBand Scheduler, 10 random samples of the search space.
tuner = tune.Tuner(tune.with_resources(train_N2N4M, resources={"gpu":0.5, "cpu": 5}), 
                   tune_config=tune.TuneConfig(scheduler=ASHAScheduler(metric="val_loss", mode="min", grace_period=10), num_samples=10), 
                   param_space=search_space)

# Run the tuning
results = tuner.fit()

# Get the "best" result according to the validation loss
best_result = results.get_best_result(metric="val_loss", mode="min")
best_config = best_result.config
best_metrics = best_result.metrics
print(f"Best config: {best_config}")
print(f"Best metrics: {best_metrics}")