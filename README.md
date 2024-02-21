## Noise 2 Noise For Mars (N2N4M)
### Official repository for the ICLR ML4RS 2024 paper "Noise2Noise Denoising of CRISM Hyperspectral Data"

#### Abstract
Hyperspectral data acquired by the Compact Reconnaissance Imaging Spectrometer for Mars (CRISM) have allowed for unparalleled mapping of the surface
mineralogy of Mars. Due to sensor degradation over time, a significant portion of the recently acquired data is considered unusable. Here a data-driven model,
Noise2Noise4Mars (N2N4M), is introduced to remove noise from CRISM images. We demonstrate its strong performance on synthetic noise data and CRISM
images, and its impact on downstream classification performance, outperforming the benchmark method on most metrics. This should allow for detailed analysis
for critical sites of interest on the Martian surface, including proposed lander sites.
#### Example image
![alt text](https://github.com/rob-platt/N2N4M/blob/main/notebooks/n2n4m_results/ATU0003561F_denoising_example_image.png)
#### Example spectrum
![alt text](https://github.com/rob-platt/N2N4M/blob/main/notebooks/n2n4m_results/ATU0003561F_denoising_example_spectrum.png)

#### Introduction

This package uses the N2N4M neural network to denoise [CRISM](http://crism.jhuapl.edu/) L sensor SWIR data.  
The code offers the following functionality:
* Apply the N2N4M model to denoise CRISM data
* Plot static and interactive visualisations of CRISM images and spectra
* Calculate summary parameters
* Ratio images using the HBM of [Plebani et al. (2022)](https://github.com/Banus/crism_ml)
* Read and write CRISM data in .img format, so that denoised images can then be map-projected
* Preprocess data and train the N2N4M model
* Evaluate the performance of the N2N4M model

#### Usage
The package is designed for use in Jupyter Notebooks. It requires python 3.8 or later, and should run on Windows and Linux.
GPU acceleration is used where available.  
To create a new environment and install the package, run the following commands in the terminal:

```bash
conda create -n n2n4m python=3.8
conda activate n2n4m
pip install -e .
```
If you wish to use the exact environment used to develop the package, you can install the environment from the environment.yml file:
```bash
conda env create -f environment.yml
conda activate n2n4m
```
To then install the package, please remove the line ' "torch >= 2.0", ' from the setup.py file, and run the following command:
```bash
pip install -e .
```


