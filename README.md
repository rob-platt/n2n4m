## Noise 2 Noise For Mars (N2N4M)
### Official repository for the ICLR ML4RS 2024 paper "Noise2Noise Denoising of CRISM Hyperspectral Data"

#### Abstract
Hyperspectral data acquired by the Compact Reconnaissance Imaging Spectrometer for Mars (CRISM) have allowed for unparalleled mapping of the surface
mineralogy of Mars. Due to sensor degradation over time, a significant portion of the recently acquired data is considered unusable. Here a data-driven model,
Noise2Noise4Mars (N2N4M), is introduced to remove noise from CRISM images. We demonstrate its strong performance on synthetic noise data and CRISM
images, and its impact on downstream classification performance, outperforming the benchmark method on most metrics. This should allow for detailed analysis
for critical sites of interest on the Martian surface, including proposed lander sites.
#### Example image
![alt text](https://github.com/rob-platt/N2N4M/blob/main/docs/ATU0003561F_denoising_example_image.png)
#### Example spectrum
![alt text](https://github.com/rob-platt/N2N4M/blob/main/docs/ATU0003561F_denoising_example_spectrum.png)

#### Introduction

This package uses the N2N4M neural network to denoise [CRISM](http://crism.jhuapl.edu/) L sensor SWIR data.  
The code offers the following functionality:
* Apply the N2N4M model to denoise CRISM data
* Plot static and interactive visualisations of CRISM images and spectra
* Calculate summary parameters
* Ratio images using the HBM of [Plebani et al. (2022)](https://github.com/Banus/crism_ml) [1]
* Read and write CRISM data in .img format, so that denoised images can then be map-projected
* Preprocess data and train the N2N4M model
* Evaluate the performance of the N2N4M model
* The Complement to CRISM Analysis Toolkit (CoTCAT) [2] denoising method has also been implemented for comparison

#### Usage
The package is designed for use in Jupyter Notebooks. It requires python 3.10 or later, and should run on Windows and Linux.
GPU acceleration is used where available.  
To create a new environment and install the package, run the following commands in the terminal:

```bash
conda create -n n2n4m python=3.11
conda activate n2n4m
pip install -e .
```
If you wish to use the exact environment used to develop the package, you can install the environment from the environment.yml file:
```bash
conda env create -f environment.yml
conda activate n2n4m
pip install -e .
```
The bland pixel dataset from [Plebani et al. (2022)](https://zenodo.org/records/13338091) is required to ratio images. This can be downloaded from the link above, and should be placed in the data folder with the following structure:
| data/
| ----/CRISM_ML/
| -------- CRISM_bland_unratioed.mat
The following bash script will do this for you:
```bash
mkdir data
cd data
mkdir CRISM_ML
cd CRISM_ML
wget https://zenodo.org/records/13338091/files/CRISM_bland_unratioed.mat
cd ..
cd ..
```
A detailed explanation of how to use the package is given in the notebooks/[tutorials](https://github.com/rob-platt/N2N4M/tree/main/notebooks/tutorials) folder.

##### Retraining and Evaluation
To retrain an N2N4M model, the following steps are required:
* Download both the mineral and bland pixel datasets from [Plebani et al. (2022)](https://zenodo.org/records/13338091). These should be placed in the /data/CRISM_ML folder.
* Download the imagery used for the Plebani et al. (2022) datasets from [MarsSI](https://marssi.univ-lyon1.fr/wiki/Home). The _CAT_corr.img files must be used, but must be renamed to match the original .img filenames. These images should be placed in the /data/raw_mineral_images and /data/raw_bland_images folders respectively.
* Run the bland_dataset_collation.py and mineral_dataset_collation.py scripts in the /scripts folder. This will extract all relevant pixels from the raw images and save them as a single .json file. 
* Run the train.py script in the /scripts folder. This will train the N2N4M model and save the weights in the /data folder.

All of the above steps are also reqeired to run any notebook in notebooks/n2n4m_results. The notebooks in this folder are designed to evaluate the performance of the N2N4M model.

#### Tests
The package includes a test suite that can be run using the following command:
```bash
pytest
```
The tests are designed to be run in the root directory of the package.
All tests except test_io and test_postprocessing should pass. The test_io requires N2N4M/tests/test_io/3561F/ATU0003561F_01_IF168L_TRR3.img to exist. The test_postprocessing requires that file to exist, and the CRISM_ML bland pixel dataset to be in the data/CRISM_ML folder.

#### Licence
This package is released under the MIT licence.

#### Citation
If you use this package, please cite the following paper:
```
@misc{platt_noise2noise_2024,
	title = {{Noise2Noise} {Denoising} of {CRISM} {Hyperspectral} {Data}},
	url = {http://arxiv.org/abs/2403.17757},
	publisher = {arXiv},
	author = {Platt, Robert and Arcucci, Rossella and John, Cédric M.},
	month = mar,
	year = {2024},
	note = {arXiv:2403.17757 [cs]},
	keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Machine Learning},
}
```

## Acknowledgement
This code is part of Robert Platt's PhD work and you can [visit his GitHub repository](https://github.com/rob-platt) where the primary version of this code resides. The work was carried out under the supervision of [Cédric John](https://github.com/cedricmjohn) and all code from the research group can be found in the [John Lab GitHub repository](https://github.com/johnlab-research).

<a href="https://www.john-lab.org">
<img src="https://www.john-lab.org/wp-content/uploads/2023/01/footer_small_logo.png" style="width:220px">
</a>

#### References
1. Plebani E, Ehlmann BL, Leask EK, Fox VK, Dundar MM. A machine learning toolkit for CRISM image analysis. Icarus. 2022 Apr;376:114849.
2. Bultel B, Quantin C, Lozac'h L. Description of CoTCAT (Complement to CRISM Analysis Toolkit). IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing. 2015 Jun;8(6):3039-49.

