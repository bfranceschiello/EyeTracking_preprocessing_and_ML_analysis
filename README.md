# "Machine learning algorithms on eye tracking trajectories to classify patients with spatial neglect"

<p float="middle">
  <img src="https://github.com/bfranceschiello/EyeTracking_preprocessing_and_ML_analysis/blob/main/3_Figures/eye_trajectory_task.png" />
</p>

This repository contains the code used for the [paper](https://www.medrxiv.org/content/medrxiv/early/2021/12/03/2020.07.02.20143941.full.pdf): "Machine learning algorithms on eye tracking trajectories to classify patients with spatial neglect". 

Please cite the paper if you are using either our dataset, preprocessing or model.

### Data
#### Preprocessing of trajectories
You can download the dataset used for the preprocessing script from [this link](https://doi.org/10.5281/zenodo.6424677). For a quick test of the preprocessing, we also uploaded 4 subjects (2 healthy and 2 with neglect) inside the [folder](https://github.com/bfranceschiello/EyeTracking_preprocessing_and_ML_analysis/tree/main/1_Preprocessing/Dataset). 
#### Machine Learning classification
If users are only interested in running the classification script, they can find the dataset inside the [ML_Analysis/dataset_preprocessed_trajectories](https://github.com/bfranceschiello/EyeTracking_preprocessing_and_ML_analysis/tree/main/2_ML_Analysis/dataset_preprocessed_trajectories).

## Usage

### 1) Preprocessing of trajectories

To run the preprocessing of the trajectories, users can run the following matlab script, located inside [this folder](https://github.com/bfranceschiello/EyeTracking_preprocessing_and_ML_analysis/tree/main/1_Preprocessing):
```matlab
main.m
```
This file loads tre trajectories from the [Dataset folder](https://github.com/bfranceschiello/EyeTracking_preprocessing_and_ML_analysis/tree/main/1_Preprocessing/Dataset) and calls the preprocessing function `preprocessing_one_subject.m` for every subject.

### 2) Machine Learning classification

To run the classification script, users can utilize the config file `config_eye_trajectories.json` located inside [this folder](https://github.com/bfranceschiello/EyeTracking_preprocessing_and_ML_analysis/tree/main/0_Config_files) and run:
```python
classification_eye_trajectories.py --config config_eye_trajectories.json
```
The user should ensure that the paths inside the config file are correct. More details regarding the input arguments of the config file can be found inside the description of the function `ml_analysis` inside the `classification_eye_trajectories.py` [script](https://github.com/bfranceschiello/EyeTracking_preprocessing_and_ML_analysis/blob/a4222d99f9584e782befc85ded5e2f4402c31d87/2_ML_Analysis/classification_eye_trajectories.py#L15)
