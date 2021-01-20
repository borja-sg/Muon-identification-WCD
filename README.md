# Muon-identification-WCD
This repository contains the scripts necessary to obtain the muon probability from WCDs' signals.

## Scripts and files description

The following files were used during the experimentation:
 - ExportData_Stations.py -> Script used to export the ROOT simulation file into an hdf. 
 - CNN_train_CV.py -> Script to train the CNN using Cross Validation. It is possible to indicate which variables to select and the preprocessing that is going to be used for the experiment. The results of the experiment will be automatically saved in latex tables.
 - XGB-RF_train_CV.py -> Script to train the XGBoost or RandomForest using Cross Validation. The results of the experiment will be automatically saved in latex tables.
 - ensemble_optimization.py -> Script to combine the CNN models with the XGBoost and produce tables with the results of each ensemble. 
 - plot_best_CNN_model.py -> Script to load the best CNN and plot its structure. 
 - ./models -> Contain the saved models and the figures with the results. 
 - ./tables/CV -> This fold contain the tables in tex format with the results for each model trained using CV. The summary fold contain the mean for each model using all the seeds.
 - ./tables/ensembles -> This fold contain the tables with the results for the ensembles.

## Simulations
 - The simulations should be downloaded from [here for training](https://drive.google.com/file/d/1LSzuLodCDLr8AaZRpyrOxSi8-Vmwt3W1/view?usp=sharing) and [here for test](https://drive.google.com/file/d/1YXuLaRWpNJ2i7dnPO5IG1irQ0cpcpbXU/view?usp=sharing), into the input_files directory.
 - This simulations contains the WCD information of roughly 3600 vertical proton showers (2000 train and 1600 test) whose primary energy is about 4 TeV, and is currently being used for train and test (each file with ~500 Mb).
 - The simulation files should be saved in ~/data. Optionally the path can be adapted in the configs files.



