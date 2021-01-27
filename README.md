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
 - No data is deposited in this repository given the size of the simulation used. Please, if you wish to download the data, contact the authors.
 - The simulations contains the WCD information of roughly 3600 vertical proton showers (2000 train and 1600 test) whose primary energy is about 4 TeV, and is currently being used for train and test.
 - The simulation files should be saved in ~/data. Optionally the path can be adapted in the configs files.

Each hdf file contains three data sets accessible with the following "keys":
 - "Signals": Signal time traces of the PMTs in each WCD. 
 - "Info_stations": Information about each WCD with signal (>300 p.e.): Id shower, Id station, Number of particles, muons, energy, etc. 
 - "Info_shower": Information about each shower event: energy, primary, inclination, number of particles, etc. 
