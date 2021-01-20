#Script that optimizes the weights when using an ensemble

import pandas as pd
import numpy as np
import os
import sys

from scipy.optimize import minimize
from scipy import optimize

import joblib
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error, log_loss
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, auc
import sklearn.metrics as metrics

import tensorflow as tf
from keras.models import load_model
import keras.backend.tensorflow_backend as tfback
import xgboost as xgb 
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('./.matplotlib/matplotlibrc.bin')

import yaml
import random


def _get_available_gpus():  
    if tfback._LOCAL_DEVICES is None:  
        devices = tf.config.list_logical_devices()  
        tfback._LOCAL_DEVICES = [x.name for x in devices]  
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


tfback._get_available_gpus = _get_available_gpus

#Set seed
seeds = [1,2,3,4,5]
Kfold = 0
random_st=seeds[Kfold] 
seed = random_st
np.random.seed(random_st)
random.seed(random_st)
tf.random.set_seed(random_st)

#Get validation dataset
s = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_st)


def NormalizeAllData(x):
    pmt = PMTNUMBER
    for j in range(x.shape[0]):
        x[j,:] = x[j,:]/np.sum(x[j,:])
    return x

#Calculate Integrals and asymmetry of a data set
def asymmetry(pmt_signal):
    #Find the PMT with the maximun signal 
    max_pmt = pmt_signal.index(max(pmt_signal))
    S1 = pmt_signal[max_pmt]
    #Find the opposite PMT
    if (max_pmt in [0,1]): 
        opposite_pmt = max_pmt + 2
    else:
        opposite_pmt = max_pmt - 2

    S4 = pmt_signal[opposite_pmt]
    #Compute the asymmetry:
    asym = (S1-S4)/(S1+S4)
    return asym

def integrals(signal):
    output = []
    pmts = []
    #Loop over events (rows)
    for j in range(len(signal)):
        
        #Loop over PMTs inside the tank
        for i in range(PMTNUMBER):
            #Save integral of each PMT
            #pmts.append(np.sum(signal[j,i*traceLength:(i+1)*traceLength]))
            pmts.append(np.sum(signal[j,i*traceLength:(i*traceLength+integralTraceLength)]))
        #Compute asymmetry of the event
        pmts.append(asymmetry(pmts))
        #Append total signal (photoelectrons):
        light_WCD = np.sum(pmts[0:PMTNUMBER])
        pmts.append(light_WCD)
        #Add percentage of signal
        for l in range(PMTNUMBER):
            pmts.append(pmts[l]/light_WCD)
        #Save integrals and asymmetry as a new row
        output.append(pmts)  
        pmts = [] 
    array = np.asarray(output)
    #output = pd.DataFrame(data=array, columns=['PMT1', 'PMT2', 'PMT3', 'PMT4', 'Asymmetry', 'Total_signal'])

    return array

#Read parameters
if (len(sys.argv) < 3):
    print("ERROR. No arguments were given to identify the run", file = sys.stderr)
    print("Please indicate the ID for the experiment and the yaml config file")
    sys.exit(1)

#Read configuration file
with open(sys.argv[2], 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

exp_ID = sys.argv[1]
print("Experiment ID: ", exp_ID)
print("Exp. Config.: ", ymlfile)


PMTNUMBER = cfg['global']['PMTNUMBER']
traceLength = cfg['global']['traceLength']
integralTraceLength = cfg['global']['integralTraceLength']

#Read Test data
TestDf = pd.read_hdf(os.path.join(cfg['global']['dataPath'],
                      cfg['experiment']['testfile']),key='Signals')

#Load signals and labels
X_test = TestDf.iloc[:,2:-1].values
Y_test = TestDf.iloc[:,-1].values

#Compute the engineered variables and then normalise the data
X_test_int = integrals(X_test)
X_test = NormalizeAllData(X_test)
X_test_rshp = X_test.reshape((X_test.shape[0], PMTNUMBER, traceLength))

#Load info stations to separate only in Single Muons
TestDf_info_stations = pd.read_hdf(os.path.join(cfg['global']['dataPath'],
                      cfg['experiment']['testfile']),key='Info_stations')

#Load single muons class: 0 (without muons), 1 (single muon), 2 (contaminated muon)
SM_class = TestDf_info_stations.iloc[:,-2].values

#Search for the index of stations without muons and with single muons
index_SM = np.where((SM_class==0) | (SM_class==1))[0]

#Prepare data set
X_test_SM = X_test[index_SM,:]
Y_test_SM = Y_test[index_SM]
X_test_SM_int = X_test_int[index_SM,:]

X_test_SM_rshp = X_test_SM.reshape((X_test_SM.shape[0], PMTNUMBER, traceLength))



#Load the data: validation data set
TrainDf = pd.read_hdf(os.path.join(cfg['global']['dataPath'],
                      cfg['experiment']['trainfile']),key='Signals')

Xtrain = TrainDf.iloc[:,2:-1].values
Ytrain = TrainDf.iloc[:,-1].values

#Get validation dataset
s = StratifiedShuffleSplit(n_splits=1, test_size=0.1)

for train_index, test_index in s.split(Xtrain,Ytrain):
    X_train, X_val = Xtrain[train_index], Xtrain[test_index]
    Y_train, Y_val = Ytrain[train_index], Ytrain[test_index]

#Validation
X_val_int = integrals(X_val)
X_val = NormalizeAllData(X_val)
X_val_rshp = X_val.reshape((X_val.shape[0], PMTNUMBER, traceLength))



#Balanced X_train
from imblearn.over_sampling import RandomOverSampler
ovs = RandomOverSampler(sampling_strategy = 0.5)
X_train_balanced, Y_train_balanced = ovs.fit_sample(X_train, Y_train)

X_train_balanced_int = integrals(X_train_balanced)
X_train_balanced = NormalizeAllData(X_train_balanced)
X_train_balanced_rshp = X_train_balanced.reshape((X_train_balanced.shape[0], PMTNUMBER, traceLength))


#X_train 
X_train_int = integrals(X_train)
X_train = NormalizeAllData(X_train)
X_train_rshp = X_train.reshape((X_train.shape[0], PMTNUMBER, traceLength))




print("Train: Muons ",len(Y_train[Y_train==1])," , E.M. ",len(Y_train[Y_train==0]))
print("Validation: Muons ",len(X_val[Y_val==1])," , E.M. ",len(Y_val[Y_val==0]))

#Load models and make predictions
predictions_train = []
predictions_train_balanced = []
predictions_validation = []


#CNN model:
exp_ID = "AllVars"
epochs = 200
model_name = exp_ID+'-noAsym'+'-'+str(epochs)+'epo_CV'+str(Kfold)
model_CNN_AllVars = load_model('models/CV/keras-CNN2_'+model_name+'.h5')

variables = [0,1,2,3,5,6,7,8,9]

print("Predicting test CNN...")
#Train
pred_m1_train = np.asarray(model_CNN_AllVars.predict([X_train_rshp,X_train_int[:,variables]]))
pred_m1_train = pred_m1_train.reshape((X_train.shape[0], 1))
predictions_train.append(pred_m1_train)

#Train balanced
pred_m1_train_balanced = np.asarray(model_CNN_AllVars.predict([X_train_balanced_rshp,X_train_balanced_int[:,variables]]))
pred_m1_train_balanced = pred_m1_train_balanced.reshape((X_train_balanced.shape[0], 1))
predictions_train_balanced.append(pred_m1_train_balanced)

#Validation
pred_m1_validation = np.asarray(model_CNN_AllVars.predict([X_val_rshp,X_val_int[:,variables]]))
pred_m1_validation = pred_m1_validation.reshape((X_val.shape[0], 1))
predictions_validation.append(pred_m1_validation)

#Test_SM
pred_m1_test_SM = np.asarray(model_CNN_AllVars.predict([X_test_SM_rshp,X_test_SM_int[:,variables]]))
pred_m1_test_SM = pred_m1_test_SM.reshape((X_test_SM.shape[0], 1))


#Test AllMuons
pred_m1_test = np.asarray(model_CNN_AllVars.predict([X_test_rshp,X_test_int[:,variables]]))
pred_m1_test = pred_m1_test.reshape((X_test.shape[0], 1))


num_round = 1000

model_name_XGB = exp_ID+'-'+str(num_round)+'rounds_CV'+str(Kfold)
filename = 'xgboost-'+model_name_XGB
#XGB model:
xgb_model =joblib.load('./models/CV/'+filename+'.pkl')
print("Predicting test XGBoost...")

#Train
dtest = xgb.DMatrix(X_train_int, label=Y_train)
pred_m2_train = np.asarray(xgb_model.predict(dtest))
pred_m2_train = pred_m2_train.reshape((X_train.shape[0], 1))
predictions_train.append(pred_m2_train)
del dtest

#Train balancedY_train_balanced
dtest = xgb.DMatrix(X_train_balanced_int, label=Y_train_balanced)
pred_m2_train_balanced = np.asarray(xgb_model.predict(dtest))
pred_m2_train_balanced = pred_m2_train_balanced.reshape((X_train_balanced.shape[0], 1))
predictions_train_balanced.append(pred_m2_train_balanced)
del dtest

#Validation
dtest = xgb.DMatrix(X_val_int, label=Y_val)
pred_m2_validation = np.asarray(xgb_model.predict(dtest))
pred_m2_validation = pred_m2_validation.reshape((X_val.shape[0], 1))
predictions_validation.append(pred_m2_validation)
del dtest


#Test_SM
dtest = xgb.DMatrix(X_test_SM_int, label=Y_test_SM)
pred_m2_test_SM = np.asarray(xgb_model.predict(dtest))
pred_m2_test_SM = pred_m2_test_SM.reshape((X_test_SM.shape[0], 1))
del dtest
#Test
dtest = xgb.DMatrix(X_test_int, label=Y_test)
pred_m2_test = np.asarray(xgb_model.predict(dtest))
pred_m2_test = pred_m2_test.reshape((X_test.shape[0], 1))



#Now we can optimize the weights 

#Define a Loss function
def mse_loss_func_train(weight):
    weights = [weight,1-weight]
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions_train):
            final_prediction += weight*prediction

    return mean_squared_error(Y_train, final_prediction)

def mse_loss_func_train_balance(weight):
    weights = [weight,1-weight]
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions_train_balanced):
            final_prediction += weight*prediction

    return mean_squared_error(Y_train_balanced, final_prediction)


def mse_loss_func_validation(weight):
    weights = [weight,1-weight]
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions_validation):
            final_prediction += weight*prediction

    return mean_squared_error(Y_val, final_prediction)

#Define the starting values to use in the algorithm
#starting_values = [1/len(predictions)]*len(predictions)
#starting_values = [0.7,0.3]
starting_values = 0.9
print(starting_values)

#As constrain the sum of the weights must be equal to 1
#cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
#Our weights are bound between 0 and 1
bounds = [(0,1)]


def RMSE_error(pred_m1,pred_m2,w,label):
    prediction = w * pred_m1 + (1-w) * pred_m2
    RMSE = np.round(mean_squared_error(prediction,label, squared=False),4)
    return RMSE



results = dict()
print("\n Optimum weight with train \n")
results['train'] = optimize.differential_evolution(mse_loss_func_train, bounds=bounds)
print(results['train'])

weight_train = results['train']['x'][0]

#weight_train = train_results[-1][1]

RMSE_train = []
RMSE_train.append(RMSE_error(pred_m1_train,pred_m2_train,weight_train,Y_train))
RMSE_train.append(RMSE_error(pred_m1_validation,pred_m2_validation,weight_train,  Y_val))
RMSE_train.append(RMSE_error(pred_m1_test_SM,pred_m2_test_SM,weight_train,Y_test_SM))
RMSE_train.append(RMSE_error(pred_m1_test,pred_m2_test,weight_train,Y_test))


print("\n Optimum weight with balanced train \n")
results['train_balanced'] = optimize.differential_evolution(mse_loss_func_train_balance, bounds=bounds)
print(results['train_balanced'])

weight_train_balanced = results['train_balanced']['x'][0]

#weight_train_balanced = train_balanced_results[-1][1]

RMSE_train_balanced = []
RMSE_train_balanced.append(RMSE_error(pred_m1_train,pred_m2_train,weight_train_balanced,Y_train))
RMSE_train_balanced.append(RMSE_error(pred_m1_validation,pred_m2_validation,weight_train_balanced,  Y_val))
RMSE_train_balanced.append(RMSE_error(pred_m1_test_SM,pred_m2_test_SM,weight_train_balanced,Y_test_SM))
RMSE_train_balanced.append(RMSE_error(pred_m1_test,pred_m2_test,weight_train_balanced,Y_test))





print("\n Optimum weight with validation \n")
results['validation'] = optimize.differential_evolution(mse_loss_func_validation, bounds=bounds)
print(results['validation'])

weight_val = results['validation']['x'][0]

#weight_val = val_results[-1][1]

RMSE_val = []
RMSE_val.append(RMSE_error(pred_m1_train,pred_m2_train,weight_val,Y_train))
RMSE_val.append(RMSE_error(pred_m1_validation,pred_m2_validation,weight_val,  Y_val))
RMSE_val.append(RMSE_error(pred_m1_test_SM,pred_m2_test_SM,weight_val,Y_test_SM))
RMSE_val.append(RMSE_error(pred_m1_test,pred_m2_test,weight_val,Y_test))



def RMSE_error_multiplicate(pred_m1,pred_m2,label):
    prediction = pred_m1 * pred_m2
    RMSE = np.round(mean_squared_error(prediction,label, squared=False),4)
    return RMSE

RMSE_mult = []
RMSE_mult.append(RMSE_error_multiplicate(pred_m1_train,pred_m2_train,Y_train))
RMSE_mult.append(RMSE_error_multiplicate(pred_m1_validation,pred_m2_validation,  Y_val))
RMSE_mult.append(RMSE_error_multiplicate(pred_m1_test_SM,pred_m2_test_SM,Y_test_SM))
RMSE_mult.append(RMSE_error_multiplicate(pred_m1_test,pred_m2_test,Y_test))



def RMSE_error_sqrt(pred_m1,pred_m2,label):
    prediction = np.sqrt(pred_m1 * pred_m2)
    RMSE = np.round(mean_squared_error(prediction,label, squared=False),4)
    return RMSE

RMSE_mult_sqrt = []
RMSE_mult_sqrt.append(RMSE_error_sqrt(pred_m1_train,pred_m2_train,Y_train))
RMSE_mult_sqrt.append(RMSE_error_sqrt(pred_m1_validation,pred_m2_validation,  Y_val))
RMSE_mult_sqrt.append(RMSE_error_sqrt(pred_m1_test_SM,pred_m2_test_SM,Y_test_SM))
RMSE_mult_sqrt.append(RMSE_error_sqrt(pred_m1_test,pred_m2_test,Y_test))

def RMSE_circle(pred_m1,pred_m2,label):
    prediction = 1/np.sqrt(2) * np.sqrt(pred_m1**2 + pred_m2**2)
    RMSE = np.round(mean_squared_error(prediction,label, squared=False),4)
    return RMSE

RMSE_circle_func = []
RMSE_circle_func.append(RMSE_circle(pred_m1_train,pred_m2_train,Y_train))
RMSE_circle_func.append(RMSE_circle(pred_m1_validation,pred_m2_validation,  Y_val))
RMSE_circle_func.append(RMSE_circle(pred_m1_test_SM,pred_m2_test_SM,Y_test_SM))
RMSE_circle_func.append(RMSE_circle(pred_m1_test,pred_m2_test,Y_test))






table_name = model_name+"_"+filename+"_ensemble"
table_results_output = "./tables/paper/ensembles/"+table_name+".tex"
a = open(table_results_output,'w')
a.write("\n Optimised   &  Weight   &  Train   &  Validation   &  Test (SM)   &  Test (AM) \n")
a.write("\n Train   & "+ str(np.round(weight_train,4)) +"  & "+ str(RMSE_train[0]) +"  & "+ str( RMSE_train[1]  ) +"  & "+ str( RMSE_train[2]  )+"  & "+ str( RMSE_train[3]  )+"\\\\ \hline")
a.write("\n Balanced Train   & "+ str(np.round(weight_train_balanced,4)) +"  & "+ str( RMSE_train_balanced[0]  ) +"  & "+ str( RMSE_train_balanced[1]  ) +"  & "+ str( RMSE_train_balanced[2]  )+"  & "+ str( RMSE_train_balanced[3]  )+"\\\\ \hline")
a.write("\n Validation   & "+ str(np.round(weight_val,4)) +"  & "+ str( RMSE_val[0]  ) +"  & "+ str( RMSE_val[1]  ) +"  & "+ str( RMSE_val[2]  )+"  & "+ str( RMSE_val[3]  )+"\\\\ \hline")
a.write("\n None   &  $P_{m1} \cdot P_{m2}$   & "+ str( RMSE_mult[0]  ) +"  & "+ str( RMSE_mult[1]  ) +"  & "+ str( RMSE_mult[2]  )+"  & "+ str( RMSE_mult[3]  )+"\\\\ \hline")
a.write("\n None   &  $\sqrt{P_{m1} \cdot P_{m2}}$   & "+ str( RMSE_mult_sqrt[0]  ) +"  & "+ str( RMSE_mult_sqrt[1]  ) +"  & "+ str( RMSE_mult_sqrt[2]  )+"  & "+ str( RMSE_mult_sqrt[3]  )+"\\\\ \hline")
a.write("\n None   &  $\\frac{1}{2} \sqrt{P_{m1}^2 + P_{m2}^2}$   & "+ str( RMSE_circle_func[0]  ) +"  & "+ str( RMSE_circle_func[1]  ) +"  & "+ str( RMSE_circle_func[2]  )+"  & "+ str( RMSE_circle_func[3]  )+"\\\\ \hline")
a.close()    



def compute_tables2(pred_model1,pred_model2,Y,table_name):

    prediction = pred_model1*pred_model2

    acc = []
    #bal_acc = []
    prec = []
    recall = []
    f1s = []
    Tmu = []
    Tpho = []


    thresholdL = np.arange(0.01,1,0.01)

    for threshold in thresholdL:

        pred = prediction > threshold
            
        pred = np.array(pred).astype(int)
            
        acc.append(accuracy_score(Y,pred)*100)
        #bal_acc.append()
        prec.append(precision_score(Y,pred)*100)
        recall.append(recall_score(Y,pred)*100)
        f1s.append(f1_score(Y,pred)*100)
        Tmu.append(100*np.sum((pred[Y==1] == 1))/np.sum(Y==1))
        Tpho.append(100*np.sum((pred[Y==0] == 0))/np.sum(Y==0))


    #Dump results to latex table:

    #Results:
    d = {'Threshold': thresholdL, 'Accuracy':acc, 'F1 score': f1s, 'Precision': prec, 'Recall': recall, '$T_{\mu}$': Tmu  , '$T_{\gamma}$': Tpho}

    table_results_output = "./tables/paper/ensembles/"+table_name+".tex"

    df_results = pd.DataFrame(data=d, columns=['Threshold', 'Accuracy','F1 score', 'Precision', 'Recall', '$T_{\mu}$', '$T_{\gamma}$'])
    #Write data to latex file
    results_table = df_results.to_latex(index = False, header = True, float_format="%.2f",
                        column_format = "|c|c|c|c|c|c|c|").replace('\\toprule', '\hline').replace('\midrule', '\hline').replace('\bottomrule','\hline')
                        #caption = "Results for experiment "+ exp_ID+"-"+str(epochs)+"epo",
                        #label = exp_ID)




    #Summary of results:
    rows = ["Mean","Min.", "Max."]
    acc_s = [np.mean(acc), np.min(acc), np.max(acc)]
    f1s_s = [np.mean(f1s), np.min(f1s), np.max(f1s)]
    prec_s = [np.mean(prec), np.min(prec), np.max(prec)]
    recall_s = [np.mean(recall), np.min(recall), np.max(recall)]
    Tpho_s = [np.mean(Tpho), np.min(Tpho), np.max(Tpho)]
    Tmu_s = [np.mean(Tmu), np.min(Tmu), np.max(Tmu)]

    d_s = {'Results': rows, 'Accuracy':acc_s, 'F1 score': f1s_s, 'Precision': prec_s, 'Recall': recall_s, '$T_{\mu}$': Tmu_s  , '$T_{\gamma}$': Tpho_s}

    table_results_summary_output = "./tables/table-summary-"+table_name+".tex"

    df_results_s = pd.DataFrame(data=d_s, columns=['Results', 'Accuracy', 'F1 score', 'Precision', 'Recall', '$T_{\mu}$', '$T_{\gamma}$'])
    df_results_s.set_index('Results', inplace =  True)
    #Write data to latex file
    summary_table = df_results_s.to_latex(index = True, header = True, float_format="%.2f",
                        column_format = "|c|c|c|c|c|c|c|").replace('\toprule', '\hline').replace('\midrule', '\hline').replace('\bottomrule','\hline')
                        #caption = "Summary of results for experiment " + exp_ID+"-"+str(epochs)+"epo",
                        #label = "sum"+exp_ID)

    a = open(table_results_output,'w')
    a.write(results_table)
    a.write(summary_table)
    a.write("\n Dev. std. accuracy = "+str(np.std(acc))+"\n Dev. std. F1-score = "+str(np.std(f1s))+
        "\n $R^2$ = "+str(r2_score(Y, prediction))+"\n MSE = "+str(mean_squared_error(Y, prediction))+
        "\n RMSE = "+str(mean_squared_error(Y, prediction, squared=False))+"\n MAE = "+str(mean_absolute_error(Y, prediction)))
    a.close()




RMSE_mult = []
RMSE_mult.append(RMSE_error_multiplicate(pred_m1_train,pred_m2_train,Y_train))
RMSE_mult.append(RMSE_error_multiplicate(pred_m1_validation,pred_m2_validation,  Y_val))
RMSE_mult.append(RMSE_error_multiplicate(pred_m1_test_SM,pred_m2_test_SM,Y_test_SM))
RMSE_mult.append(RMSE_error_multiplicate(pred_m1_test,pred_m2_test,Y_test))



table_name = model_name+"_"+filename+"_ensemble_thresholds_train_Original"
compute_tables2(pred_m1_train,pred_m2_train,Y_train,table_name)

table_name = model_name+"_"+filename+"_ensemble_thresholds_train"
compute_tables2(pred_m1_train_balanced,pred_m2_train_balanced,Y_train_balanced,table_name)

table_name = model_name+"_"+filename+"_ensemble_thresholds_val"
compute_tables2(pred_m1_validation,pred_m2_validation,  Y_val,table_name)

table_name = model_name+"_"+filename+"_ensemble_thresholds_test_SM"
compute_tables2(pred_m1_test_SM,pred_m2_test_SM,Y_test_SM,table_name)

table_name = model_name+"_"+filename+"_ensemble_thresholds_test_AM"
compute_tables2(pred_m1_test,pred_m2_test,Y_test,table_name)



#Plot MSE
MSE = []
w = []

for i in np.arange(0,1.01,0.01):
    w.append(i)
    MSE.append(mse_loss_func(i))




#Plot with all the efficiencies: paper
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111) 
plt.plot(w, MSE,'go-',color='black',markersize=2.5, linewidth=1.5)
plt.autoscale(enable=True)
plt.xlabel("Weight")
plt.ylabel('MSE')
#ax.legend(loc = 'best', edgecolor="black",fontsize="small")
fig.tight_layout()
filename = './models/figs/papers_plots/astro/weight_MSE.pdf'
plt.savefig(filename)
plt.close()

#Single Muons

#CNN
fpr_CNN, tpr_CNN, threshold_CNN = metrics.roc_curve(Y_test_SM, pred_m1_test_SM)
roc_auc_CNN = metrics.auc(fpr_CNN, tpr_CNN)

#XGboost
fpr_XGB, tpr_XGB, threshold_XGB = metrics.roc_curve(Y_test_SM, pred_m2_test_SM)
roc_auc_XGB = metrics.auc(fpr_XGB, tpr_XGB)


#Ensemble
fpr_ensemble, tpr_ensemble, threshold_ensemble = metrics.roc_curve(Y_test_SM, pred_m1_test_SM*pred_m2_test_SM)
roc_auc_ensemble = metrics.auc(fpr_ensemble, tpr_ensemble)



plt.figure(figsize=(8,7))
#plt.title("ROC curve for test data set \n"+exp_ID)
plt.plot(fpr_CNN, tpr_CNN, 'r', label = 'CNN, AUC = %0.2f' % roc_auc_CNN)
plt.plot(fpr_XGB, tpr_XGB, 'b', label = 'XGB, AUC = %0.2f' % roc_auc_XGB)
plt.plot(fpr_ensemble, tpr_ensemble, 'g', label = 'CNN $\cdot$ XGB, AUC = %0.2f' % roc_auc_ensemble)
plt.legend(loc = 'lower right', edgecolor="black")
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('./models/figs/papers_plots/NeuralComp//ROC-SingleMuons.pdf')
plt.close()


#All Muons

#CNN
fpr_CNN, tpr_CNN, threshold_CNN = metrics.roc_curve(Y_test, pred_m1_test)
roc_auc_CNN = metrics.auc(fpr_CNN, tpr_CNN)

#XGboost
fpr_XGB, tpr_XGB, threshold_XGB = metrics.roc_curve(Y_test, pred_m2_test)
roc_auc_XGB = metrics.auc(fpr_XGB, tpr_XGB)


#Ensemble
fpr_ensemble, tpr_ensemble, threshold_ensemble = metrics.roc_curve(Y_test, pred_m1_test*pred_m2_test)
roc_auc_ensemble = metrics.auc(fpr_ensemble, tpr_ensemble)



plt.figure(figsize=(8,7))
#plt.title("ROC curve for test data set \n"+exp_ID)
plt.plot(fpr_CNN, tpr_CNN, 'r', label = 'CNN, AUC = %0.2f' % roc_auc_CNN)
plt.plot(fpr_XGB, tpr_XGB, 'b', label = 'XGBoost, AUC = %0.2f' % roc_auc_XGB)
plt.plot(fpr_ensemble, tpr_ensemble, 'g', label = 'CNN $\cdot$ XGB, AUC = %0.2f' % roc_auc_ensemble)
plt.legend(loc = 'lower right', edgecolor="black")
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.savefig('./models/figs/papers_plots/NeuralComp/ROC-AllMuons.pdf')
plt.close()



#Single Muons

#CNN
fpr_CNN, tpr_CNN, threshold_CNN = metrics.roc_curve(Y_test_SM, pred_m1_test_SM)
roc_auc_CNN = metrics.auc(fpr_CNN, tpr_CNN)

#XGboost
fpr_XGB, tpr_XGB, threshold_XGB = metrics.roc_curve(Y_test_SM, pred_m2_test_SM)
roc_auc_XGB = metrics.auc(fpr_XGB, tpr_XGB)


#Ensembles
#$0.2323 \cdot P_{\mu, \mathrm{CNN}} + 0.7677 \cdot P_{\mu, \mathrm{XGB}}$
fpr_ensemble_1, tpr_ensemble_1, threshold_ensemble_1 = metrics.roc_curve(Y_test_SM, 0.2323*pred_m1_test_SM+0.7677*pred_m2_test_SM)
roc_auc_ensemble_1 = metrics.auc(fpr_ensemble_1, tpr_ensemble_1)

#$0.1833 \cdot P_{\mu, \mathrm{CNN}} + 0.8167 \cdot P_{\mu, \mathrm{XGB}}$
fpr_ensemble_2, tpr_ensemble_2, threshold_ensemble_2 = metrics.roc_curve(Y_test_SM, 0.1833*pred_m1_test_SM+0.8167*pred_m2_test_SM)
roc_auc_ensemble_2 = metrics.auc(fpr_ensemble_2, tpr_ensemble_2)

#Product
fpr_ensemble_3, tpr_ensemble_3, threshold_ensemble_3 = metrics.roc_curve(Y_test_SM, pred_m1_test_SM*pred_m2_test_SM)
roc_auc_ensemble_3 = metrics.auc(fpr_ensemble_3, tpr_ensemble_3)

#sqrt Product
fpr_ensemble_4, tpr_ensemble_4, threshold_ensemble_4 = metrics.roc_curve(Y_test_SM, np.sqrt(pred_m1_test_SM*pred_m2_test_SM))
roc_auc_ensemble_4 = metrics.auc(fpr_ensemble_4, tpr_ensemble_4)

#$\frac{1}{\sqrt{2}} \cdot \sqrt{P_{\mu,m_1}^2 + P_{\mu,m_2}^2}$
fpr_ensemble_5, tpr_ensemble_5, threshold_ensemble_5 = metrics.roc_curve(Y_test_SM,  (1/np.sqrt(2))*np.sqrt(pred_m1_test_SM**2 + pred_m2_test_SM**2))
roc_auc_ensemble_5 = metrics.auc(fpr_ensemble_5, tpr_ensemble_5)


fig = plt.figure(figsize=(9,8))
ax = fig.add_subplot(111) 
#plt.title("ROC curve for test data set \n"+exp_ID)
plt.plot(fpr_CNN, tpr_CNN, 'r', label = 'CNN, AUC = %0.3f' % roc_auc_CNN)
plt.plot(fpr_XGB, tpr_XGB, 'b', label = 'XGB, AUC = %0.3f' % roc_auc_XGB)
#plt.plot(fpr_ensemble, tpr_ensemble, 'g', label = 'CNN $\cdot$ XGB, AUC = %0.2f' % roc_auc_ensemble)
ax.legend(loc = 'lower right', edgecolor="black",fontsize=22)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])￼

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('./models/figs/papers_plots/NeuralComp/ROC-SingleMuons_CNN-XGB.pdf')
plt.close()


fig = plt.figure(figsize=(9,8))
ax = fig.add_subplot(111) 
#plt.title("ROC curve for test data set \n"+exp_ID)
plt.plot(fpr_ensemble_1, tpr_ensemble_1, 'black', label = '$0.2323 \cdot P_{\mu, \mathrm{CNN}} + 0.7677 \cdot P_{\mu, \mathrm{XGB}}$, AUC = %0.3f' % roc_auc_ensemble_1)
plt.plot(fpr_ensemble_2, tpr_ensemble_2, 'aqua', label = '$0.1833 \cdot P_{\mu, \mathrm{CNN}} + 0.8167 \cdot P_{\mu, \mathrm{XGB}}$, AUC = %0.3f' % roc_auc_ensemble_2)
plt.plot(fpr_ensemble_4, tpr_ensemble_4, 'g', label = '$\sqrt{P_{\mu, \mathrm{CNN}} \cdot P_{\mu, \mathrm{XGB}}}$, AUC = %0.3f' % roc_auc_ensemble_4)
plt.plot(fpr_ensemble_5, tpr_ensemble_5, 'orange', label = '$\\frac{1}{\sqrt{2}} \cdot \sqrt{P_{\mu,CNN}^2 + P_{\mu,XGB}^2}$, AUC = %0.3f' % roc_auc_ensemble_5)
plt.plot(fpr_ensemble_3, tpr_ensemble_3, 'magenta', label = '$P_{\mu, \mathrm{CNN}} \cdot P_{\mu, \mathrm{XGB}}$, AUC = %0.3f' % roc_auc_ensemble_3)


#plt.plot(fpr_ensemble, tpr_ensemble, 'g', label = 'CNN $\cdot$ XGB, AUC = %0.2f' % roc_auc_ensemble)
ax.legend(loc = 'lower right', edgecolor="black",fontsize=14)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('./models/figs/papers_plots/NeuralComp/ROC-SingleMuons_ensembles.pdf')
plt.close()