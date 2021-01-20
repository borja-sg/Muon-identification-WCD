#This script trains a CNN using a cross validation with 5 folds
#Inputs: Exp_id, config_file

import pandas as pd
import numpy as np
import os
import sys

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.preprocessing import StandardScaler 
import copy 


import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, Reshape, concatenate, Input
from keras.optimizers import SGD,Adam
from keras.wrappers.scikit_learn import KerasRegressor
from keras.backend import one_hot
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import load_model

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


import joblib

import yaml
import random



def _get_available_gpus():  

    if tfback._LOCAL_DEVICES is None:  
        devices = tf.config.list_logical_devices()  
        tfback._LOCAL_DEVICES = [x.name for x in devices]  
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


tfback._get_available_gpus = _get_available_gpus

if (len(sys.argv) < 3):
    print("ERROR. No arguments were given to identify the run", file = sys.stderr)
    print("Please indicate the ID for the experiment and the yaml config file")
    sys.exit(1)


with open(sys.argv[2], 'r') as ymlfile:   
    cfg = yaml.safe_load(ymlfile)


exp_ID = sys.argv[1]
print("Experiment ID: ", exp_ID)
print("Exp. Config.: ", cfg)

"""
LIST OF EXPERIMENTS:
CNN1                        CNN1: Signals
CNN2_AllVars                CNN2: Signals+Integrals+Asymmetry+WCD+NormIntegrals
CNN2_AllVars_noAsym         CNN2: Signals+Integrals+WCD+NormIntegrals
CNN2_AllVars_Int            CNN2: Signals+Integrals+WCD+Asymmetry
CNN2_AllVars_NormInt        CNN2: Signals+NormIntegrals+WCD+Asymmetry
CNN2_Asymmetry              CNN2: Signals+Asymmetry
CNN2_WCD                    CNN2: Signals+WCD
CNN2_Int                    CNN2: Signals+Integrals
CNN2_NormInt                CNN2: Signals+NormIntegrals
CNN2_NormInt_WCD            CNN2: Signals+Integrals + WCD
CNN2_Int_WCD                CNN2: Signals+NormIntegrals + WCD
"""

#Input exp_ID to select the experiment:

integrate = "yes"

if (exp_ID == "CNN1"):
    strategy = "NormalisePMTs"
    integrate = "no"
    variables = []
elif (exp_ID == "CNN2_AllVars"):
    strategy = "AllVars"
    variables = [0,1,2,3,4,5,6,7,8,9]
elif (exp_ID == "CNN2_AllVars-noAsym"):
    strategy = "AllVars"
    variables = [0,1,2,3,5,6,7,8,9]
elif (exp_ID == "CNN2_AllVars-noAsym_inclined"):
    strategy = "AllVars"
    variables = [0,1,2,3,5,6,7,8,9]
elif (exp_ID == "CNN2_AllVars_Int"):
    strategy = "Integrals"
    variables = [0,1,2,3,4,5]
elif (exp_ID == "CNN2_AllVars_NormInt"):
    strategy = "NormalisePMTs"
    variables = [0,1,2,3,4,5]
elif (exp_ID == "CNN2_Asymmetry"):
    strategy = "NormalisePMTs"
    variables = [4]
elif (exp_ID == "CNN2_WCD"):
    strategy = "NormalisePMTs"
    variables = [5]
elif (exp_ID == "CNN2_Int"):
    strategy = "Integrals"
    variables = [0,1,2,3]
elif (exp_ID == "CNN2_NormInt"):
    strategy = "NormalisePMTs"
    variables = [0,1,2,3]
elif (exp_ID == "CNN2_NormInt_WCD"):
    strategy = "NormalisePMTs"
    variables = [0,1,2,3,5]
elif (exp_ID == "CNN2_Int_WCD"):
    strategy = "Integrals"
    variables = [0,1,2,3,5]






#Load congig paramenters from config file
PMTNUMBER = cfg['global']['PMTNUMBER']
traceLength = cfg['global']['traceLength']
integralTraceLength = cfg['global']['integralTraceLength']



#Normalisation 
def NormalizeAllData(x):
    pmt = PMTNUMBER
    for j in range(x.shape[0]):
        x[j,:] = x[j,:]/np.sum(x[j,:])
    return x

#Calculate asymmetry of a tank
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


#Compute engineered variables:
def integrals(signal):
    output = []
    pmts = []
    #Loop over events (rows)
    for j in range(len(signal)):        
        #Loop over PMTs inside the tank
        for i in range(PMTNUMBER):
            #Save integral of each PMT
            pmts.append(np.sum(signal[j,i*traceLength:(i*traceLength+integralTraceLength)]))
        #Compute asymmetry of the event
        pmts.append(asymmetry(pmts))
        #Append total signal (photoelectrons):
        pmts.append(np.sum(pmts[0:PMTNUMBER]))
        #Change integrals for percentage of signal
        if (strategy=="NormalisePMTs"): 
            pmts[0:PMTNUMBER] = pmts[0:PMTNUMBER]/np.sum(pmts[0:PMTNUMBER])
        elif (strategy=="AllVars"): #Add percetage in adition to integrals
            light_WCD = np.sum(pmts[0:PMTNUMBER])
            for l in range(PMTNUMBER):
                pmts.append(pmts[l]/light_WCD)               
        #Save integrals and asymmetry as a new row
        pmts = np.asarray(pmts)
        output.append(pmts[variables].tolist())  
        pmts = [] 
    array = np.asarray(output)
    #output = pd.DataFrame(data=array, columns=['PMT1', 'PMT2', 'PMT3', 'PMT4', 'Asymmetry', 'Total_signal'])

    return array


#Function to create the tables for a given trained model and data set
def compute_tables(X,X_int,Y,model,table_name):
    X_rshp = X.reshape((X.shape[0], PMTNUMBER, traceLength))
    if (integrate=="yes"):
        #prediction = model.predict([X_rshp,X_int])
        prediction = np.asarray(model.predict([X_rshp,X_int]))
        prediction = prediction.reshape((X.shape[0], 1))
    else:
        #prediction = model.predict(X_rshp)
        prediction = np.asarray(model.predict(X_rshp))
        prediction = prediction.reshape((X.shape[0], 1))

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

    table_results_output = "./tables/paper/CV/"+table_name+".tex"

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

    R2 = r2_score(Y, prediction)
    MSE = mean_squared_error(Y, prediction)
    RMSE = mean_squared_error(Y, prediction, squared=False)
    MAE = mean_absolute_error(Y, prediction)

    resultados = [R2,MSE,RMSE,MAE]

    a = open(table_results_output,'w')
    a.write(results_table)
    a.write(summary_table)
    a.write("\n Dev. std. accuracy = "+str(np.std(acc))+"\n Dev. std. F1-score = "+str(np.std(f1s))+
                "\n $R^2$ = "+str(R2)+"\n MSE = "+str(MSE)+"\n RMSE = "+str(RMSE)+"\n MAE = "+str(MAE))
    a.close()    

    return resultados

#Create a summary table of all the data sets for the ML paper 
def summary_tables(datas,table_name):

    table_results_output = "./tables/paper/CV/summary/"+table_name+".tex"
    a = open(table_results_output,'w')

    r2 = []
    r2_std = []
    MSE = []
    MSE_std = []
    RMSE = []
    RMSE_std = []
    MAE = []
    MAE_std = []


    tests = ["\n TRAIN \n", "\n TRAIN ORIGINAL \n","\n VALIDATION \n","\n TEST Single Muons\n","\n TEST All Muons \n"]
    j=0
    for data in datas:
        metric = []
        metric_std = []
        for i in range(data.shape[1]):
            metric.append(np.round(np.mean(data[:,i]),4)) 
            metric_std.append(np.round(np.std(data[:,i]),4))
            if (i == 0):
                r2.append(np.round(np.mean(data[:,i]),4))
                r2_std.append(np.round(np.std(data[:,i]),4))
            elif (i==1):
                MSE.append(np.round(np.mean(data[:,i]),4))
                MSE_std.append(np.round(np.std(data[:,i]),4))
            elif (i==2):
                RMSE.append(np.round(np.mean(data[:,i]),4))
                RMSE_std.append(np.round(np.std(data[:,i]),4))
            elif (i==3):
                MAE.append(np.round(np.mean(data[:,i]),4))
                MAE_std.append(np.round(np.std(data[:,i]),4))


        a.write(tests[j])
        a.write("$R^2$ = "+str(metric[0])+" ("+str(metric_std[0])+")\n MSE = "+str(metric[1])+" ("+str(metric_std[1])+
        ")\n RMSE = "+str(metric[2])+" ("+str(metric_std[2])+")\n MAE = "+str(metric[3])+" ("+str(metric_std[3])+")")
        a.write("\n & "+str(metric[0])+" ("+str(metric_std[0])+") & "+str(metric[1])+" ("+str(metric_std[1])+
        ") & "+str(metric[2])+" ("+str(metric_std[2])+") & "+str(metric[3])+" ("+str(metric_std[3])+") \hline")
        
        j += 1


    a.write("\n TRAIN VALIDATION TEST_SM TEST_AM \n")
    a.write("\n R2 \n")
    #R^2
    a.write("\n & "+str(r2[0])+" ("+str(r2_std[0])+") & "+str(r2[1])+" ("+str(r2_std[1])+
        ") & "+str(r2[2])+" ("+str(r2_std[2])+") & "+str(r2[3])+" ("+str(r2_std[3])+") \\ \hline")

    #MSE
    a.write("\n MSE \n")
    a.write("\n & "+str(MSE[0])+" ("+str(MSE_std[0])+") & "+str(MSE[1])+" ("+str(MSE_std[1])+
        ") & "+str(MSE[2])+" ("+str(MSE_std[2])+") & "+str(MSE[3])+" ("+str(MSE_std[3])+") \\ \hline")   

    #RMSE
    a.write("\n RMSE \n")
    a.write("\n & "+str(RMSE[0])+" ("+str(RMSE_std[0])+") & "+str(RMSE[1])+" ("+str(RMSE_std[1])+
        ") & "+str(RMSE[2])+" ("+str(RMSE_std[2])+") & "+str(RMSE[3])+" ("+str(RMSE_std[3])+") \\ \hline") 

    #MAE
    a.write("\n MAE \n")
    a.write("\n & "+str(MAE[0])+" ("+str(MAE_std[0])+") & "+str(MAE[1])+" ("+str(MAE_std[1])+
        ") & "+str(MAE[2])+" ("+str(MAE_std[2])+") & "+str(MAE[3])+" ("+str(MAE_std[3])+") \\ \hline")   

    a.close()    





#Read Train data
TrainDf = pd.read_hdf(os.path.join(cfg['global']['dataPath'],
                      cfg['experiment']['trainfile']), key="Signals")

Xtrain = np.asarray(TrainDf.iloc[:,2:-1].values,dtype="float64")
Ytrain = TrainDf.iloc[:,-1].values

print("Muon proportion in train:")
print(np.sum(Ytrain == 1) /(np.sum(Ytrain == 0)+np.sum(Ytrain == 1)))  
print("Stations with muons in train:")
print(np.sum(Ytrain == 1))
print("Stations without muons in train:")
print(np.sum(Ytrain == 0))  


#Read Test data
TestDf = pd.read_hdf(os.path.join(cfg['global']['dataPath'],
                      cfg['experiment']['testfile']),key='Signals')

#Load signals and labels
X_test = np.asarray(TestDf.iloc[:,2:-1].values,dtype="float64")
Y_test = TestDf.iloc[:,-1].values

#Compute the engineered variables and then normalise the data
X_test_int = integrals(X_test)
X_test = NormalizeAllData(X_test)


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


#Prepare list to save the data
resultados_train = []
resultados_train_original = []
resultados_val = []
resultados_test = []
resultados_testSM = []

#Number of folds
Kfolds = 5

#Seed to use in each of the folds
seeds = [1,2,3,4,5]

for Kfold in range(Kfolds):

    print("\n Computing results for Kfold "+str(Kfold+1)+"/"+str(Kfolds)+"\n")
    #Establish the seed
    random_st=seeds[Kfold] 
    seed = random_st
    np.random.seed(random_st)
    random.seed(random_st)
    tf.random.set_seed(random_st)

    #Get validation dataset for the given kfold
    s = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_st)

    for train_index, test_index in s.split(Xtrain,Ytrain):
        X_train, X_val = Xtrain[train_index], Xtrain[test_index]
        Y_train, Y_val = Ytrain[train_index], Ytrain[test_index]

    X_train_original = copy.deepcopy(X_train)
    Y_train_original = copy.deepcopy(Y_train)
    X_train_original_int = integrals(X_train_original)
    X_train_original = NormalizeAllData(X_train_original)


    print("Proporcion de muones en train:")
    print(np.sum(Y_train == 1) /np.sum(Y_train == 0))  
    print("Proporcion de muones en validation:")
    print(np.sum(Y_val == 1) /np.sum(Y_val == 0)) 

    #########
    #Imbalanced...
    #########

    undersample = np.array([])
    oversample = np.array([])
    muonrows = np.array([])
    shf = np.array([])

    sizeClass0 = np.sum(Y_train == 0)
    sizeClass1 = np.sum(Y_train == 1)

    if (cfg['FE']['UD'] == 'smote' ):
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(sampling_strategy = cfg['FE']['UDfactor'])
        X_train, Y_train = sm.fit_sample(X_train, Y_train)
        
        del sm

    if (cfg['FE']['UD'] == 'Rand' ):   

        from imblearn.under_sampling import RandomUnderSampler
        UDfactor = cfg['FE']['UDfactor']
        prop_und = 1/UDfactor
        und = RandomUnderSampler(sampling_strategy = prop_und)
        X_train, Y_train = und.fit_sample(X_train, Y_train)
        
        del und 


    if (cfg['FE']['OS'] == 'Yes'):   

        from imblearn.over_sampling import RandomOverSampler
        ovs = RandomOverSampler(sampling_strategy = 0.5)
        X_train, Y_train = ovs.fit_sample(X_train, Y_train)
        
        del ovs
        


    #shuffle the data before training...
    shf = np.concatenate((X_train,Y_train.reshape(-1,1)), axis = 1)
    np.random.shuffle(shf)
    X_train = shf[:,0:-1]
    Y_train = shf[:,-1]

    del undersample
    del oversample
    del muonrows
    del shf



    #Calculating integrals and asymmetry for train and validation. After that, normalise the data
    #Train:
    X_train_int = integrals(X_train)
    X_train = NormalizeAllData(X_train)



    #Test
    X_val_int = integrals(X_val)
    X_val = NormalizeAllData(X_val)

    #############
    #Build models 
    #############

    #4 channels + engineered vars
    def build_model_4ch_int():
        #Convolutions with signals
        signals = Input(shape=(PMTNUMBER, traceLength), name='Signal_traces')
        conv1 = Conv1D(kernel_size = [2], filters = 20, strides = 2, activation='relu',data_format="channels_first")(signals)
        conv2 = Conv1D(kernel_size = [2], filters = 15, activation='relu')(conv1)
        conv3 = Conv1D(kernel_size = [2], filters = 10, activation='relu')(conv2)
        res_conv = Flatten()(conv3)

        #Now we can add the integrals and asymmetry to the dense layer
        shape = 1

        if (len(X_train_int.shape)>1):
            shape = X_train_int.shape[1]
        
        integral = Input(shape=(shape,), name='Signal_integrals')
        
        
        concat = concatenate([res_conv, integral], name='Concatenate')

        dense_layer1 = Dense (30, activation='sigmoid')(concat)
        dense_layer2 = Dense (15, activation='sigmoid')(dense_layer1)
        dense_layer3 = Dense (10, activation='sigmoid')(dense_layer2)
        output = Dense(1, activation = 'sigmoid')(dense_layer3)
        
        adam_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

        final_model = Model(inputs=[signals, integral], outputs=output,
                        name='Final_output')

        final_model.compile(loss='mean_squared_error', optimizer=adam_optimizer)
        return final_model

    #4 channels with signals
    def build_model_4ch():
        model_m = Sequential()
        model_m.add(Conv1D(kernel_size = [2], filters = 20, strides = 2, activation='relu', input_shape=(PMTNUMBER, traceLength),data_format="channels_first"))
        model_m.add(Conv1D(kernel_size = [2], filters = 15, activation='relu'))
        model_m.add(Conv1D(kernel_size = [2], filters = 10, activation='relu'))
        
        model_m.add(Flatten())
        model_m.add(Dense (30, activation='sigmoid'))
        model_m.add(Dense (15, activation='sigmoid'))
        model_m.add(Dense (10, activation='sigmoid'))
        model_m.add(Dense(1, activation = 'sigmoid'))
        
        adam_optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model_m.compile(loss='mean_squared_error', optimizer=adam_optimizer)
        return model_m


    #define model epochs and batch size...
    epochs = 200
    batch = 512


    if (integrate == "yes"):
        estimator = KerasRegressor(build_fn=build_model_4ch_int, nb_epoch=epochs, verbose=1)
        #Reshape to 4 channels:
        X_train_rshp = X_train.reshape((X_train.shape[0],PMTNUMBER, traceLength))
        X_val = X_val.reshape((X_val.shape[0],PMTNUMBER, traceLength))

        estimator.fit([X_train_rshp,X_train_int],Y_train, batch_size = batch, epochs=epochs, validation_data=([X_val,X_val_int], Y_val))
    else:
        estimator = KerasRegressor(build_fn=build_model_4ch, nb_epoch=epochs, verbose=1)
        #Reshape to 4 channels:
        X_train_rshp = X_train.reshape((X_train.shape[0],PMTNUMBER, traceLength))
        X_val = X_val.reshape((X_val.shape[0],PMTNUMBER, traceLength))

        estimator.fit(X_train_rshp,Y_train, batch_size = batch, epochs=epochs, validation_data=(X_val, Y_val))


    #Save the model
    model_name = exp_ID+'-'+str(epochs)+'epo_CV'+str(Kfold)
    estimator.model.save('models/CV/keras-'+model_name+'.h5')
    #model_name = exp_ID+'-'+str(epochs)+'epo_CV'+str(Kfold)
    model_trained = load_model('models/CV/keras-'+model_name+'.h5')


    #Compute the table for each data set

    print("Testing train data set...")
    #Train
    resultados_train.append(compute_tables(X_train,X_train_int,Y_train,model_trained,model_name+"_train"))

    print("Testing original train data set...")
    #Train
    resultados_train_original.append(compute_tables(X_train_original,X_train_original_int,Y_train_original,model_trained,model_name+"_train_original"))

    print("Testing validation data set...")
    #Train
    resultados_val.append(compute_tables(X_val,X_val_int,Y_val,model_trained,model_name+"_val"))

    print("Testing test data set...")
    #Train
    resultados_test.append(compute_tables(X_test,X_test_int,Y_test,model_trained,model_name+"_test"))

    print("Testing Single Muons test data set...")
    #Train
    resultados_testSM.append(compute_tables(X_test_SM,X_test_SM_int,Y_test_SM,model_trained,model_name+"_test_SM"))



print("Finished loop over Kfolds. Dumping summary data in tables...")
#Prepare the summary table with all the results
resultados_train = np.asarray(resultados_train)
resultados_train_original = np.asarray(resultados_train_original)
resultados_val = np.asarray(resultados_val)
resultados_test = np.asarray(resultados_test)
resultados_testSM = np.asarray(resultados_testSM)

name = exp_ID+'-'+str(epochs)+'epo'
resultados = [resultados_train,resultados_train_original,resultados_val,resultados_testSM,resultados_test]
summary_tables(resultados,name)
