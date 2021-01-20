#!/usr/bin/env python

"""
 This script saves the data as the 4 traces of each WCD. It also saves some information of every station/shower event.
"""

__author__ = "Borja Serrano González"
__email__ = "borjasg@lip.pt"

import ROOT
from ROOT import TFile, TTree, TH1D, AddressOf, TCanvas, gPad
from ROOT import kBlack, kBlue, kRed, kGreen

import numpy as np

from scipy import interpolate
import os
import joblib
import h5py
import tables
import pandas as pd

import sys


import yaml

with open("./configs/config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)


path_lib = "../../LATTESsim/libEventROOT.so"
bool_val = ROOT.gSystem.Load(path_lib)

# Define the number of PMTs in the station
PMTNUMBER = cfg['global']['PMTNUMBER']

#Load the signal calibrator to calculate the e.m. contamination
signal_calibrator = joblib.load("./models/calibration_signal_SingleMuons_linear_spline.pkl")


#Input and output files definition:
filename = 'protonRndmE_alt5200_invEgen_newconcept_cylindrical_D400H170cm_PMT4D15cmR150cm'
#filename = sys.argv[1]
#Input:
inputfilename = './input_files/'+filename+'.root'

# trace length in nanoseconds
traceLength = cfg['global']['traceLength']

print("PMTNUMBER: " + str(PMTNUMBER))
print("TracLength: "+str(traceLength))


f = TFile(inputfilename)
f.ls()
t = f.Get("Tree")

nbins = 310
hTrace = TH1D("AverageTrace",";t (ns)",nbins,-10.0,300.0)

e = ROOT.EventROOT();


NEVENTS = 0
ie = 0
NSTATIONS = 0

t.SetBranchAddress("EventROOT",AddressOf(e));
nentries = t.GetEntriesFast()
print('nentries',nentries)

# comment nentries to run over all file
totalEntries = nentries  #nentries    #maximum of 3693
TrainEntries = 2000



def FillTraceROOT(pmt, hTrace,tStart):
    time = 0.0
    trace = pmt.GetTrace()
    for trace_iter in trace:
        time = trace_iter.first
        hTrace.Fill(time - tStart, trace_iter.second)
    return

def FillTracePy(pmt, SumTrace, tStart):
    time = 0.0
    trace = pmt.GetTrace()
    for trace_iter in trace:
        time = trace_iter.first
        SumTrace.append([(time - tStart),trace_iter.second])
    return


# defines a fixed array of limit size and updates index with signal recorded
def SimpleFillTracePy(pmt, tStart, limit = 100):
    #make row of limit nanoseconds
    FilledSumTrace = np.zeros(limit)
    time = 0.0
    trace = pmt.GetTrace()
    for trace_iter in trace:
        time = trace_iter.first
        idx=int(time - tStart)
        if idx<limit:        
            FilledSumTrace[idx] = trace_iter.second
    return FilledSumTrace

#Get array with positions from a TVector3
def GetTV3Position(position):
    #Build an array to store coordinates
    coordinates = []
    #Read the coordinates from the ROOT TVector3
    coordinates.append(position.X())
    coordinates.append(position.Y())
    coordinates.append(position.Z())
    return coordinates

ie = 0
#ie = 1999

FilledSumTrace = []

SumTrace = []

#pbar = tqdm_notebook(total = nentries)

# output_file = open(outputfilename, 'w', newline='')
# wr = csv.writer(output_file)#, dialect='excel')
data_dump = []
data_dumpSM = []
data_dumpTest = []
data_dumpTest_AllMuons = []
data_infoTest_AllMuons = []
data_infoSM = []
data_infoTest = []
data_info = []
Shower_event_info = []

contTrainEntries = 0

columns_stations_info = ["Shower_id","Station_id","E_T [GeV]","E_em [GeV]","E_mu [GeV]","Npart","Nmu","Nelectrons","Nphotons","OpticalHits","Distance_to_core","Signal_mu","Signal_em","Contamination","SingleMuon_class","Class"]
columns_shower_info = ["Shower_id","E0 [GeV]","Primary","Theta0 [rad]","Phi0 [rad]","Nmuons","Nelectrons","Nphotons","Nhadrons","NOtherPart","X_Core_pos [m]","Y_Core_pos [m]",
                        "Nmuons_AllSt","NSt_Muons_AllSt","Nmuons_300pe","NSt_Muons_300pe","NSt_SingleMuons_300pe","NSt_EM_300pe"]


#Output
outputfilename = './data/'+filename+'_MLPAPER_+300pe_train_SingleMuons_TL'+str(traceLength)+'_N'+str(TrainEntries)+'.hdf' #Train
outputfilenameAllInfo = './data/'+filename+'_dataset_+300pe_train_SingleMuons_info_TL'+str(traceLength)+'_N'+str(TrainEntries)+'.hdf' #Train - info

outputfilenameTest_AllMuons = './data/'+filename+'_MLPAPER_+300pe_test_TL'+str(traceLength)+'_N'+str(totalEntries-TrainEntries)+'.hdf' #Test
outputfilenameTestInfo_AllMuons =  './data/'+filename+'_dataset_+300pe_test_info_TL'+str(traceLength)+'_N'+str(totalEntries-TrainEntries)+'.hdf' #Test -info

NStationMuons = 0
NStationMuonsSingle = 0
NStationEM = 0

# loop over showers
while (ie < nentries and contTrainEntries < totalEntries):

    #if (ie+1 not in shower_id_to_export):
    #    ie+=1
    #    continue

    #if (ie < 500):
    #    ie+=1
    #    continue


    t.GetEntry(int(ie))
    #pbar.update(1)
    
    print("processing shower: ",ie,"/",totalEntries,"Number of stations with muons", NStationMuons, "Single Muons", NStationMuonsSingle, " EM ", NStationEM)
    
    NEVENTS += 1
    
    shower_id = int(ie)          
    contTrainEntries += 1
    ie += 1  

    #if ie < TrainEntries-1: #Only stations with more than 350 p.e.
    #    continue

    #event info
    E0 = np.float32(np.round(e.GetCORSIKAEvent().EPRI,2))
    THE0 = np.float32(np.round(e.GetCORSIKAEvent().THE0,2))
    PHI0 = np.float32(np.round(e.GetCORSIKAEvent().PHI0,2))
    primary = e.GetCORSIKAEvent().PRIM
    Nmuons = int(e.GetCORSIKAEvent().fNMuons)
    Nelectrons = int(e.GetCORSIKAEvent().fNElectrons)
    Nphotons = int(e.GetCORSIKAEvent().fNPhotons)
    Nhadrons = int(e.GetCORSIKAEvent().fNHadrons)
    NOther = int(e.GetCORSIKAEvent().fNOther)
    X_Core_pos = np.float32(np.round(e.GetCorePosition()[0],2))
    Y_Core_pos = np.float32(np.round(e.GetCorePosition()[1],2))

    #Start the count with the cut in 300 p.e.
    NumbStationMuons = int(0)
    NumbStationMuonsSingle = int(0)
    NumbStationEM = int(0)
    Nmuons_above300pe = int(0)
    Nmuons_below300pe = int(0)
    NumbStationMuons_below300pe = int(0)

    # loop over WCD stations
    nextStation = ROOT.TIter(e.GetStations())
    myNSTATIONS = 0
    while True:
        s = nextStation()
        
        if not s :
            break
        core = e.GetCorePosition()
        station_id = int(s.GetId())
        #print(ie, s.GetPosition().x())

        # For a test consider only stations close to the core
        #if (s.GetPosition() - core).Perp() > 10.0:
        #    continue

        nb_particles = int(s.GetNParticles())
        nb_muons = int(s.GetNMuons())
        nb_electrons = int(s.GetNElectrons())
        nb_photons = int(s.GetNPhotons())
        nb_optical = int(s.GetNOpticalHits())
        distance_to_core = (s.GetPosition() - core).Perp()
        nb_em = nb_electrons + nb_photons

        #Nmuons_above300pe += nb_muons

        #Energy (and convert to GeV)
        Energy_total = s.GetEnergyTotal()*10**(-9)
        Energy_em = s.GetEnergyElectrons()*10**(-9) + s.GetEnergyPhotons()*10**(-9)
        Energy_muons = s.GetEnergyMuons()*10**(-9)


        if nb_optical < 300: #Only stations with more than 300 p.e.
            if nb_muons > 0:
                Nmuons_below300pe += nb_muons
                NumbStationMuons_below300pe += 1
            continue
        else:
            Nmuons_above300pe += nb_muons


        if (contTrainEntries <= TrainEntries) and (nb_muons > 1): #Only single muons for training
            continue


        #if nb_muons > 1: #Only single muons for now
        #    continue

        SingleMuon = 0 #If the station have a single muon = 1, Cont. muon = 2, No muon = 0.
        if nb_muons > 0:
            SingleMuon = 2
            #print(nb_muons, nb_electrons, nb_photons)
            NumbStationMuons += 1
            NStationMuons += 1
            #Calculate the proportion of e.m. signal
            muonic_signal = 10**(interpolate.splev(np.log10(Energy_muons), signal_calibrator))
            em_signal = nb_optical - muonic_signal
            if (em_signal<0): 
                em_signal=0
            contamination = em_signal/nb_optical
            if nb_electrons == 0 and nb_photons == 0:
                NStationMuonsSingle += 1
                em_signal = 0
                contamination = em_signal/nb_optical
                SingleMuon = 1
        else:
            muonic_signal = 0
            em_signal = nb_optical - muonic_signal
            contamination = em_signal/nb_optical
            NumbStationEM += 1
            NStationEM += 1


        info_particles = [Energy_total,Energy_em,Energy_muons,nb_particles,nb_muons,nb_electrons,nb_photons,nb_optical,distance_to_core,muonic_signal,em_signal,contamination,SingleMuon]

        NSTATIONS += 1
        myNSTATIONS += 1
        tStart = ROOT.TMath.Floor(s.GetStartTime())

        traces=[]
        # loop over PMTs in WCD station
        pmts = ROOT.TIter(s.GetPMTs())
        PMTcounter = 0
        FullTrace = np.zeros(traceLength*PMTNUMBER)
        
        while True:
            pmt = pmts()
            PMTcounter += 1

            if not pmt:
                break
            
            #pmt_id = pmt.GetID() - 1    
            pmt_id = pmt.GetID() -1
            pmt_pos = pmt.GetPosition()
            #print(pmt_id)
            #place the trace generated where it corresponds respect the SiPM id
            FullTrace[traceLength*pmt_id:traceLength*(pmt_id+1)] = SimpleFillTracePy(pmt,tStart,traceLength)
            
        #print([contTrainEntries,TrainEntries,traces,nb_muons])
        if contTrainEntries <= TrainEntries:
            if nb_muons == 0:
                data_dump.append(np.concatenate((shower_id,station_id,np.asarray(FullTrace,dtype="int16"),0),axis=None))
                data_info.append(np.concatenate((shower_id,station_id,info_particles,0),axis=None))
            else:
                #Export only single muons for training
                if nb_electrons == 0 and nb_photons == 0:#nb_muons==nb_particles:
                    data_dump.append(np.concatenate((shower_id,station_id,np.asarray(FullTrace,dtype="int16"),1),axis=None))
                    data_info.append(np.concatenate((shower_id,station_id,info_particles,1),axis=None))
               
        #Use this if we have to separate single and all muons in different files.
        else:
            if nb_muons == 0:
                data_dumpTest_AllMuons.append(np.concatenate((shower_id,station_id,np.asarray(FullTrace,dtype="int16"),0),axis=None))
                data_infoTest_AllMuons.append(np.concatenate((shower_id,station_id,info_particles,0),axis=None))
            else:
                data_dumpTest_AllMuons.append(np.concatenate((shower_id,station_id,np.asarray(FullTrace,dtype="int16"),1),axis=None))
                data_infoTest_AllMuons.append(np.concatenate((shower_id,station_id,info_particles,1),axis=None))
                #if nb_electrons == 0 and nb_photons == 0:
                #    data_dumpTest.append(np.concatenate((shower_id,station_id,FullTrace,1),axis=None))
                #    data_infoTest.append(info_particles)


        del FullTrace, info_particles
        del FilledSumTrace
        FilledSumTrace = []
        #print("Dumped info for shower",shower_id,"Station ", myNSTATIONS)
        #print(data_dump)

    #Sum number of muons and stations with all muons without the cut on 300 p.e.
    Nmuons_AllSt = Nmuons_above300pe + Nmuons_below300pe
    NumbStationMuons_AllSt = NumbStationMuons_below300pe + NumbStationMuons
    #Append Shower event info
    Shower_event_info.append([shower_id,E0,primary,THE0,PHI0,Nmuons,Nelectrons,Nphotons,Nhadrons,NOther,X_Core_pos,Y_Core_pos,
                                Nmuons_AllSt,NumbStationMuons_AllSt,Nmuons_above300pe,NumbStationMuons,NumbStationMuonsSingle,NumbStationEM])



        
    if contTrainEntries == TrainEntries:
        print("Writing to disk...")
        #Save train data

        #Signals
        data_dumpDf = pd.DataFrame(data_dump)
        data_dumpDf.to_hdf(outputfilename, key='Signals', index=False)

        del data_dumpDf
        del data_dump

        #Station info
        info_dumpDf = pd.DataFrame(data_info,columns = columns_stations_info)
        info_dumpDf.to_hdf(outputfilename, key='Info_stations', index=False,mode='a')

        del data_info,info_dumpDf
        
        #Shower info
        info_dumpDf = pd.DataFrame(np.asarray(Shower_event_info),columns = columns_shower_info)
        info_dumpDf.to_hdf(outputfilename, key='Info_shower', index=False,mode='a')

        del info_dumpDf, Shower_event_info
        Shower_event_info = []
        NStationMuons = 0
        NStationMuonsSingle = 0
        NStationEM = 0



print("Writing to disk test...")

#Signals
data_dumpTestDf_AllMuons = pd.DataFrame(data_dumpTest_AllMuons)
data_dumpTestDf_AllMuons.to_hdf(outputfilenameTest_AllMuons, key='Signals', index=False)

#Info All Muons
info_dumpTestDf_AllMuons = pd.DataFrame(data_infoTest_AllMuons,columns = columns_stations_info)
info_dumpTestDf_AllMuons.to_hdf(outputfilenameTest_AllMuons, key='Info_stations', index=False,mode='a')

#Shower info
info_dumpDf = pd.DataFrame(np.asarray(Shower_event_info),columns = columns_shower_info)
info_dumpDf.to_hdf(outputfilenameTest_AllMuons, key='Info_shower', index=False,mode='a')
