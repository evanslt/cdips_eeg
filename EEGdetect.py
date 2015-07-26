# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:06:28 2015

@author: Eric Dodds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from mne.time_frequency import stft
from scipy.signal import decimate
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

def file_to_raw(fname):
    """Return an MNE RawArray object with the data in the specified file."""
    EEGdata = pd.read_csv(fname)
    labels = EEGdata.columns.values[1:]
    info = mne.create_info(labels.tolist(), 500.0, "eeg")
    return mne.io.RawArray(EEGdata.values[:,1:].T,info, verbose = False)

def prepare(datafiles, read_events = True):
    """Given list of files, return MNE RawArray with the data in them. If 
    read_events is True (as by default), also return a numpy array with events."""
    rawdata = mne.concatenate_raws([file_to_raw(f) for f in datafiles]) 
    if read_events:
        eventfiles = [file.replace("_data", "_events") for file in datafiles]
        events = np.concatenate([pd.read_csv(f).values[:,1:] for f in eventfiles])
        return rawdata, events
    else:
        return rawdata

def get_logpsd(ica, rawdata):    
    """Given MNE ICA object already fit to data, and MNE RawArray, return numpy array
    of the log power spectral density of the data in the RawArray."""
    sources = ica.get_sources(rawdata).to_data_frame().values
    nsamples = 256 # number of samples in FT window
    FTtstep = int(nsamples/2)
    fourtrans = stft(sources.T, nsamples, tstep = FTtstep)
    return np.log(np.abs(fourtrans)**2), FTtstep

def psd_to_features(logpsd):
    """Given a numpy array of log PSD with shape (nchannels, nfreqs, ntimes), return matrix of feature vectors."""    
    #features = decimate(logpsd, freqsperbin, axis=1)  
    nfreqs = logpsd.shape[1]
    ntimebins = logpsd.shape[2]
    nbins = 10
    freqsperbin = int(nfreqs/nbins)
    # bin the frequencies; each bin's value is just the mean of the values for freqs in that bin
    #features = [[np.mean(logpsd[ch, freqsperbin*i:freqsperbin*(i+1)], axis=0) for i in range(nbins)] for ch in range(logpsd.shape[0])]
    features = decimate(logpsd, freqsperbin, axis=1)
    # TODO: handle the leftover frequencies at the top. I don't think these frequencies will matter, so this is a low priority.

    # flatten into a vector for each time point
    features = np.array(features)
    return features.reshape(-1,ntimebins).T, ntimebins 

def get_features(ica, rawdata):
    """Return matrix of feature vectors for given data. ICA object used in processing."""
    logpsd, FTtstep = get_logpsd(ica, rawdata)
    return psd_to_features(logpsd)[0], FTtstep

def events_to_labels(events, FTtstep, ntimebins):
    """Assigns a label to each bin of times corresponding to a time point in the
    Fourier transform."""
    labels = np.zeros((ntimebins,events.shape[-1]))
    for timebin in range(ntimebins-1):
        # Notice the no-future-info rule means we can't use FT values in a bin 
        # to predict the event in that bin, so we look one bin ahead.
        labels[timebin] = events[FTtstep*(timebin + 1),:]
    return labels
    
def labels_to_events(labels, FTtstep, ntimes):
    """Returns array of events corresponding to given labels; interpolation
    performed by just taking the value of the nearest bin-center."""
    events = np.zeros((ntimes, labels.shape[-1]))
    ntimebins = labels.shape[0]
    for timebin in range(ntimebins-1):
        events[FTtstep*(timebin+1):FTtstep*(timebin+2),:] = labels[timebin,:]
    return events
    
subject = 1

# read in training data
trseries = range(1,9)
datafiles = ["../../Data/train/subj{0}_series{1}_data.csv".format(subject, s) for s in trseries]
rawdata, events = prepare(datafiles)
ntrtimes = rawdata.n_times
nevents = events.shape[-1]

ica = ICA(verbose=False)
ica.fit(rawdata)

# get features and labels in time bins associated with fourier transform
logpsd, FTtstep = get_logpsd(ica, rawdata)
features, ntimebins = psd_to_features(logpsd)
labels = events_to_labels(events, FTtstep, ntimebins)

# separate some data for cross-validation
features_train, features_cv, labels_train, labels_cv = cross_validation.train_test_split(
    features, labels, test_size = 0.3)

# train classifiers. Note we can't use just one classifier object 
# because some events overlap so we want to be able to predict combinations of classes
classifiers = [LogisticRegression() for event in range(nevents)]
for event in range(nevents):
    classifiers[event].fit(features_train, labels_train[:,event])
    
# naively score classifiers on training set
trscores = np.zeros((nevents))
for event in range(nevents):
    trscores[event] = classifiers[event].score(features, labels[:,event])
print ("Scores on training set in binned time: " + str(trscores) )

# naively score classifiers on CV set
testscores = np.zeros((nevents))
for event in range(nevents):
    testscores[event] = classifiers[event].score(features_cv, labels_cv[:,event])
print ("Scores on CV set in binned time: " + str(testscores) )


# generate ROC curves for CV set in binned time
predlabels_cv = np.transpose([classifiers[e].predict_proba(features_cv)[:,1] for e in range(nevents)])
trues_cv = predlabels_cv*labels_cv 
falses_cv = predlabels_cv*(1 - labels_cv)
thresholds = np.arange(0,1,.001)
falserates = [np.sum(falses_cv > th, axis=0)/np.sum(1-labels_cv,axis=0) for th in thresholds]
truerates = [np.sum(trues_cv > th, axis=0)/np.sum(labels_cv,axis=0) for th in thresholds]
plt.plot(falserates, truerates)
rocscores = np.trapz(truerates, falserates, axis=0)
print("Areas under ROC curves:")
print (rocscores)

# read and prepare test data
#testseries = [9,10]
#testfiles = ["../../Data/test/subj{0}_series{1}_data.csv".format(subject, s) for s in testseries]
#rawtestdata = prepare(datafiles, read_events=False)
#testfeatures, _ = get_features(ica, rawtestdata)
#ntesttimebins = testfeatures.shape[0]

# get predictions and errors for individual time steps
#ntimes = rawdata.n_times
#predlabels = np.zeros((ntimebins,nevents))
#for event in range(nevents):
#    predlabels[:,event] = classifiers[event].predict_proba(features)[:,1]
#predevents = labels_to_events(predlabels, FTtstep, ntimes)
#real_errors = np.sum(np.abs(events - predevents),0)/ntimes
#print (real_errors)


# TODO: write code to predict events in test data and put in submission format
