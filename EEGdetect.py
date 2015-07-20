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
    """Given a numpy array of log PSD, return matrix of feature vectors."""
    features = decimate(logpsd, 65, axis=1)  
    ntimebins = features.shape[-1]
    # flatten into a vector for each time point
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
    

# read in training data
trseries = range(1,8) #change 8 to 9 in the end, this was just for experiment
datafiles = ["Data/train/subj1_series{0}_data.csv".format(s) for s in trseries]
rawdata, events = prepare(datafiles)
ntrtimes = rawdata.n_times
nevents = events.shape[-1]

ica = ICA(verbose=False)
ica.fit(rawdata)
#ica.plot_sources(raw)
#other ideas: band-pass filter at freqs thought to be relevant; CSP

logpsd, FTtstep = get_logpsd(ica, rawdata)
features, ntimebins = psd_to_features(logpsd)
labels = events_to_labels(events, FTtstep, ntimebins)

# train classifiers. Note we can't use just one classifier object 
# because some events overlap so we want to be able to predict combinations of classes
classifiers = [LogisticRegression() for i in range(nevents)]
for i in range(nevents):
    classifiers[i].fit(features, labels[:,i])
    
# score classifiers on training set
trscores = np.zeros((nevents))
for event in range(nevents):
    trscores[event] = classifiers[event].score(features, labels[:,event])
print ("Scores on series 1-7 in binned time: " + str(trscores) )

# score in original time space

# score classifiers on CV set
exdata, exevents = prepare(["Data/train/subj1_series8_data.csv"])
exfeatures, FTtstep = get_features(ica, exdata)
exlabels = events_to_labels(exevents, FTtstep, exfeatures.shape[0])
testscores = np.zeros((nevents))
for event in range(nevents):
    testscores[event] = classifiers[event].score(exfeatures, exlabels[:,event])
print ("Scores on series 8 in binned time: " + str(testscores) )

# score on CV set in original time space
n_extimes = exdata.n_times

# read and prepare test data
testseries = [9,10]
testfiles = ["Data/test/subj1_series{0}_data.csv".format(s) for s in testseries]
rawtestdata = prepare(datafiles, read_events=False)
testfeatures = get_features(ica, rawtestdata)
ntesttimebins = testfeatures.shape[0]

# get predictions and errors for individual time steps
ntimes = rawdata.n_times
predlabels = np.zeros((ntimebins,nevents))
for event in range(nevents):
    predlabels[:,event] = classifiers[event].predict_proba(features)[:,1]
predevents = labels_to_events(predlabels, FTtstep, ntimes)
real_errors = np.sum(np.abs(events - predevents),0)/ntimes
print (real_errors)

plt.figure()
plt.plot(predlabels)
plt.show()

# TODO: write code to predict events in test data and put in submission format
