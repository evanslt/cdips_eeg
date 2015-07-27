# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 15:44:55 2015

@author: Eric Dodds
"""

import pandas as pd
import numpy as np
import mne
from mne.preprocessing import ICA
from mne.time_frequency import stft
from scipy.signal import decimate
from sklearn.preprocessing import StandardScaler

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

def psd_to_features(logpsd, nbins = 7):
    """Given a numpy array of log PSD with shape (nchannels, nfreqs, ntimes), return matrix of feature vectors."""    
    nfreqs = logpsd.shape[1]
    ntimebins = logpsd.shape[2]
    freqsperbin = int(nfreqs/nbins)
    features = decimate(logpsd, freqsperbin, axis=1) # downsample in frequency dimension

    # flatten into a vector for each time point
    features = np.array(features)
    features = features.reshape(-1, ntimebins).T
    return features, ntimebins 

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
    
def preprocess(subject = 1):
    """Preprocess data for given subject number. Returns feature matrix, label/target matrix,
    the number of events (6), and the number of timepoints in the raw data."""
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
    
    # feature scaling (remove mean and standard deviation)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    return features, labels, nevents, ntrtimes
