# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:06:28 2015

@author: Eric Dodds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import time

import preprocessing

def crossvalidation(subject=1):
    time.clock() 
    
    features, labels, nevents, ntrtimes, ica, FTtstep, _ = preprocessing.preprocess(subject = subject)
    
    preptime = time.clock()
    print ("Preprocessing took " + str(preptime) + " seconds.")    
    
    # separate some data for cross-validation
    features_train, features_cv, labels_train, labels_cv = cross_validation.train_test_split(
        features, labels, test_size = 0.3)
    
    # train classifiers. Note we can't use just one classifier object 
    # because some events overlap so we want to be able to predict combinations of classes
    classifiers = [LogisticRegression(C=1) for event in range(nevents)]
    for event in range(nevents):
        classifiers[event].fit(features_train, labels_train[:,event])
        
    traintime = time.clock() - preptime
    print ("Trained the classifiers in " + str(traintime) + " seconds.")
        
    # naively score classifiers on training set
    trscores = np.zeros((nevents))
    for event in range(nevents):
        trscores[event] = classifiers[event].score(features_train, labels_train[:,event])
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
    rocscores = np.abs(np.trapz(truerates, falserates, axis=0))
    print("Areas under ROC curves:")
    print (rocscores)
    print ("Average ROC score:" + str(np.mean(rocscores)))
    return np.mean(rocscores)

def do_all(subfile = "EEGbears.csv"):
    cols = ['HandStart','FirstDigitTouch', 'BothStartLoadPhase','LiftOff','Replace','BothReleased']
    ids_tot = []
    pred_tot = []
    for subject in range(1,9):
        features_train, labels_train, nevents, ntrtimes, ica, FTtstep, _ = preprocessing.preprocess(subject = subject)
        
        # train classifiers. Note we can't use just one classifier object 
        # because some events overlap so we want to be able to predict combinations of classes
        classifiers = [LogisticRegression(C=1) for event in range(nevents)]
        for event in range(nevents):
            classifiers[event].fit(features_train, labels_train[:,event])

        # read and prepare test data
        features_test, _, _, ntesttimes, _, _, ids = preprocessing.preprocess(subject = subject,
                                                                             train = False,
                                                                             ica = ica)
        ids_tot.append(ids)
        # get predictions for individual time steps
        ntimebins = features_test.shape[0]
        predlabels = np.zeros((ntimebins,nevents))
        for event in range(nevents):
            predlabels[:,event] = classifiers[event].predict_proba(features_test)[:,1]
        predevents = preprocessing.labels_to_events(predlabels, FTtstep, ntesttimes)
        pred_tot.append(predevents)
    # create pandas object for sbmission, write to file
    submission = pd.DataFrame(index=np.concatenate(ids_tot),
                          columns=cols,
                          data=np.concatenate(pred_tot))                             
    submission.to_csv(subfile, index_label='id', float_format='%.3f')

