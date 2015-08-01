# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:06:28 2015

@author: Eric Dodds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression as SKLearnClf
#from sklearn.ensemble import RandomForestClassifier as SKLearnClf
#from sklearn import cross_validation
import time

import preprocessing


def multiCV(subjects = [1,3]):
    predictions, truths, features, clfs, meanbinneds, meanreals = np.swapaxes([crossvalidation(subject = s) for s in subjects],0,1)
    print ("Individual subject ROC scores:")
    for i in range(len(subjects)):
        print("Binned: " + str(meanbinneds[i]) + ", real: " + str(meanreals[i]) )
    print ("Combined ROC score:")
    catpredictions = np.concatenate(predictions, axis=0)
    cattruths = np.concatenate(truths, axis=0)
    catrocscores = ROCcurve(catpredictions, cattruths)
    print (catrocscores)
    return predictions, truths, meanbinneds, meanreals, catrocscores

def crossvalidation(subject=1):
    time.clock() 
    
    features_train, labels_train, nevents, _, ntrtimes, ica, FTtstep, _ = preprocessing.preprocess(subject = subject, series = range(1,7))
     
    preptime = time.clock()
    print ("Preprocessing took " + str(preptime) + " seconds.")      
     
    # train classifiers. Note we can't use just one classifier object 
    # because some events overlap so we want to be able to predict combinations of classes
    classifiers = [SKLearnClf() for event in range(nevents)]
    for event in range(nevents):
        classifiers[event].fit(features_train, labels_train[:,event])
    
    traintime = time.clock() - preptime
    print ("Trained the classifiers in " + str(traintime) + " seconds.")    
    
    # read and prepare test data
    features_cv, labels_cv, _, events_cv, ncvtimes, _, _, _ = preprocessing.preprocess(subject = subject,
                                                                          train = True,
                                                                          series = range(7,9),
                                                                          ica = ica)     
    events_cv = events_cv.astype(int) # I don't know why but it's an object array before this    
    
    # separate some data for cross-validation
    #features_train, features_cv, labels_train, labels_cv = cross_validation.train_test_split(
        #features, labels, test_size = 0.3)
        
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
    rocscoresbinned = ROCcurve(predlabels_cv, labels_cv)
    print("For binned time...")
    print("Areas under ROC curves:")
    print (rocscoresbinned)
    print ("Average ROC score:" + str(np.mean(rocscoresbinned)))
    
    # generate ROC curves for CV set in real time
    predevents_cv = preprocessing.labels_to_events(predlabels_cv, FTtstep, ncvtimes)
    rocscoresreal = ROCcurve(predevents_cv, events_cv)
    print("For real time...")
    print("Areas under ROC curves:")
    print (rocscoresreal)
    print ("Average ROC score:" + str(np.mean(rocscoresreal)))
    return predevents_cv, events_cv, features_cv, classifiers, np.mean(rocscoresbinned), np.mean(rocscoresreal)

def ROCcurve(predevents, trueevents):
    """Given matrix of predictions and ground truth, plots the ROC curves and 
    an estimate of the areas under them."""
    trues = predevents*trueevents
    falses = predevents*(1 - trueevents)
    thresholds = np.arange(0,1,.01)
    falserates = [np.sum(falses > th, axis=0)/np.sum(1-trueevents,axis=0) for th in thresholds]
    truerates = [np.sum(trues > th, axis=0)/np.sum(trueevents,axis=0) for th in thresholds]
    plt.figure()
    plt.plot(falserates, truerates)
    plt.title("ROC")
    #rocscores = np.abs(np.trapz(truerates, falserates, axis=0))
    rocscores = [roc_auc_score(trueevents[:,e],predevents[:,e]) for e in range(6)]
    return rocscores

def do_all(subfile = "EEGbears.csv"):
    cols = ['HandStart','FirstDigitTouch', 'BothStartLoadPhase','LiftOff','Replace','BothReleased']
    ids_tot = []
    pred_tot = []
    for subject in range(1,13):
        features_train, labels_train, nevents, _, ntrtimes, ica, FTtstep, _ = preprocessing.preprocess(subject = subject)
        
        # train classifiers. Note we can't use just one classifier object 
        # because some events overlap so we want to be able to predict combinations of classes
        classifiers = [SKLearnClf() for event in range(nevents)]
        for event in range(nevents):
            classifiers[event].fit(features_train, labels_train[:,event])

        # read and prepare test data
        features_test, _, _, _, ntesttimes, _, _, ids = preprocessing.preprocess(subject = subject,
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
        print ("Finished subject " + str(subject) + ".")
    # create pandas object for sbmission, write to file
    submission = pd.DataFrame(index=np.concatenate(ids_tot),
                          columns=cols,
                          data=np.concatenate(pred_tot))                             
    submission.to_csv(subfile, index_label='id', float_format='%.3f')
    return submission
