# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:06:28 2015

@author: Eric Dodds
"""

#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import time

import preprocessing

time.clock() 

features, labels, nevents, ntrtimes = preprocessing.preprocess(subject = 1)

# separate some data for cross-validation
features_train, features_cv, labels_train, labels_cv = cross_validation.train_test_split(
    features, labels, test_size = 0.3)
    
preptime = time.clock()
print ("Preprocessing took " + str(preptime) + " seconds.")

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
    trscores[event] = classifiers[event].score(features_train, labels[:,event])
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
