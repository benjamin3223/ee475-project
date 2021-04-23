#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 21:49:55 2021

@author: Benjamin
"""

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import datetime
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from functions import *

# List houses to be included in test.
house_ids = ['27', '661', '1222', '1642', '3000',
             '4373', '4767', '5679', '6139', '8156', '9053']
# house_ids = ['9053']

# List different models to be used in test.
models = ['RF', 'XGB', 'ADA', 'LGBM', 'Cat', 'DT']
models = ['RF']

# Defining variables for 1-minute data.
# window_size = 30
# overlap = 29
# sample_interval = 1

# Defining variables for 15-minute data.
window_size = 5
overlap = 4
sample_interval = 15

# Column names for data information.
cols = ['House ID', 'Total Windows', 'Train Size',
        'Test %', 'Test Disconnected', 'Test Charging']
# Column names for various metrics.
metric_names = ['Accuracy', 'F1-Score', 'Confusion Matrix', 'Report', 'Acc', 'Match Rate', 'RMSE',
                'Consumption', 'Predicted Consumption', 'Charges', 'Predicted Charges']
# List testing scenarios for each set of model predictions.
tests = ['0.5 Prob', '0.7 Prob', 'Corrections']
tests = ['Corrections']
# Array for all results to be exported to csv.
all_results = []

# Defining arrays for overall results.
cms = []
real_consumption = []
acc_num_overall = []
mr_den_overall = []
mr_num_overall = []

# Initialise arrays depending on number of tests.
n = 4
for i in range(n):
    cms.append([[0, 0], [0, 0]])
    real_consumption.append(0)
    acc_num_overall.append(0)
    mr_den_overall.append(0)
    mr_num_overall.append(0)

f_results = []
a_results = []
acc_results = []
mr_results = []

# Iterating through houses and training/testing on each individually.
for house_id in house_ids:

    # Call self defined function to read in houses data and slit into overlaping windows.
    window_df, house_df = csv_data_to_windows(
        house_id, window_size, overlap, sample_interval)
    results = [house_id, window_df.shape[0]]

    # Drop null rows.
    window_df = window_df.dropna()
    window_df.reset_index(drop=True, inplace=True)

    # Call self defined algorithm for splitting data into train & test datasets.
    train_df, test_df = data_split(
        window_df, window_size, overlap, sample_interval, 0.25)
    train = train_df[train_df.columns[0:window_size]]
    train_labels = train_df.Classification
    test = test_df[test_df.columns[0:window_size]]
    test_labels = test_df.Classification

    # Define different train and test datasets for instantaneous vs. consumption values and solar vs. no solar.
    if house_id == '9053':  # Different for house 9053 as only house without solar generation.
        pca_train = train
        pca_test = test
        cs_train = train_df[train_df.columns[window_size:window_size*2]]
        cs_test = test_df[test_df.columns[window_size:window_size*2]]
    else:
        # Datasets with solar removed.
        pca_train = train_df[train_df.columns[window_size*2:window_size*3]]
        pca_test = test_df[test_df.columns[window_size*2:window_size*3]]
        # Datasets with solar removed and consumption values instead of instantaneous.
        cs_train = train_df[train_df.columns[window_size*3:window_size*4]]
        cs_test = test_df[test_df.columns[window_size*3:window_size*4]]

    # Datasets with consumption values instead of instantaneous.
    c_train = train_df[train_df.columns[window_size:window_size*2]]
    c_test = test_df[test_df.columns[window_size:window_size*2]]

    # Optionally apply PCA to train and test datasets separately as alternative input to model.
    # pca_train = apply_pca(train_df, window_size)
    # pca_test = apply_pca(test_df, window_size)

    # Train & test datasets size and classification ratio.
    train_len = train.shape[0]
    test_len = test.shape[0]
    test_percent = 100*test_len / (train_len + test_len)
    results.append(train_len)
    results.append(test_percent)
    results.append(test_labels.value_counts()[0])
    try:
        results.append(test_labels.value_counts()[1])
    except:
        results.append(0)

    for m in models:

        # =============================================================================
        # Using windows directly as input to model using raw data with solar generation.
        # =============================================================================
        i = 0

        start = datetime.datetime.now()

        # Fit on training data
        model = build_model(train, train_labels, model_name=m)
        train_time = datetime.datetime.now() - start

        # Test
        predictions = model.predict(test)
        test_time = datetime.datetime.now() - start - train_time
        probs = model.predict_proba(test)[:, 1]

        results.append(str(train_time)[2:9])
        results.append(str(test_time)[2:9])
        cols.append(m + ' Train Time')
        cols.append(m + ' Test Time')

        test_df['Predictions'] = predictions
        test_df['Probabilities'] = probs

        # Run alternative tests listed above.
        for t in tests:

            if t == '0.7 Prob' or t == 'Corrections':
                # Increase probability threshold.
                predictions = []
                for p in probs:
                    if p > 0.7:
                        predictions.append(1)
                    else:
                        predictions.append(0)

                test_df['Predictions'] = predictions
                test_df['Probabilities'] = probs

            if t == 'Corrections':
                # Using corrected predictions for test.
                test_df = apply_corrections(
                    test_df, window_size, overlap, sample_interval)
                predictions = test_df['Predictions']

            report = classification_report(test_labels, predictions)
            accuracy = accuracy_score(test_labels, predictions)*100
            print(house_id + ' ' + m + ' ' + t +
                  ' cs Classification Report:\n' + report)

            # Generate confusion matrix from results and calculate F1 score.
            cm = confusion_matrix(test_labels, predictions)
            tn = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tp = cm[1][1]
            f_score = 100 * 2*tp / (2*tp + fn + fp)
            cm_str = str(tn)+' \t '+str(fp)+'\n'+str(fn)+'  \t  '+str(tp)
            
            # Append all classification results for saving to file.
            results.append(accuracy)
            results.append(f_score)
            results.append(cm_str)
            results.append(report)

            test_df['Predictions'] = predictions
            test_df['Probabilities'] = probs
            consumption_metrics = test_data_metrics(
                test_df, house_df, house_id, window_size, overlap, sample_interval)
            
            # Append all consumption metrics for saving to file.
            for val in consumption_metrics:
                results.append(val)
            for name in metric_names:
                cols.append(m + ' ' + ' ' + t + ' ' + name)
                
            # Add current results to array of overall metrics.
            cms[i][0][0] = cms[i][0][0] + tn
            cms[i][0][1] = cms[i][0][1] + fp
            cms[i][1][0] = cms[i][1][0] + fn
            cms[i][1][1] = cms[i][1][1] + tp

            real_consumption[i] += consumption_metrics[3]
            acc_num_overall[i] += consumption_metrics[-3]
            mr_num_overall[i] += consumption_metrics[-2]
            mr_den_overall[i] += consumption_metrics[-1]


# ======================================================================================
        # Repeat with alternative data e.g. PCs as input to model or consumption values.
# ======================================================================================

        i = 1
        
        start = datetime.datetime.now()

        # Fit on pca_training data
        model = build_model(pca_train, train_labels)
        pca_train_time = datetime.datetime.now() - start

        # pca_test
        predictions = model.predict(pca_test)
        pca_test_time = datetime.datetime.now() - start - pca_train_time
        probs = model.predict_proba(pca_test)[:, 1]

        results.append(str(pca_train_time)[2:9])
        results.append(str(pca_test_time)[2:9])
        cols.append(m + ' PCA Train Time')
        cols.append(m + ' PCA Test Time')

        test_df['Predictions'] = predictions
        test_df['Probabilities'] = probs

        # Run alternative tests listed above.
        for t in tests:

            if t == '0.7 Prob' or t == 'Corrections':
                # Increase probability threshold.
                predictions = []
                for p in probs:
                    if p > 0.7:
                        predictions.append(1)
                    else:
                        predictions.append(0)

                test_df['Predictions'] = predictions
                test_df['Probabilities'] = probs

            if t == 'Corrections':
                # Using corrected predictions for test.
                test_df = apply_corrections(
                    test_df, window_size, overlap, sample_interval)
                predictions = test_df['Predictions']

            report = classification_report(test_labels, predictions)
            accuracy = accuracy_score(test_labels, predictions)*100
            print(house_id + ' ' + m + ' ' + t +
                  ' cs Classification Report:\n' + report)

            # Generate confusion matrix from results and calculate F1 score.
            cm = confusion_matrix(test_labels, predictions)
            tn = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tp = cm[1][1]
            f_score = 100 * 2*tp / (2*tp + fn + fp)
            cm_str = str(tn)+' \t '+str(fp)+'\n'+str(fn)+'  \t  '+str(tp)
            
            # Append all classification results for saving to file.
            results.append(accuracy)
            results.append(f_score)
            results.append(cm_str)
            results.append(report)

            test_df['Predictions'] = predictions
            test_df['Probabilities'] = probs
            consumption_metrics = test_data_metrics(
                test_df, house_df, house_id, window_size, overlap, sample_interval)
            
            # Append all consumption metrics for saving to file.
            for val in consumption_metrics:
                results.append(val)
            for name in metric_names:
                cols.append(m + ' ' + ' ' + t + ' ' + name)
                
            # Add current results to array of overall metrics.
            cms[i][0][0] = cms[i][0][0] + tn
            cms[i][0][1] = cms[i][0][1] + fp
            cms[i][1][0] = cms[i][1][0] + fn
            cms[i][1][1] = cms[i][1][1] + tp

            real_consumption[i] += consumption_metrics[3]
            acc_num_overall[i] += consumption_metrics[-3]
            mr_num_overall[i] += consumption_metrics[-2]
            mr_den_overall[i] += consumption_metrics[-1]

# ======================================================================================
        # Repeat for alternative experiments e.g. solar removed.
# ======================================================================================
            
        i = 2
        
        start = datetime.datetime.now()

        # Fit on c_training data
        model = build_model(c_train, train_labels)
        c_train_time = datetime.datetime.now() - start

        # c_test
        predictions = model.predict(c_test)
        c_test_time = datetime.datetime.now() - start - c_train_time
        probs = model.predict_proba(c_test)[:, 1]

        results.append(str(c_train_time)[2:9])
        results.append(str(c_test_time)[2:9])
        cols.append(m + ' c Train Time')
        cols.append(m + ' c Test Time')

        test_df['Predictions'] = predictions
        test_df['Probabilities'] = probs

        # Run alternative tests listed above.
        for t in tests:

            if t == '0.7 Prob' or t == 'Corrections':
                # Increase probability threshold.
                predictions = []
                for p in probs:
                    if p > 0.7:
                        predictions.append(1)
                    else:
                        predictions.append(0)

                test_df['Predictions'] = predictions
                test_df['Probabilities'] = probs

            if t == 'Corrections':
                # Using corrected predictions for test.
                test_df = apply_corrections(
                    test_df, window_size, overlap, sample_interval)
                predictions = test_df['Predictions']

            report = classification_report(test_labels, predictions)
            accuracy = accuracy_score(test_labels, predictions)*100
            print(house_id + ' ' + m + ' ' + t +
                  ' cs Classification Report:\n' + report)

            # Generate confusion matrix from results and calculate F1 score.
            cm = confusion_matrix(test_labels, predictions)
            tn = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tp = cm[1][1]
            f_score = 100 * 2*tp / (2*tp + fn + fp)
            cm_str = str(tn)+' \t '+str(fp)+'\n'+str(fn)+'  \t  '+str(tp)
            
            # Append all classification results for saving to file.
            results.append(accuracy)
            results.append(f_score)
            results.append(cm_str)
            results.append(report)

            test_df['Predictions'] = predictions
            test_df['Probabilities'] = probs
            consumption_metrics = test_data_metrics(
                test_df, house_df, house_id, window_size, overlap, sample_interval)
            
            # Append all consumption metrics for saving to file.
            for val in consumption_metrics:
                results.append(val)
            for name in metric_names:
                cols.append(m + ' ' + ' ' + t + ' ' + name)
                
            # Add current results to array of overall metrics.
            cms[i][0][0] = cms[i][0][0] + tn
            cms[i][0][1] = cms[i][0][1] + fp
            cms[i][1][0] = cms[i][1][0] + fn
            cms[i][1][1] = cms[i][1][1] + tp

            real_consumption[i] += consumption_metrics[3]
            acc_num_overall[i] += consumption_metrics[-3]
            mr_num_overall[i] += consumption_metrics[-2]
            mr_den_overall[i] += consumption_metrics[-1]

            
# ========================================================================================
        # Repeat with further experimental conditions e.g. consumption values and no solar
# ========================================================================================
            
        i = 3
        
        start = datetime.datetime.now()

        # Fit on cs_training data
        model = build_model(cs_train, train_labels)
        cs_train_time = datetime.datetime.now() - start

        # cs_test
        predictions = model.predict(cs_test)
        cs_test_time = datetime.datetime.now() - start - cs_train_time
        probs = model.predict_proba(cs_test)[:, 1]

        results.append(str(cs_train_time)[2:9])
        results.append(str(cs_test_time)[2:9])
        cols.append(m + ' cs Train Time')
        cols.append(m + ' cs Test Time')

        test_df['Predictions'] = predictions
        test_df['Probabilities'] = probs

        # Run alternative tests listed above.
        for t in tests:

            if t == '0.7 Prob' or t == 'Corrections':
                # Increase probability threshold.
                predictions = []
                for p in probs:
                    if p > 0.7:
                        predictions.append(1)
                    else:
                        predictions.append(0)

                test_df['Predictions'] = predictions
                test_df['Probabilities'] = probs

            if t == 'Corrections':
                # Using corrected predictions for test.
                test_df = apply_corrections(
                    test_df, window_size, overlap, sample_interval)
                predictions = test_df['Predictions']

            report = classification_report(test_labels, predictions)
            accuracy = accuracy_score(test_labels, predictions)*100
            print(house_id + ' ' + m + ' ' + t +
                  ' cs Classification Report:\n' + report)

            # Generate confusion matrix from results and calculate F1 score.
            cm = confusion_matrix(test_labels, predictions)
            tn = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tp = cm[1][1]
            f_score = 100 * 2*tp / (2*tp + fn + fp)
            cm_str = str(tn)+' \t '+str(fp)+'\n'+str(fn)+'  \t  '+str(tp)
            
            # Append all classification results for saving to file.
            results.append(accuracy)
            results.append(f_score)
            results.append(cm_str)
            results.append(report)

            test_df['Predictions'] = predictions
            test_df['Probabilities'] = probs
            consumption_metrics = test_data_metrics(
                test_df, house_df, house_id, window_size, overlap, sample_interval)
            
            # Append all consumption metrics for saving to file.
            for val in consumption_metrics:
                results.append(val)
            for name in metric_names:
                cols.append(m + ' ' + ' ' + t + ' ' + name)
                
            # Add current results to array of overall metrics.
            cms[i][0][0] = cms[i][0][0] + tn
            cms[i][0][1] = cms[i][0][1] + fp
            cms[i][1][0] = cms[i][1][0] + fn
            cms[i][1][1] = cms[i][1][1] + tp

            real_consumption[i] += consumption_metrics[3]
            acc_num_overall[i] += consumption_metrics[-3]
            mr_num_overall[i] += consumption_metrics[-2]
            mr_den_overall[i] += consumption_metrics[-1]

    all_results.append(results)

# Calculate all overall metrics.
for i in range(n):
    
    tn = cms[i][0][0]
    fp = cms[i][0][1]
    fn = cms[i][1][0]
    tp = cms[i][1][1]
    
    f_score = 100 * 2*tp / (2*tp + fn + fp)
    f_results.append(f_score)
    
    accuracy = 100 * (tn + tp) / (tn + tp + fn + fp)
    a_results.append(accuracy)
    
    acc = (1 - (acc_num_overall[i] / (2*real_consumption[i]))) * 100
    acc_results.append(acc)
    
    mr_overall = (mr_num_overall[i]/ mr_den_overall[i]) * 100
    mr_results.append(mr_overall)

# Convert to DataFrame
results_df = pd.DataFrame.from_records(all_results)
results_df.columns = cols[0:results_df.shape[1]]
# Save to file.
directory = "/Users/Benjamin/OneDrive - University of Strathclyde/EE475 Project/Results/"
file = "single_house_15_minute_final_consumption_solar.csv"
# results_df.to_csv(directory + file)
