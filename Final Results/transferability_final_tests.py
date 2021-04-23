#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 4 09:24:16 2021

Testing transferability performance using pre-loaded train/test sets for each house.

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


austin_houses = ['661', '1642', '4373', '4767', '6139', '8156']
newyork_houses = ['27', '1222', '3000', '5679', '9053']
houses_3kW = ['27', '661', '1642', '3000', '4373', '4767', '6139', '8156']
houses_7kW = ['1222', '5679']

house_ids = ['27', '661', '1222', '1642', '3000', '4373', '4767', '5679', '6139', '8156', '9053']
# Define houses for training.
train_ids = ['661', '1642', '6139']
# Define houses for testing.
test_ids = ['27', '3000', '4373', '8156']
# house_dict = {}
models = ['RF', 'ADA', 'XGB', 'LGBM', 'Cat', 'DT']
models = ['RF']

# Defining variables for 1-minute data.
window_size = 30
overlap = 29
sample_interval = 1

# Defining variables for 15-minute data.
window_size = 5
overlap = 4
sample_interval = 15


train_dfs = []
for train_id in train_ids:
    train_dfs.append(house_dict[train_id][0])
    
train_df = pd.concat(train_dfs, ignore_index=True)
# train_df = train_df.sample(frac=1/len(train_dfs))
train_df.reset_index(drop=True, inplace=True)
train = train_df[train_df.columns[0:window_size]]
# train = train_df[train_df.columns[window_size:window_size*2]]
train_labels = train_df.Classification


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

cms = []
real_consumption = []
acc_num_overall = []
mr_den_overall = []
mr_num_overall = []

# n = 1
for i in range(len(models)):
    cms.append([[0, 0], [0, 0]])
    real_consumption.append(0)
    acc_num_overall.append(0)
    mr_den_overall.append(0)
    mr_num_overall.append(0)

f_results = []
a_results = []
acc_results = []
mr_results = []



for house_id in test_ids:
    
    results = [house_id]
    
    house = house_dict[house_id]
    test_df = house[1]
    house_df = house[2]
    # if house_id == '9053':
    #     test = test_df[test_df.columns[0:window_size]]
    # else:
    #     test = test_df[test_df.columns[window_size:window_size*2]]
        
    test = test_df[test_df.columns[0:window_size]]
    test_labels = house[1].Classification
    
    # Apply PCA to train and test datasets separately as alternative input to model.
    # pca_train = apply_pca(train_df, window_size)
    # pca_test = apply_pca(test_df, window_size)
    
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
    
    i = 0
    
    for m in models:
    
        # =============================================================================
        # Using windows directly as input to model.
        # =============================================================================
    
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
    
        # Run optional alternative tests listed above.
        for t in tests:
    
            if (t == '0.7 Prob' or t == 'Corrections') and m != 'ADA':
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
                # Using corrected predictions.
                test_df = apply_corrections(
                    test_df, window_size, overlap, sample_interval)
                predictions = test_df['Predictions']
    
            report = classification_report(test_labels, predictions)
            accuracy = accuracy_score(test_labels, predictions)*100
            print(house_id + ' ' + m + ' ' + t +
                  ' Classification Report:\n' + report)
    
            # Generate confusion matrix from results and calculate F1 score.
            cm = confusion_matrix(test_labels, predictions)
            tn = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tp = cm[1][1]
            f_score = 100 * 2*tp / (2*tp + fn + fp)
            cm_str = str(tn)+' \t '+str(fp)+'\n'+str(fn)+'  \t  '+str(tp)
    
            results.append(accuracy)
            results.append(f_score)
            results.append(cm_str)
            results.append(report)
    
            test_df['Predictions'] = predictions
            test_df['Probabilities'] = probs
            consumption_metrics = test_data_metrics(
                test_df, house_df, id_test, window_size, overlap, sample_interval)
            for val in consumption_metrics:
                results.append(val)
            for name in metric_names:
                cols.append(m + ' ' + ' ' + t + ' ' + name)
    
            cms[i][0][0] = cms[i][0][0] + tn
            cms[i][0][1] = cms[i][0][1] + fp
            cms[i][1][0] = cms[i][1][0] + fn
            cms[i][1][1] = cms[i][1][1] + tp
    
            real_consumption[i] += consumption_metrics[3]
            acc_num_overall[i] += consumption_metrics[-3]
            mr_num_overall[i] += consumption_metrics[-2]
            mr_den_overall[i] += consumption_metrics[-1]

            i += 1
    
# ==============================================================================
    # Repeat with optional alternative data e.g. PCs or data with solar removed.
# ==============================================================================

            # i = 1
            
            # start = datetime.datetime.now()
        
            # # Fit on pca_training data
            # model = build_model(pca_train, train_labels)
            # pca_train_time = datetime.datetime.now() - start
        
            # # pca_test
            # predictions = model.predict(pca_test)
            # pca_test_time = datetime.datetime.now() - start - pca_train_time
            # probs = model.predict_proba(pca_test)[:, 1]
        
            # results.append(str(pca_train_time)[2:9])
            # results.append(str(pca_test_time)[2:9])
            # cols.append(m + ' PCA Train Time')
            # cols.append(m + ' PCA Test Time')
        
            # test_df['Predictions'] = predictions
            # test_df['Probabilities'] = probs
        
            # # Run alternative tests listed above.
            # for t in tests:
        
            #     if t == '0.7 Prob' or t == 'Corrections':
            #         # Increase probability threshold for second test.
            #         predictions = []
            #         for p in probs:
            #             if p > 0.7:
            #                 predictions.append(1)
            #             else:
            #                 predictions.append(0)
        
            #         test_df['Predictions'] = predictions
            #         test_df['Probabilities'] = probs
        
            #     if t == 'Corrections':
            #         # Using corrected predictions for final test.
            #         test_df = apply_corrections(
            #             test_df, window_size, overlap, sample_interval)
            #         predictions = test_df['Predictions']
        
            #     report = classification_report(test_labels, predictions)
            #     accuracy = accuracy_score(test_labels, predictions)*100
            #     print(house_id + ' ' + m + ' ' + t +
            #           ' PCA Classification Report:\n' + report)
        
            #     # Generate confusion matrix from results and calculate F1 score.
            #     cm = confusion_matrix(test_labels, predictions)
            #     tn = cm[0][0]
            #     fp = cm[0][1]
            #     fn = cm[1][0]
            #     tp = cm[1][1]
            #     f_score = 100 * 2*tp / (2*tp + fn + fp)
            #     cm_str = str(tn)+' \t '+str(fp)+'\n'+str(fn)+'  \t  '+str(tp)
        
            #     results.append(accuracy)
            #     results.append(f_score)
            #     results.append(cm_str)
            #     results.append(report)
        
            #     test_df['Predictions'] = predictions
            #     test_df['Probabilities'] = probs
            #     consumption_metrics = test_data_metrics(
            #         test_df, house_df, house_id, window_size, overlap, sample_interval)
        
            #     for val in consumption_metrics:
            #         results.append(val)
            #     for name in metric_names:
            #         cols.append(m + ' ' + ' ' + t + ' ' + name)
                
                
            #     cms[i][0][0] = cms[i][0][0] + tn
            #     cms[i][0][1] = cms[i][0][1] + fp
            #     cms[i][1][0] = cms[i][1][0] + fn
            #     cms[i][1][1] = cms[i][1][1] + tp
        
            #     real_consumption[i] += consumption_metrics[3]
            #     acc_num_overall[i] += consumption_metrics[-3]
            #     mr_num_overall[i] += consumption_metrics[-2]
            #     mr_den_overall[i] += consumption_metrics[-1]

    all_results.append(results)
    

# Calculate overall metrics.
for i in range(len(models)):
    
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


# Bar plot of results.
# =============================================================================

# afont = {'fontname': 'Arial'}


# x = np.arange(len(models))  # the label locations
# width = 0.2  # the width of the bars
    
# fig, ax1 = plt.subplots()

# plt.grid(color='gainsboro', linestyle='dashed', zorder=0)
# f = ax1.bar(x-width/2, f_results, width, color='tab:blue', zorder=3)
# mr = ax1.bar(x+width/2, mr_results, width, color = 'tab:green', zorder=3)

# ax1.set_title('Comparing DT Ensemble Classifiers for Transferability – 1-minute Data', **afont, size=14)
# ax1.set_ylabel('Metric Score (%)', **afont, size=13)
# ax1.set_yticks((np.arange(0, 105, 10)))
# ax1.set_xticks(x)
# ax1.set_xticklabels(models, **afont, size = 13)
# ax1.legend((f, mr), ('F-score', 'Match Rate'), prop={'size':11})

# fig.set_size_inches(9, 4.5)
# # fig.tight_layout()
# plt.show()

# =============================================================================

# Save results to file.
results_df = pd.DataFrame.from_records(all_results)
results_df.columns = cols[0:results_df.shape[1]]

directory = "/Users/Benjamin/OneDrive - University of Strathclyde/EE475 Project/Results/"
file = "transferability_15_minute_final_3kW_final_3.csv"
results_df.to_csv(directory + file)
