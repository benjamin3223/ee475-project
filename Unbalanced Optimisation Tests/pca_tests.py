#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 7 11:29:22 2021

Comparing direct use of windows to applying PCA for input to ensemble classifier.

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
house_ids = ['27', '661', '1222', '1642', '3000', '4373', '4767', '5679', '6139', '8156', '9053']
house_ids = ['27', '661', '1222', '1642', '3000', '4373', '4767', '5679', '8156', '9053']
house_ids = ['27']

# List different models to be used in test.
models = ['RF', 'XGB', 'ADA', 'LGBM', 'Cat', 'DT']
models = ['RF']

# Defining variables for 1-minute data.
# window_size = 30
# overlap = 29
# sample_interval = 1
# pcs = [30]

# Defining variables for 15-minute data.
window_size = 5
overlap = 4
sample_interval = 15
pcs = [5]

rf_results = []
mr_results = []
overall_rf_results = []
overall_mr_results = []

for n_pcs in pcs:
    f_average = 0
    mr_average = 0
    f_overall = [[0, 0], [0, 0]]
    mr_den_overall = 0
    mr_num_overall = 0
    
    pca_f_overall = [[0, 0], [0, 0]]
    pca_mr_den_overall = 0
    pca_mr_num_overall = 0
    
    for house_id in house_ids:
        
        # Call self defined function to read in houses data and slit into overlaping windows.
        window_df, house_df = csv_data_to_windows(house_id, window_size, overlap, sample_interval)
            
        # Drop null rows.
        window_df = window_df.dropna()
        window_df.reset_index(drop=True, inplace=True)
        
        # Call self defined algorithm for splitting data into train & test datasets.
        train_df, test_df = data_split(window_df, window_size, overlap, sample_interval, 0.25)
        train = train_df[train_df.columns[0:window_size]]
        train_labels = train_df.Classification
        test = test_df[test_df.columns[0:window_size]]
        test_labels = test_df.Classification    
        
        # Apply PCA to train and test datasets separately as alternative input to model.
        pca_train = apply_pca(train_df, window_size, n_pcs=n_pcs)
        pca_test = apply_pca(test_df, window_size, n_pcs=n_pcs)
        
        train_len = train.shape[0]
        test_len = test.shape[0]
        test_percent = 100*test_len / (train_len + test_len)
        
        for m in models:          
            
            start = datetime.datetime.now()
            
            # Fit on training data
            model = build_model(train, train_labels, model_name=m)
            train_time = datetime.datetime.now() - start
            
            # Test
            predictions = model.predict(test)
            test_time = datetime.datetime.now() - start - train_time
            probs = model.predict_proba(test)[:, 1]
            
            test_df['Predictions'] = predictions
            test_df['Probabilities'] = probs
            
            # Increase probability threshold.
            predictions = []
            for p in probs:
                if p > 0.7:
                    predictions.append(1)
                else:
                    predictions.append(0)
        
            # Using corrected predictions.
            test_df = apply_corrections(test_df, window_size, overlap, sample_interval)
            predictions = test_df['Predictions']
                
            report = classification_report(test_labels, predictions)
            accuracy = accuracy_score(test_labels, predictions)*100
            print(house_id + ' ' + m + ' Classification Report:\n' + report)
            
            # Generate confusion matrix from results and calculate F1 score.
            cm = confusion_matrix(test_labels, predictions)
            tn = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tp = cm[1][1]
            f_score = 100 * 2*tp / (2*tp + fn + fp)
            cm_str = str(tn)+' \t '+str(fp)+'\n'+str(fn)+'  \t  '+str(tp)
            
            consumption_metrics = test_data_metrics(test_df, house_df, house_id, window_size, overlap, sample_interval)
            
            f_average += f_score
            f_overall[0][0] = f_overall[0][0] + tn
            f_overall[0][1] = f_overall[0][1] + fp
            f_overall[1][0] = f_overall[1][0] + fn
            f_overall[1][1] = f_overall[1][1] + tp
            
            mr_average += consumption_metrics[1]
            mr_num_overall += consumption_metrics[-2]
            mr_den_overall += consumption_metrics[-1]
        
    # =============================================================================
            # Repeat with PCs as input to model.
    # =============================================================================
            
            start = datetime.datetime.now()
            
            # Fit on pca_training data
            model = build_model(pca_train, train_labels)
            pca_train_time = datetime.datetime.now() - start
            
            # pca_test
            predictions = model.predict(pca_test)
            pca_test_time = datetime.datetime.now() - start - pca_train_time
            probs = model.predict_proba(pca_test)[:, 1]
            
            test_df['Predictions'] = predictions
            test_df['Probabilities'] = probs
           
            # Increase probability threshold.
            predictions = []
            for p in probs:
                if p > 0.7:
                    predictions.append(1)
                else:
                    predictions.append(0)

            # Using corrected predictions.
            test_df = apply_corrections(test_df, window_size, overlap, sample_interval)
            predictions = test_df['Predictions']
            
            report = classification_report(test_labels, predictions)
            accuracy = accuracy_score(test_labels, predictions)*100
            print(house_id + ' ' + m + ' PCA Classification Report:\n' + report)
            
            # Generate confusion matrix from results and calculate F1 score.
            cm = confusion_matrix(test_labels, predictions)
            tn = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tp = cm[1][1]
            f_score = 100 * 2*tp / (2*tp + fn + fp)
            cm_str = str(tn)+' \t '+str(fp)+'\n'+str(fn)+'  \t  '+str(tp)
            
            consumption_metrics = test_data_metrics(test_df, house_df, house_id, window_size, overlap, sample_interval)
            
            pca_f_overall[0][0] = pca_f_overall[0][0] + tn
            pca_f_overall[0][1] = pca_f_overall[0][1] + fp
            pca_f_overall[1][0] = pca_f_overall[1][0] + fn
            pca_f_overall[1][1] = pca_f_overall[1][1] + tp
            
            pca_mr_num_overall += consumption_metrics[-2]
            pca_mr_den_overall += consumption_metrics[-1]
            
    # Calculate overall metrics.
    f_average /= len(house_ids)
    tn = f_overall[0][0]
    fp = f_overall[0][1]
    fn = f_overall[1][0]
    tp = f_overall[1][1]
    f_score = 100 * 2*tp / (2*tp + fn + fp)
    rf_results.append(f_average)
    overall_rf_results.append(f_score)
    
    mr_average /= len(house_ids)
    mr_results.append(mr_average)
    mr_overall = (mr_num_overall / mr_den_overall) * 100
    overall_mr_results.append(mr_overall)
    
    tn = pca_f_overall[0][0]
    fp = pca_f_overall[0][1]
    fn = pca_f_overall[1][0]
    tp = pca_f_overall[1][1]
    
    pca_f = 100 * 2*tp / (2*tp + fn + fp)
    pca_mr = (pca_mr_num_overall / pca_mr_den_overall) * 100


# Bar Plot
# =============================================================================

afont = {'fontname': 'Arial'}

labels = ['', 'Direct Use of Windows','', 'PCA Applied', '']

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

# Direct windows
rf_mr = mr_overall
rf_f = f_score

mr_list = [0, rf_mr, 0,  pca_mr, 0]
f_list = [0, rf_f, 0, pca_f, 0]

# Plot aggregate and submetered EV 'Active Power' data.
fig, ax1 = plt.subplots()

plt.grid(color='gainsboro', linestyle='dashed', zorder=0)
f = ax1.bar(x-width/2, f_list, width, color='tab:blue', zorder=3)
mr = ax1.bar(x+width/2, mr_list, width, color = 'tab:green', zorder=3)

ax1.xaxis.set_major_locator(plt.MaxNLocator(len(pcs)))
ax1.set_title('Effect of PCA – 15-minute Data', **afont, size=14)
ax1.set_ylabel('Metric Score (%)', **afont, size=13)
ax1.set_yticks((np.arange(0, 105, 10)))
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend((f, mr), ('F-score', 'Match Rate'), prop={'size':12})

fig.set_size_inches(7, 4)
# fig.tight_layout()
plt.show()

# =============================================================================

