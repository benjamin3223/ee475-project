#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:54:21 2021

Testing effect of window size on methodology performance.

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
house_ids = ['8156']

# List different models to be used in test.
models = ['RF', 'XGB', 'ADA', 'LGBM', 'Cat', 'DT']
models = ['RF']

# Defining variables for 1-minute data.
window_size = 30
overlap = 29
sample_interval = 1
windows = np.arange(10, 50, 10)
overlap_dif = 1

# Defining variables for 15-minute data.
# window_size = 5
# overlap = 4
# sample_interval = 15
# windows = np.arange(2,13)
# overlap_dif = 1

cms = []
mr_denom = []
mr_numer = []

for i in range(len(windows)):
    cms.append([[0, 0], [0, 0]])
    mr_denom.append(0)
    mr_numer.append(0)

f_results = []
mr_results = []

for house_id in house_ids:
    
    for m in models:          
        
        for i in range(len(windows)):
            
            window_size = windows[i]
            overlap = window_size - overlap_dif
            
            # Call self defined function to read in houses data and slit into overlaping windows.
            window_df, house_df = csv_data_to_windows(house_id, window_size, overlap, sample_interval)
            window_df['Date'] = window_df['Timestamp']
            window_df['Date'] = window_df['Date'].apply(lambda x : x[:10])
              
            # Drop null rows.
            window_df = window_df.dropna()
            window_df.reset_index(drop=True, inplace=True)
            
            if i == 0:
                # Call self defined algorithm for splitting data into train & test datasets.
                train_df, test_df = data_split(window_df, window_size, overlap, sample_interval, 0.25)
                test_dates = test_df['Date'].unique()
            else:
                # Use same test days after initial split for each house so only window size is changing.
                test_df = window_df[window_df.Date.isin(test_dates)].copy()
                test_df.reset_index(drop=True, inplace=True)
                train_df = undersample(window_df[~(window_df.Date.isin(test_dates))].copy())
            
            train = train_df[train_df.columns[0:window_size]]
            train_labels = train_df.Classification
            test = test_df[test_df.columns[0:window_size]]
            test_labels = test_df.Classification
            
            train_len = train.shape[0]
            test_len = test.shape[0]
            test_percent = 100*test_len / (train_len + test_len)
            
            start = datetime.datetime.now()
            
            # Fit on training data
            model = build_model(train, train_labels, model_name=m)
            train_time = datetime.datetime.now() - start
            
            # Test
            predictions = model.predict(test)
            test_time = datetime.datetime.now() - start - train_time
            probs = model.predict_proba(test)[:, 1]
    
            # Increase probability threshold.
            predictions = []
            for p in probs:
                if p > 0.7:
                    predictions.append(1)
                else:
                    predictions.append(0)
                
            test_df['Predictions'] = predictions
            test_df['Probabilities'] = probs
    
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
            
            # Maintain record of overall confusion matrices and consumption metrics.
            cms[i][0][0] = cms[i][0][0] + tn
            cms[i][0][1] = cms[i][0][1] + fp
            cms[i][1][0] = cms[i][1][0] + fn
            cms[i][1][1] = cms[i][1][1] + tp
            
            mr_numer[i] += consumption_metrics[-2]
            mr_denom[i] += consumption_metrics[-1]

# Calculate overall metrics.
for i in range(len(windows)):
    
    tn = cms[i][0][0]
    fp = cms[i][0][1]
    fn = cms[i][1][0]
    tp = cms[i][1][1]
    f_score = 100 * 2*tp / (2*tp + fn + fp)
    f_results.append(f_score)
    
    mr_overall = (mr_numer[i]/ mr_denom[i]) * 100
    mr_results.append(mr_overall)


# Line Plot of overall F-score and Match Rate vs. window size.
# =============================================================================

afont = {'fontname': 'Arial'}
    
# Plot aggregate and submetered EV 'Active Power' data.
fig, ax1 = plt.subplots()

plt.grid(color='gainsboro', linestyle='dashed', zorder=0)
f, = ax1.plot(windows, f_results, color='tab:blue', zorder=3, marker='.')
mr, = ax1.plot(windows, mr_results, color = 'tab:green', zorder=3, marker='.')

ax1.set_title('Effect of Increasing Window Size – 1-minute Data', **afont, size=14)
ax1.set_ylabel('Metric Score (%)', **afont, size=13)
ax1.set_yticks((np.arange(0, 105, 10)))
ax1.xaxis.set_major_locator(plt.MaxNLocator(len(windows)))
ax1.set_xlabel('Window Size (ω)', **afont, size=13)
ax1.legend((f, mr), ('F-score', 'Match Rate'), prop={'size':11})

fig.set_size_inches(8, 4)
# fig.tight_layout()
plt.show()

# =============================================================================

