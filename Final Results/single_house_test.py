#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:11:15 2021

@author: Benjamin

File to test individual houses on unbalanced test dataset with various parameters
and comparing results using standard predictions vs. increased probability threshold
vs. applying corrections.

"""
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import datetime
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# Import all self defined functions.
from functions import *

# Specify house ID
house_id = '27'

# Parameters for 1-minute data.
window_size = 30
overlap = 29
sample_interval = 1

# Parameters for 15-minute data.
window_size = 5
overlap = 4
sample_interval = 15

# Specify model name.
model_n = 'RF'

# Get data from specified house in form of windows and raw aggregate and sub-metered data.
window_df, house_df = csv_data_to_windows(house_id, window_size, overlap, sample_interval)
print(len(window_df))
# Drop nulls.
window_df = window_df.dropna()
window_df.reset_index(drop=True, inplace=True)

# Obtain train and test datasets with self-defined train-test split function.
train_df, test_df = data_split(window_df, window_size, overlap, sample_interval, 0.25)
train = train_df[train_df.columns[0:window_size]]
train_labels = train_df.Classification
test = test_df[test_df.columns[0:window_size]]
test_labels = test_df.Classification

# Optional application of PCA to see 2D representation / explained variances.
apply_pca(train, window_size, labels=train_labels, n_pcs=2, time=0, var=0, plot_2D=1, house_id=house_id)

train_len = train.shape[0]
test_len = test.shape[0]
test_percent = 100*test_len / (train_len + test_len)
print(train_len, test_len, test_percent)
print(train_labels.value_counts())
print(test_labels.value_counts())

# Standard train and test of model.
# =============================================================================

rf_start = datetime.datetime.now()

# Fit on training data
rf_model = build_model(train[train.columns[0:]], train_labels, model_n)
train_time = datetime.datetime.now() - rf_start

# Test
rf_predictions = rf_model.predict(test)
test_time = datetime.datetime.now() - rf_start - train_time
rf_probs = rf_model.predict_proba(test)[:, 1]

print(train_time)
print(test_time)

report = classification_report(test_labels, rf_predictions)
accuracy = accuracy_score(test_labels, rf_predictions)*100
print('Classification Report:\n', report)
print('Accuracy Score:', accuracy_score(test_labels, rf_predictions)*100)

# Generate confusion matrix from results and calculate F1 score.
cm = confusion_matrix(test_labels, rf_predictions)
tn = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tp = cm[1][1]
f_score = 100 * 2*tp / (2*tp + fn + fp)
print(f_score)


# Increasing probability threshold for EV Charging prediction.
# =============================================================================

alt_predictions = []
for p in rf_probs:
    if p > 0.7:
        alt_predictions.append(1)
    else:
        alt_predictions.append(0)

# Higher probablity threshold.
report = classification_report(test_labels, alt_predictions)
accuracy = accuracy_score(test_labels, alt_predictions)*100
print('Classification Report:\n', report)
print('Accuracy Score:', accuracy_score(test_labels, alt_predictions)*100)

# Generate confusion matrix from results and calculate F1 score.
cm = confusion_matrix(test_labels, alt_predictions)
tn = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tp = cm[1][1]
f_score = 100 * 2*tp / (2*tp + fn + fp)
print(f_score)

# Using corrected predictions.
test_df['Predictions'] = alt_predictions
test_df['Probabilities'] = rf_probs
# test_df = test_df.iloc[0:2880,]
# test_labels = test_labels[0:2880]
test_data_metrics(test_df, house_df, house_id, window_size, overlap, sample_interval)


# Applying corrections to predictions using correction algorithm.
# =============================================================================

test_df = apply_corrections(test_df, window_size, overlap, sample_interval)
corr_predictions = test_df['Predictions']

report = classification_report(test_labels, corr_predictions)
accuracy = accuracy_score(test_labels, corr_predictions)*100

# Generate confusion matrix from results and calculate F1 score.
cm = confusion_matrix(test_labels, corr_predictions)
tn = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tp = cm[1][1]
f_score = 100 * 2*tp / (2*tp + fn + fp)
print('Classification Report:\n', report)
print('Accuracy Score:\t ', accuracy_score(test_labels, corr_predictions)*100)
print('F-score:\t\t\t ', f_score)

test_data_metrics(test_df, house_df, house_id, window_size, overlap, sample_interval)

# =============================================================================
