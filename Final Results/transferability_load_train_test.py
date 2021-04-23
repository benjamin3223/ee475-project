#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 4 08:59:58 2021

File for loading train and test data into dictionary for transferability tests.

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

house_ids = ['27', '661', '1222', '1642', '3000', '4373', '4767', '5679', '6139', '8156', '9053']
train_houses = ['1642']
test_houses = ['27']
house_dict = {}
models = ['RF', 'LGBM', 'Cat']
models = ['RF']

# Defining variables for 1-minute data.
window_size = 30
overlap = 29
sample_interval = 1
# Defining variables for 15-minute data.
window_size = 5
overlap = 4
sample_interval = 15

cols = ['House ID Train', 'House ID Test']
all_results = []

for house_id in house_ids:
    
    window_df, house_df = csv_data_to_windows(house_id, window_size, overlap, sample_interval)
    
    # Standardise data and apply PCA.
    window_df = window_df.dropna()
    window_df.reset_index(drop=True, inplace=True)
    
    train_df, test_df = data_split(window_df, window_size, overlap, sample_interval, 0.3)
    train_df = undersample(window_df)
    train = train_df[train_df.columns[0:window_size]]
    train_labels = train_df.Classification
    test = test_df[test_df.columns[0:window_size]]
    test_labels = test_df.Classification
    
    # Apply PCA to train and test datasets separately as alternative input to model.
    pca_train = apply_pca(train_df[train_df.columns[0:window_size]], window_size, labels=train_labels, n_pcs=window_size, var=0, plot_2D=0, house_id=house_id)
    pca_test = apply_pca(test_df[test_df.columns[0:window_size]], window_size, labels=test_labels, n_pcs=window_size, var=0, plot_2D=0, house_id=house_id)
    
    house_data = [train_df.copy(), test_df.copy(), house_df.copy()]
    house_dict[house_id] = house_data


