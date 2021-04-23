#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 05:48:52 2021

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


afont = {'fontname': 'Arial'}

# labels = []
# for threshold in models:
#     str_thresh = "{:.1f}".format(threshold)
#     labels.append('Probability > ' + str_thresh)
house_ids = ['27', '661', '1222', '1642', '3000', '4373', '4767', '5679', '6139', '8156', '9053']
house_ids = ['27', '3000', '4373', '5679', '8156', '9053']

# 1 minute
predicted = [76.56,172.535,54.56,198.165,22.8,457.545,136,47.19,105.765,135.975,94.45]
real = [119.20225,162.09445,46.90768333,250.05055,21.4793,466.34945,207.5231833,47.99486667,98.62883333,139.3025667,85.6339]

# metrics
predicted = [117, 83, 726, 75, 528, 10]
real = [162, 85, 865, 116, 215, 123]

# 15 minute
# predicted = [105.6,221.1,82.5,270.6,49,669.075,221,135.3,279.675,187.275,118.125]
# real = [143.4755,208.62625,57.72975,266.942,36.81775,599.14525,315,122.7125,167.0055,148.1435, 112.99675]
# metrics
# predicted = []
# real = []

x = np.arange(len(house_ids))  # the label locations
width = 0.2  # the width of the bars
    
# Plot aggregate and submetered EV 'Active Power' data.
fig, ax1 = plt.subplots()

plt.grid(color='gainsboro', linestyle='dashed', zorder=0)
f = ax1.bar(x-width/2, predicted, width, color='tab:gray', zorder=3)
mr = ax1.bar(x+width/2, real, width, color = 'tab:orange', zorder=3)

ax1.set_title('Predicted vs. Real Consumption for Each House – 15-minute Data', **afont, size=14)
ax1.set_ylabel('Energy Consumption (kWh)', **afont, size=13)
# ax1.set_yticks((np.arange(0, 105, 10)))
ax1.set_xticks(x)
ax1.set_xticklabels(house_ids, **afont, size = 13)
ax1.set_xlabel('House ID', **afont, size=13)
ax1.legend((f, mr), ('Predicted', 'Real'), prop={'size':11})

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.set_size_inches(12, 4)
# fig.tight_layout()
plt.show()