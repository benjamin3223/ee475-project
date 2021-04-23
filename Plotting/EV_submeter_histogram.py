#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 09:41:06 2021

@author: Benjamin
"""

import pandas as pd
import matplotlib.pyplot as plt

directory = "/Users/Benjamin/OneDrive - University of Strathclyde/EE475 Project/Data/PECAN Street/"
folder = "1minute_all_ev_data/"
file = "1minute_ev_data_"
# List of all houses w/ non-null EV power data.
house_ids = ['27', '661', '1222', '1642', '3000', '4373', '4767', '5679', '6139', '7719', '8156', '9053']
house_ids = ['9053']

# List to store Dataframes for each houses data.
all_dfs = []

# Read in each houses data into Dataframes from csv files.
for x in house_ids:
    
    house_df = pd.read_csv(directory + folder + file + x + '.csv')
    
    # Removing redundant columns & redundant timestamp accuracy.
    house_df = house_df.car1
    all_dfs.append(house_df)
    
all_houses = pd.concat(all_dfs, ignore_index = True)

plt.figure(figsize=(10, 8))
plt.hist(all_houses, density=True, bins=100)  # density=False would make counts
plt.title('Histogram of Submetered EV Power Values')
plt.ylabel('Frequency')
plt.xlabel('Value of Active Power (kW)')
plt.show()