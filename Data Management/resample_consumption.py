#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 19:04:14 2021

Resampling 1-minute instantaneous data as 15-minute consumption values.

@author: Benjamin
"""

import pandas as pd
import numpy as np

house_ids = ['27', '661', '1222', '1642', '3000', '4373', '4767', '5679', '6139', '8156', '9053']
new_folder = '15minute_all_ev_data_consumption/'

for house_id in house_ids:
    # Reading in one house's, one minute PECAN Street power data into Pandas Dataframe.
    directory = "/Users/Benjamin/OneDrive - University of Strathclyde/EE475 Project/Data/PECAN Street/"
    folder = "1minute_all_ev_data/"
    file = "1minute_ev_data_" + house_id + ".csv"
    house_df = pd.read_csv(directory + folder + file)
    
    # Removing redundant columns & redundant timestamp accuracy.
    cols = ['localminute', 'air1', 'car1', 'drye1', 'furnace1', 'grid', 'heater1', 'oven1', 'clotheswasher1', 'solar']
    # print(house_df.localminute.isna().sum())
    house_df = house_df[cols]
    # Making sure data is ordered correctly.
    house_df = house_df.sort_values(by='localminute')
    # Fill nulls with adjacent values.
    house_df.fillna(method='ffill', inplace=True)
    house_df.fillna(method='bfill', inplace=True)
    length = house_df.shape[0]
    
    # Build alternative DataFrame with consumption values as a dictionary.
    consumptions = {}
    null_cols = []
    
    # Resample dates.
    col = 'localminute'
    count = 1
    values = []
    values.append(house_df[col].iloc[0])
    
    for i in range(1,length):
        
        if count == 15:
            values.append(house_df[col].iloc[i])
            count = 0
            
        count += 1
            
    consumptions['local_15min'] = values
    
    # Resample aggregate and submetered columns to consumption values.
    for col in cols[1:]:
        
        if house_df[col].isna().sum() > 0:
            null_cols.append(col)
            continue
        
        count = 1
        aggregate = 0
        values = []
        values.append(house_df[col].iloc[0] * 15/60)
        
        for i in range(1,length):
            
            aggregate += house_df[col].iloc[i] / 60
            
            if count == 15:
                values.append(aggregate)
                aggregate = 0
                count = 0
                
            count += 1
    
        consumptions[col] = values
    
    # Convert to DataFrame.
    consumption_df = pd.DataFrame(consumptions)
    for col in null_cols:
        consumption_df[col] = np.nan
    
    cols[0] = 'local_15min'
    consumption_df = consumption_df[cols]
    # Export to file.
    new_file = "15minute_ev_data_" + house_id + ".csv"
    consumption_df.to_csv(directory + new_folder + new_file)
    