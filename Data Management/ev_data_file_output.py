#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 23:13:18 2020

@author: Benjamin
"""
import pandas as pd

directory = "/Users/Benjamin/OneDrive - University of Strathclyde/EE475 Project/Data/PECAN Street/"
city = "austin"
# city = "newyork"
folder = "15minute_data_"+ city + "/"
file = "15minute_data_"+ city + ".csv"
df = pd.read_csv(directory + folder + file)
data_ids = df.dataid.unique()
# print(data_ids)
# print(df.dataid.describe())
# print(df.car1.describe())

meta_file = 'metadata.csv'
meta_df = pd.read_csv(directory + meta_file)
meta_df_ev = meta_df[meta_df.car1 == 'yes']
ev_ids = meta_df_ev.dataid.unique()
ev_ids = list(map(int, ev_ids))
# print(ev_ids)


city_ev_ids = []
for i in data_ids:
    if i in ev_ids:
        city_ev_ids.append(i)

for i in city_ev_ids:
    user_df = df[df.dataid == i]
    new_folder = "15minute_all_ev_data/"
    new_file = "15minute_ev_data_"+str(i)+".csv"
    user_df.to_csv(directory + new_folder + new_file)
