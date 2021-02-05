#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:31:02 2020

@author: Benjamin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Reading in one house's, 15 minute PECAN Street power data into Pandas Dataframe.
directory = "/Users/Benjamin/OneDrive - University of Strathclyde/EE475 Project/Data/PECAN Street/"
folder = "15minute_all_ev_data/"
file = "15minute_ev_data_661.csv"
house_df = pd.read_csv(directory + folder + file)

# Removing redundant columns & sorting by date/time.
house_df = house_df.drop(columns=['Unnamed: 0', 'dataid'])
house_df = house_df.sort_values(by='local_15min')
# Changing date format.
house_df.local_15min = house_df.local_15min.apply(lambda x : 
                                                              (x[8:10]+'/'+x[5:7]+'/'+x[2:4]+' '+x[10:-6]))
house_df = house_df.set_index('local_15min')

# Take splice of Dataframe for region to plot.
plot_region = house_df.loc['08/01/18  14:00':'09/01/18  17:00', :]
# plot_region = house_df.loc['01/05/19  00:00':'02/05/19  00:00', :]
plot_region['time'] = plot_region.index

afont = {'fontname': 'Arial'}

# Plot aggregate and submetered EV 'Active Power' data.
fig, ax1 = plt.subplots()

ev, = ax1.plot(plot_region.time, plot_region.car1, color='tab:green')
grid, = ax1.plot(plot_region.time, plot_region.grid, color='tab:blue')

# ev, = ax1.plot(plot_region.time.apply(lambda x : x[-5:]), plot_region.car1, color='tab:green')
# grid, = ax1.plot(plot_region.time.apply(lambda x : x[-5:]), plot_region.grid, color='tab:blue')

ax1.set_title('Fifteen Minute Data', **afont, size=13)
# ax1.set_xlabel('Date & Time')
ax1.set_ylabel('Active Power (kW)')
ax1.legend((grid, ev), ('Grid', 'EV'))

ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
ax1.tick_params(axis='x', rotation=50)
ax1.set_yticks((np.arange(0, 12, 2)))

fig.set_size_inches(9, 4)
plt.show()