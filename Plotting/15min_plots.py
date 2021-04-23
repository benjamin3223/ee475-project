#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:31:02 2020

Plotting 15-minute data with optional plotting of consumption values.

@author: Benjamin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Reading in one house's, 15 minute PECAN Street power data into Pandas Dataframe.
directory = "/Users/Benjamin/OneDrive - University of Strathclyde/EE475 Project/Data/PECAN Street/"
folder = "1minute_all_ev_data/"
folderc = "15minute_all_ev_data_consumption/"
file = "1minute_ev_data_661.csv"
house_df = pd.read_csv(directory + folder + file)
# house_dfc = pd.read_csv(directory + folderc + file)

# Removing redundant columns & sorting by date/time.
# house_df = house_df.drop(columns=['Unnamed: 0', 'dataid'])
house_df = house_df[['localminute', 'air1', 'car1', 'drye1', 'furnace1', 'grid', 'heater1', 'oven1', 'solar', 'clotheswasher1']]
house_df = house_df.sort_values(by='localminute')
# house_dfc = house_dfc.sort_values(by='localminute')
house_df.localminute = house_df.localminute.apply(lambda x : x[:-6])
# house_dfc.localminute = house_dfc.localminute.apply(lambda x : x[:-6])
# Changing date format.
# house_df.localminute = house_df.localminute.apply(lambda x : 
                                                              # (x[8:10]+'/'+x[5:7]+'/'+x[2:4]+' '+x[10:-6]))
house_df = house_df.set_index('localminute')
# house_dfc = house_dfc.set_index('localminute')
house_df.columns = ['AC', 'EV', 'Dryer', 'Furnace', 'Grid', 'Heater', 'Oven', 'Solar', 'Washer']

# Take splice of Dataframe for region to plot.
plot_region = house_df.loc['2018-07-11 00:00':'2018-07-12 00:00', :]
# plot_regionc = house_dfc.loc['2018-07-11 00:00':'2018-07-12 00:00', :]
# plot_region = house_df.loc['01/05/19  00:00':'02/05/19  00:00', :]
plot_region['Timestamp'] = plot_region.index
# plot_regionc['Timestamp'] = plot_regionc.index

afont = {'fontname': 'Arial'}

# Plot aggregate and submetered EV 'Active Power' data.
fig, ax1 = plt.subplots()

# dry, = ax1.plot(plot_region.Timestamp, plot_region.Dryer, color='tab:blue')
fur, = ax1.plot(plot_region.Timestamp, plot_region.Furnace, color='tab:orange')
# oven, = ax1.plot(plot_region.Timestamp, plot_region.Oven, color='tab:red')
# wash, = ax1.plot(plot_region.Timestamp, plot_region.Washer, color='tab:pink')
ac, = ax1.plot(plot_region.Timestamp, plot_region.AC, color='tab:gray')
ev, = ax1.plot(plot_region.Timestamp, plot_region.EV, color='tab:green')
grid, = ax1.plot(plot_region.Timestamp, plot_region.Grid, color='black')
# gridc, = ax1.plot(plot_region.Timestamp, plot_regionc.grid, color='tab:blue')

# ev, = ax1.plot(plot_region.time.apply(lambda x : x[-5:]), plot_region.car1, color='tab:green')
# grid, = ax1.plot(plot_region.time.apply(lambda x : x[-5:]), plot_region.grid, color='tab:blue')
# 
ax1.set_title('Aggregate and Sub-metered Data for House 661 – 1-minute Data', **afont, size=14)
# ax1.set_xlabel('Date & Time')
ax1.set_ylabel('Active Power (kW)', **afont, size=13)
# ax1.set_ylabel('Power', **afont, size=13)

ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
ax1.tick_params(axis='x', rotation=50)
ax1.set_yticks((np.arange(-6, 12, 2)))

# ax1.legend((ac, ev, grid), ('AC', 'EV', 'Grid'))
# plt.show()

# ax1.legend((grid, ev, ac, dry, fur, oven, wash), ('Grid', 'EV', 'AC', 'Dryer', 'Furnace', 'Oven', 'Washer'))
ax1.legend((grid, ev, ac, fur), ('Grid', 'EV', 'AC', 'Furnace'))
# ax1.legend((grid, gridc, ev), ('Instantaneous Power (kW)', 'Consumption Values (kWh)', 'EV (kW)'))
plt.grid(color='gainsboro', linestyle='dashed', zorder=0)
fig.set_size_inches(12, 5)
plt.show()


# ev, = ax1.plot(plot_region.time, plot_region.car1, color='tab:green')
# grid, = ax1.plot(plot_region.time, plot_region.grid, color='tab:blue')

# # ev, = ax1.plot(plot_region.time.apply(lambda x : x[-5:]), plot_region.car1, color='tab:green')
# # grid, = ax1.plot(plot_region.time.apply(lambda x : x[-5:]), plot_region.grid, color='tab:blue')

# ax1.set_title('Fifteen Minute Data', **afont, size=13)
# # ax1.set_xlabel('Date & Time')
# ax1.set_ylabel('Active Power (kW)')
# ax1.legend((grid, ev), ('Grid', 'EV'))

# ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
# ax1.tick_params(axis='x', rotation=50)
# ax1.set_yticks((np.arange(0, 12, 2)))

# fig.set_size_inches(9, 4)
# plt.show()