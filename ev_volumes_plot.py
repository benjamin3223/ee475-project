#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:24:33 2021

@author: Benjamin
"""
import matplotlib.pyplot as plt

afont = {'fontname': 'Arial'}

evs = [0.02, 0.05, 0.11, 0.22, 0.4, 0.72, 1.18, 1.93, 3.27, 4.79]
years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']

fig, ax1 = plt.subplots()

ev_bar = ax1.bar(years, evs, 0.5, color='tab:blue')
ax1.set_title('Global Battery Electric Vehicle Volumes', **afont, size=13 )
ax1.set_ylabel('BEVs (Millions)')
# ax1.set_xlabel('')

fig.set_size_inches(8, 4)
plt.show()