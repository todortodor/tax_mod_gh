#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:48:32 2022

@author: simonl
"""

main_path = '/Users/simonl/Dropbox/Mac/Documents/taff/tax_mod/'
import sys
sys.path.append(main_path+"lib/")
import solver_funcs as s
import data_funcs as d
import treatment_funcs as t
import pandas as pd
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm as tqdm

#%% Set seaborn parameters
sns.set()
sns.set_context('talk')
sns.set_style('white')

#%%

results_path = main_path+'results/'
data_path = main_path+'data/'

EEA = d.countries_from_fta('EEA')
EU = d.countries_from_fta('EU')
NAFTA = d.countries_from_fta('NAFTA')
ASEAN = d.countries_from_fta('ASEAN')
AANZFTA = d.countries_from_fta('AANZFTA')
APTA = d.countries_from_fta('APTA')
MERCOSUR = d.countries_from_fta('MERCOSUR')


taxed_countries_list = [None]
taxing_countries_list = [None,EU,NAFTA,ASEAN,AANZFTA,APTA,EEA,MERCOSUR,
                          ['USA'],['CHN'],
                          EEA+NAFTA,EEA+ASEAN,EEA+APTA,EEA+AANZFTA,EEA+['USA'],EEA+['CHN'],
                          NAFTA+APTA,NAFTA+MERCOSUR,
                          APTA+AANZFTA,EU+NAFTA+['CHN'],EU+NAFTA+APTA]
taxed_sectors_list = [None]
specific_taxing_list = [None]
fair_tax_list = [False]
dir_num = [2]
years = [2018]
carb_cost_list = [1e-4]

cases = d.build_cases(carb_cost_list,taxed_countries_list,taxing_countries_list,
                      taxed_sectors_list,specific_taxing_list,fair_tax_list)
sols, baselines, relevant_runs, found_cases, not_found_cases = t.sol.load_sols(cases,
                                                      years,
                                                      dir_num,
                                                      results_path,
                                                      data_path,
                                                      baselines = None,
                                                      compute_sols = True,
                                                      # compute_hats= True,
                                                      return_not_found_cases=True,
                                                      drop_duplicate_runs=True,
                                                      keep='last')



#%%

ba = baselines[2018]
share_traded_ba = (ba.iot[ba.iot['row_country'] != ba.iot['col_country']].value.sum() + ba.cons[ba.cons['row_country'] != ba.cons['col_country']].value.sum()) \
                / (ba.iot.value.sum() + ba.cons.value.sum())

labor = ba.labor.sort_values('country')['2018']

emissions = [so.co2_prod.value.sum()/1e3 for so in sols]
gdp_world = [so.va.value.sum()/1e6 for so in sols]
utilities = [(so.utility.hat).mean() for so in sols]
utilities_to_consumption = [so.run.utility for so in sols]
# utilities_to_consumption = [(so.utility.hat.values*ba.cons.groupby('col_country').value.sum().values).sum()/ba.cons.value.sum() for so in sols]
utilities_to_population = [(so.utility.hat*labor).sum()/labor.sum() for so in sols]
share_traded = [(so.iot[so.iot['row_country'] != so.iot['col_country']].value.sum() + so.cons[so.cons['row_country'] != so.cons['col_country']].value.sum()) \
                / (so.iot.value.sum() + so.cons.value.sum()) for so in sols]
taxing_countries = [str(d.fta_from_countries(so.run.taxing_countries))[1:-1] for so in sols]
taxing_countries[0] = "All countries"

cost_of_reduction = [(1-utilities_to_consumption[i])/(ba.co2_prod.value.sum()-emissions[i]) for i in range(0,len(taxing_countries))]

emissions,gdp_world,utilities,share_traded,taxing_countries,utilities_to_consumption,utilities_to_population,cost_of_reduction = \
    zip(*sorted(zip(emissions,gdp_world,utilities,share_traded,taxing_countries,utilities_to_consumption,utilities_to_population,cost_of_reduction),reverse = True))

#%%

x = np.arange(len(taxing_countries))
w = 0.3

fig,ax = plt.subplots(figsize = (12,8))
ax1 = ax.twinx()

# em = ax.bar(x, height = emissions, width=2*w, align='center',label='Emissions (Gt)')
em = ax.bar(x-w/2, height = emissions, width=w, align='center',label='Emissions (Gt)')

# gd = ax1.bar(x+w/2, gdp_world, width=w, color = sns.color_palette()[1], align='center',label='World gdp (B$)')

# ut = ax1.bar(x+w/2, utilities, width=w, color = sns.color_palette()[1], align='center',label='World welfare')
# ax1.set_ylim(0.98,1.04) #utilities
# ax1.hlines(xmin=0,xmax=len(taxing_countries),y=1,ls='--', color = sns.color_palette()[1],label='Baseline')

# ut = ax1.bar(x+w/2, utilities_to_population, width=w, color = sns.color_palette()[1], align='center',label='World welfare, weighted by population')
# ax1.set_ylim(0.98,1.06) #utilities
# ax1.hlines(xmin=0,xmax=len(taxing_countries),y=1,ls='--', color = sns.color_palette()[1],label='Baseline')

# ut = ax1.bar(x+w/2, utilities_to_consumption, width=w, color = sns.color_palette()[1], align='center',label='World welfare, weighted by comsuption')
# ax1.set_ylim(0.9929,1.0018) #utilities
# ax1.hlines(xmin=0,xmax=len(taxing_countries),y=1,ls='--', color = sns.color_palette()[1],label='Baseline')

# ut = ax1.bar(x+w/2, cost_of_reduction, width=w, color = sns.color_palette()[1], align='center',label='Global Welfare cost per ton of carbon')
# ax1.set_ylim(0,1.8e-7)

# st = ax1.bar(x+w/2, share_traded, width=w, color = sns.color_palette()[1], align='center',label='Share traded')
# ax1.hlines(xmin=0,xmax=len(taxing_countries),y=share_traded_ba,ls='--',label='Baseline',color = sns.color_palette()[1])
# ax1.set_ylim(0.128,0.132)

# st = ax1.bar(x+w/2, [-np.log(c) for c in cost_of_reduction], width=w, color = sns.color_palette()[1], align='center',label='Share traded')
# ax1.hlines(xmin=0,xmax=len(taxing_countries),y=share_traded_ba,ls='--',label='Baseline',color = sns.color_palette()[1])
# ax1.set_ylim(0.128,0.132)

# ax1.hlines(xmin=0,xmax=len(taxing_countries),y=baselines[2018].va.value.sum()/1e6,ls='--', color = sns.color_palette()[1])

ax.hlines(xmin=0,xmax=len(taxing_countries),y=baselines[2018].co2_prod.value.sum()/1e3,ls='--',label='Baseline')
# ax.hlines(xmin=0,xmax=len(taxing_countries),y=baselines[2018].co2_prod.value.sum()/1e3,ls='--',label='Baseline',color='k')
# ax1.hlines(xmin=0,xmax=len(taxing_countries),y=1,ls='--',label='Baseline',color='k')

# ax.set_ylabel('Scores')
ax.set_title('Impact of localized carbon pricing\nRegions adopting a carbon price scheme with carbon price 100$/ton') 
ax.set_xticks(x, taxing_countries, rotation = 80, rotation_mode='anchor',ha='right')
ax.tick_params(axis='x', which='major', pad=-9)
ax.set_ylim(33,50)
# ax1.set_ylim(0.98,1.04) #utilities

ax.legend(loc='upper left')
ax1.legend(loc='upper right')
# ax.set_xlabel('Regions adopting a carbon price scheme with carbon price 100$/ton')

# ax.autoscale(tight=True)

plt.show()
#%%

x = np.arange(len(taxing_countries))
w = 0.3

fig,ax = plt.subplots(figsize = (12,8))
ax1 = ax.twinx()

em = ax.bar(x-w/2, height = emissions, width=w, align='center',label='Emissions (Gt)')
# ax.bar(x, z, width=w, color='g', align='center')
# gd = ax1.bar(x+w/2, gdp_world, width=w, color = sns.color_palette()[1], align='center',label='World gdp (B$)')
# ut = ax1.bar(x+w/2, utilities, width=w, color = sns.color_palette()[1], align='center',label='World welfare')
cost = ax1.bar(x+w/2, cost_of_reduction, width=w, color = sns.color_palette()[1], align='center',label='Global Welfare cost per ton of carbon')

# ax1.hlines(xmin=0,xmax=len(taxing_countries),y=baselines[2018].va.value.sum()/1e6,ls='--', color = sns.color_palette()[1])
# ax1.hlines(xmin=0,xmax=len(taxing_countries),y=1,ls='--', color = sns.color_palette()[1],label='Baseline')
ax.hlines(xmin=0,xmax=len(taxing_countries),y=baselines[2018].co2_prod.value.sum()/1e3,ls='--',label='Baseline')
# ax.hlines(xmin=0,xmax=len(taxing_countries),y=baselines[2018].co2_prod.value.sum()/1e3,ls='--',label='Baseline',color='k')
# ax1.hlines(xmin=0,xmax=len(taxing_countries),y=share_traded_ba,ls='--',label='Baseline',color = sns.color_palette()[1])
# ax1.hlines(xmin=0,xmax=len(taxing_countries),y=1,ls='--',label='Baseline',color='k')

# ax.set_ylabel('Scores')
ax.set_title('Impact of localized carbon pricing\nRegions adopting a carbon price scheme with carbon price 100$/ton') 
ax.set_xticks(x, taxing_countries, rotation = 80, rotation_mode='anchor',ha='right')
ax.tick_params(axis='x', which='major', pad=-9)
ax.set_ylim(33,50)
# ax1.set_ylim(0.993,1.0013) #utilities
ax1.set_ylim(0,1.8e-7)
ax.legend(loc='upper left')
ax1.legend(loc='upper right')
# ax.set_xlabel('Regions adopting a carbon price scheme with carbon price 100$/ton')

# ax.autoscale(tight=True)

plt.show()
