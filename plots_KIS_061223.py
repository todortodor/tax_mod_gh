#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:20:24 2023

@author: slepot
"""

main_path = './'
import sys
sys.path.append(main_path+"lib/")
import solver_funcs as s
import data_funcs as d
import treatment_funcs as t
import pandas as pd
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from adjustText import adjust_text
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (20, 15),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')

save_path = '/Users/slepot/Dropbox/Green Logistics/KIS/KIS_07_23/'
save_formats = ['eps','png','pdf']

#%% load solution

results_path = main_path+'results/'
data_path = main_path+'data/'

dir_num = [1]
# year = 2018


eta_path = ['uniform_elasticities_4.csv']
sigma_path = ['uniform_elasticities_4.csv']

dol_adjust = pd.read_csv('/Users/slepot/Documents/taff/tax_model_gh/data/dollar_adjustment.csv',
                         sep=';', 
                         decimal=',',index_col=0)
carb_cost_list = [1e-4]
# carb_cost_list = np.linspace(0,1e-4,1001).tolist()
carb_cost_list = ((dol_adjust.dollar_adjusted*100).round()/1e6).tolist()

taxed_countries_list = [None]

taxing_countries_list = [None]
taxed_sectors_list = [None]
specific_taxing_list = [None]
fair_tax_list = [False]
pol_pay_tax_list = [False]

cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                      taxed_sectors_list,specific_taxing_list,fair_tax_list,pol_pay_tax_list)

years = [y for y in range(1995,2019)]

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

# sol = sols[0]
# b = baselines[2018]
# b = b.make_np_arrays().compute_shares_and_gammas()
# sol_pol_pay = sols[1]
# sol_fair = sols[2]
#%%
years = [y for y in range(1995,2019)]
temp_sols = {y:None for y in years}
for i,y in enumerate(years):
    carb_cost = dol_adjust.loc[y,'dollar_adjusted']/1e4
    print(y)
    for sol in sols:
        if sol.run.year == y:
            if temp_sols[y] is None or np.abs(temp_sols[y].run.carb_cost - carb_cost) > np.abs(sol.run.carb_cost - carb_cost):
                temp_sols[y] = sol

palette = [sns.color_palette()[i] for i in [0,2,3]]
income_colors = {
    'Low-income' : sns.color_palette()[3],
    'Middle-income' : sns.color_palette()[0],
    'High-income' : sns.color_palette()[2],
                    }

#%%

baseline_share_trade_embedded_emissions = []
cf_share_trade_embedded_emissions = []
for i,y in enumerate(years):
    print(y)
    trade_bsl = baselines[y].iot.groupby(['row_country','row_sector','col_country']).sum()+baselines[y].cons
    trade_emissions_bsl = trade_bsl * baselines[y].co2_intensity.rename_axis(
        ['row_country','row_sector']
        )
    baseline_share_trade_embedded_emissions.append(trade_emissions_bsl.query('row_country!=col_country').sum()
        /trade_emissions_bsl.sum())
        
    trade_cf = temp_sols[y].iot.groupby(['row_country','row_sector','col_country']).sum()+temp_sols[y].cons
    trade_emissions_cf = trade_cf * baselines[y].co2_intensity.rename_axis(
        ['row_country','row_sector']
        )
    cf_share_trade_embedded_emissions.append(trade_emissions_cf.query('row_country!=col_country').sum()
        /trade_emissions_cf.sum())
    
#%%

fig, ax = plt.subplots()

ax.plot(years,np.array(baseline_share_trade_embedded_emissions)*100,
        label='Data')
ax.plot(years,np.array(cf_share_trade_embedded_emissions)*100,
        label='Tax = $100')
ax.set_ylabel('% of total emissions')

plt.legend()

plt.xticks(ticks=years, 
           labels=[str(y) for y in years],
           rotation = 45)

for save_format in save_formats:
    plt.savefig(save_path+'share_emissions_embedded_trade.'+save_format,format=save_format)

df = pd.DataFrame(index=years)
df['baseline'] = np.array(baseline_share_trade_embedded_emissions)*100
df['cf'] = np.array(cf_share_trade_embedded_emissions)*100

df.to_csv(save_path+'trade_em_shares.csv')

plt.show()

#%%
em_gains_trade = []
for i,y in enumerate(years):
    print(y)
    trade_diff = baselines[y].iot.groupby(
        ['row_country','row_sector','col_country']
        ).sum()+baselines[y].cons - (
           temp_sols[y].iot.groupby(
        ['row_country','row_sector','col_country']
        ).sum()+temp_sols[y].cons 
        )
    
    em_gains_trade.append(
        (
        trade_diff.query('row_country!=col_country')
        *baselines[y].co2_intensity.rename_axis(['row_country','row_sector'])
        ).sum()/(
        trade_diff
        *baselines[y].co2_intensity.rename_axis(['row_country','row_sector'])
        ).sum()
            )

#%%

fig, ax = plt.subplots()

ax.plot(years,np.array(em_gains_trade)*100,
        label='Emissions gains from trade')
ax.set_ylabel('Emissions gains from trade')

# plt.legend()

plt.xticks(ticks=years, 
           labels=[str(y) for y in years],
           rotation = 45)

for save_format in save_formats:
    plt.savefig(save_path+'emissions_gains_from_trade.'+save_format,format=save_format)

df = pd.DataFrame(index=years)
df['em_gains_trade'] = np.array(em_gains_trade)*100

df.to_csv(save_path+'trade_em_gains.csv')

plt.show()

#%%

from scipy.spatial.distance import pdist

gsi = [1 - pdist([baselines[y].output.value , temp_sols[y].output.value] , 
                  metric = 'correlation') for i,y in enumerate(years)] 

#%%

# y_max = max(gsi+(max(gsi)-min(gsi))/2)

fig, ax = plt.subplots()

ax.plot(years,gsi,
        label='SGI')
ax.set_ylabel('Sustainable Globalization index')
ax.axhline(1,ls = '--', color='r')
# plt.legend()

plt.xticks(ticks=years, 
           labels=[str(y) for y in years],
           rotation = 45)

for save_format in save_formats:
    plt.savefig(save_path+'SGI.'+save_format,format=save_format)
    
plt.show()

#%%

from scipy.spatial.distance import pdist

gsi_dic = {
   'Origin - Destination':[1 - pdist([baselines[y].iot.value.groupby(
       ['row_country','col_country']
       ).sum()
       +baselines[y].cons.value.groupby(
           ['row_country','col_country']
           ).sum()
       , 
        temp_sols[y].iot.value.groupby(
            ['row_country','col_country']
            ).sum()
        +temp_sols[y].cons.value.groupby(
            ['row_country','col_country']
            ).sum()] , 
               metric = 'correlation') 
        for i,y in enumerate(years)],
   'Sectors':[1 - pdist([baselines[y].output.value.groupby('sector').sum(), 
               temp_sols[y].output.value.groupby('sector').sum()], 
               metric = 'correlation') 
        for i,y in enumerate(years)], 
   # 'Origin':[1 - pdist([baselines[y].output.value.groupby('country').sum(), 
   #             temp_sols[y].output.value.groupby('country').sum()], 
   #             metric = 'correlation') 
   #      for i,y in enumerate(years)], 
   # 'Destination':[1 - pdist([baselines[y].iot.value.groupby('col_country').sum()
   #                      +baselines[y].cons.value.groupby('col_country').sum(), 
   #             temp_sols[y].iot.value.groupby('col_country').sum()
   #             +temp_sols[y].cons.value.groupby('col_country').sum()],
   #             metric = 'correlation') 
   #      for i,y in enumerate(years)], 
   'Origin - Destination - Sector':[1 - pdist([baselines[y].iot.value.groupby(
       ['row_country','row_sector','col_country']
       ).sum()
       +baselines[y].cons.value.groupby(
           ['row_country','row_sector','col_country']
           ).sum()
       , 
        temp_sols[y].iot.value.groupby(
            ['row_country','row_sector','col_country']
            ).sum()
        +temp_sols[y].cons.value.groupby(
            ['row_country','row_sector','col_country']
            ).sum()],
               metric = 'correlation') 
        for i,y in enumerate(years)], 
 }

#%%

fig, ax = plt.subplots()

for k,v in gsi_dic.items():
    ax.plot(years,v,
            label=k)
ax.set_ylabel('Sustainable globalization index')
ax.axhline(1,ls = '--', color='r')

plt.legend()

plt.xticks(ticks=years, 
           labels=[str(y) for y in years],
           rotation = 45)
for save_format in save_formats:
    plt.savefig(save_path+'different_SGIs.'+save_format,format=save_format)
    
plt.show()


#%%

from scipy.spatial.distance import pdist

gsi_dic_2 = {
   'Intermediate inputs':[1 - pdist([baselines[y].iot.value, 
               temp_sols[y].iot.value], 
               metric = 'correlation') 
        for i,y in enumerate(years)], 
   'Final goods':[1 - pdist([baselines[y].cons.value, 
               temp_sols[y].cons.value], 
               metric = 'correlation') 
        for i,y in enumerate(years)], 
 }

#%%

for k,v in gsi_dic_2.items():
    fig, ax = plt.subplots()
    ax.plot(years,v,
            label=k)
    ax.set_ylabel('Sustainable globalization index')
    ax.axhline(1,ls = '--', color='r')
    
    plt.legend()
    
    plt.xticks(ticks=years, 
               labels=[str(y) for y in years],
               rotation = 45)
    for save_format in save_formats:
        plt.savefig(save_path+f'{k}_SGIs.'+save_format,format=save_format)
        
    plt.show()

#%%

country_with_regions = pd.read_csv('data/country_continent.csv',sep=';',index_col=1)[['region_1']]
region_dict = {
    'Southern Asia': 'Southern Asia',
    'Northern Europe': 'Northern Europe',
    'Southern Europe': 'Southern Europe',
    'Northern Africa': 'Africa',
    'Polynesia': 'Oceania',
    'Middle Africa': 'Africa',
    'Caribbean': 'North America',
    'Antarctica': 'Antarctica',
    'South America': 'South America',
    'Western Asia': 'Western Asia',
    'Australia and New Zealand': 'Oceania',
    'Western Europe': 'Western Europe',
    'Eastern Europe': 'Eastern Europe',
    'Central America': 'North America',
    'Western Africa': 'Africa',
    'Northern America': 'North America',
    'Southern Africa': 'Africa',
    'Eastern Africa': 'Africa',
    'South-eastern Asia': 'South-eastern Asia',
    'Eastern Asia': 'Eastern Asia',
    'Melanesia': 'Oceania',
    'Micronesia': 'Oceania',
    'Central Asia': 'Central Asia'
}
country_with_regions['region_short'] = country_with_regions['region_1'].map(
        region_dict
    )

y = 2018

trade_bsl = (baselines[y].iot.groupby(['row_country','row_sector','col_country']).sum()
             +baselines[y].cons).groupby(['row_country','col_country']).sum().reset_index()
trade_cf = (temp_sols[y].iot.groupby(['row_country','row_sector','col_country']).sum()
            +temp_sols[y].cons).groupby(['row_country','col_country']).sum().reset_index()

trade_bsl['row_country'] = trade_bsl['row_country'].map(country_with_regions['region_short'])
trade_bsl['col_country'] = trade_bsl['col_country'].map(country_with_regions['region_short'])
trade_cf['row_country'] = trade_cf['row_country'].map(country_with_regions['region_short'])
trade_cf['col_country'] = trade_cf['col_country'].map(country_with_regions['region_short'])

trade_cf = trade_cf.groupby(['row_country','col_country']).sum()
trade_bsl = trade_bsl.groupby(['row_country','col_country']).sum()

trade_bsl['change'] = trade_cf['value']/trade_bsl['value']
trade_bsl.to_csv(save_path+'heatmap_trade_regions.csv')

def region_key(index_of_region):
    region_ranking = {
        'Africa': 1,
        'North America': 2,
        'South America': 3,
        'Oceania': 4,
        'Central Asia': 5,
        'Eastern Asia': 6,
        'South-eastern Asia': 7,
        'Southern Asia': 8,
        'Western Asia': 9,
        'Eastern Europe': 10,
        'Northern Europe': 11,
        'Southern Europe': 12,
        'Western Europe': 13
    }
    return [region_ranking[i] for i in index_of_region]

trade_bsl = trade_bsl.reset_index().sort_values(['row_country','col_country'],
    key=region_key
    ).set_index(
        ['row_country','col_country']
    )[['change']].rename_axis(['Exporter','Importer'])

trade = trade_bsl.copy()
trade['change'] = trade['change']*100-100
trade = trade.reset_index().pivot(
    index = 'Exporter',
    columns = 'Importer',
    values = 'change'
    )
trade = trade.sort_index(key=region_key).T.sort_index(key=region_key).T

import seaborn as sns
sns.heatmap(trade,cmap = sns.diverging_palette(
    25, 100, s=200, as_cmap=True
    ))  

for save_format in save_formats:
    plt.savefig(save_path+'heatmap_trade_regions_raw.'+save_format,format=save_format)
plt.show()
sns.heatmap(trade,cmap = sns.diverging_palette(
    25, 100, s=200, as_cmap=True
    ), vmin=-20, vmax=20)  

for save_format in save_formats:
    plt.savefig(save_path+'heatmap_trade_regions_centered.'+save_format,format=save_format)
plt.show()

#%%

country_with_regions = pd.read_csv('data/country_continent.csv',sep=';',index_col=1)[['region_1']]
region_dict = {
    'Southern Asia': 'Asia',
    'Northern Europe': 'Europe',
    'Southern Europe': 'Europe',
    'Northern Africa': 'Africa',
    'Polynesia': 'Oceania',
    'Middle Africa': 'Africa',
    'Caribbean': 'North America',
    'Antarctica': 'Antarctica',
    'South America': 'South America',
    'Western Asia': 'Asia',
    'Australia and New Zealand': 'Oceania',
    'Western Europe': 'Europe',
    'Eastern Europe': 'Europe',
    'Central America': 'South America',
    'Western Africa': 'Africa',
    'Northern America': 'North America',
    'Southern Africa': 'Africa',
    'Eastern Africa': 'Africa',
    'South-eastern Asia': 'Asia',
    'Eastern Asia': 'Asia',
    'Melanesia': 'Oceania',
    'Micronesia': 'Oceania',
    'Central Asia': 'Asia'
}
country_with_regions['region_short'] = country_with_regions['region_1'].map(
        region_dict
    )

y = 2018

trade_bsl = (baselines[y].iot.groupby(['row_country','row_sector','col_country']).sum()
             +baselines[y].cons).groupby(['row_country','col_country']).sum().reset_index()
trade_cf = (temp_sols[y].iot.groupby(['row_country','row_sector','col_country']).sum()
            +temp_sols[y].cons).groupby(['row_country','col_country']).sum().reset_index()

trade_bsl['row_country'] = trade_bsl['row_country'].map(country_with_regions['region_short'])
trade_bsl['col_country'] = trade_bsl['col_country'].map(country_with_regions['region_short'])
trade_cf['row_country'] = trade_cf['row_country'].map(country_with_regions['region_short'])
trade_cf['col_country'] = trade_cf['col_country'].map(country_with_regions['region_short'])

trade_cf = trade_cf.groupby(['row_country','col_country']).sum()
trade_bsl = trade_bsl.groupby(['row_country','col_country']).sum()

trade_bsl['change'] = trade_cf['value']/trade_bsl['value']
trade_bsl.to_csv(save_path+'heatmap_trade_continents.csv')

def region_key(index_of_region):
    region_ranking = {
        'Africa': 1,
        'North America': 2,
        'South America': 3,
        'Oceania': 4,
        'Asia': 5,

        'Eastern Europe': 10,
        'Northern Europe': 11,
        'Southern Europe': 12,
        'Western Europe': 13,
        'Europe':14
    }
    return [region_ranking[i] for i in index_of_region]

trade_bsl = trade_bsl.reset_index().sort_values(['row_country','col_country'],
    key=region_key
    ).set_index(
        ['row_country','col_country']
    )[['change']].rename_axis(['Exporter','Importer'])

trade = trade_bsl.copy()
trade['change'] = trade['change']*100-100
trade = trade.reset_index().pivot(
    index = 'Exporter',
    columns = 'Importer',
    values = 'change'
    )
trade = trade.sort_index(key=region_key).T.sort_index(key=region_key).T

import seaborn as sns
sns.heatmap(trade,cmap = sns.diverging_palette(
    25, 100, s=200, as_cmap=True
    ))  

for save_format in save_formats:
    plt.savefig(save_path+'heatmap_trade_continents_raw.'+save_format,format=save_format)
plt.show()
sns.heatmap(trade,cmap = sns.diverging_palette(
    25, 100, s=200, as_cmap=True
    ), vmin=-15, vmax=15)  

for save_format in save_formats:
    plt.savefig(save_path+'heatmap_trade_continents_centered.'+save_format,format=save_format)
plt.show()

#%%

y=2018

trade_bsl = (
    baselines[y].iot.query('row_country!=col_country').groupby(
    ['row_country','row_sector','col_country']
    ).sum()
    +baselines[y].cons.query('row_country!=col_country')
    ).groupby(['row_sector']).sum().rename_axis('sector').reset_index()
trade_cf = (
    temp_sols[y].iot.query('row_country!=col_country').groupby(
    ['row_country','row_sector','col_country']
    ).sum()
    +temp_sols[y].cons.query('row_country!=col_country')
    ).groupby(['row_sector']).sum().rename_axis('sector').reset_index()

sectors = pd.read_csv('data/industry_labels_after_agg_expl_wgroup.csv')
sectors['ind_code'] = sectors['ind_code'].str.replace('D','')
sectors = sectors.set_index('ind_code').rename_axis('sector')

trade_bsl['sector'] = trade_bsl['sector'].map(sectors['industry'])
output_bsl = (baselines[y].output.groupby('sector').sum()).reset_index()
output_bsl['sector'] = output_bsl['sector'].map(sectors['industry'])
# trade_bsl['value'] = trade_bsl['value']/baselines[y].output['value'].groupby('sector').sum()
trade_bsl['value'] = trade_bsl['value']/output_bsl['value']

trade_bsl = trade_bsl.set_index('sector')

trade_cf['sector'] = trade_cf['sector'].map(sectors['industry'])
output_cf = (temp_sols[y].output.groupby('sector').sum()).reset_index()
output_cf['sector'] = output_cf['sector'].map(sectors['industry'])
trade_cf['value'] = trade_cf['value']/output_cf['value']

trade_cf = trade_cf.set_index('sector')

trade_bsl['change'] = trade_cf.value/trade_bsl.value

df = trade_bsl.sort_values('change')
df['change'] = df['change']*100-100

fig,ax = plt.subplots()
ax.set_xlabel('Change in share of traded output (%)')
ax.barh(y=df.index,width=df.change)
ax.set_yticklabels([''])
ax.bar_label(ax.containers[0],
             labels=df.index.get_level_values(0), 
             # rotation=90,
              label_type = 'edge',
              padding=5,zorder=10)

for save_format in save_formats:
    plt.savefig(save_path+'sector_trade_change.'+save_format,format=save_format)

df['change'].to_csv(save_path+'sector_trade_change.csv')

plt.show()
