#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:37:31 2023

@author: slepot
"""
main_path = './'
from tqdm import tqdm
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from adjustText import adjust_text
import matplotlib as mpl
# from labellines import labelLines
# import treatment_funcs as t
import lib.data_funcs as d
import lib.treatment_funcs as t
import os
import matplotlib.patches as patches

sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 25})
plt.rcParams['text.usetex'] = False

#%% load baseline

data_path = main_path+'data/'
results_path = 'results/'

baseline = d.baseline(2018, data_path)

va = baseline.va.groupby('col_country').sum()
labor = baseline.labor.set_index('country').rename_axis('col_country')['2018'].to_frame()
labor.columns = ['value']

gdp_p_c = va/labor

poor_countries = gdp_p_c.loc[gdp_p_c.value < gdp_p_c.value.loc['PER']].index.to_list()
emerging_countries = gdp_p_c.loc[(gdp_p_c.value >= gdp_p_c.value.loc['PER']) & 
                                 (gdp_p_c.value < gdp_p_c.value.loc['POL'])].index.to_list()
rich_countries = gdp_p_c.loc[gdp_p_c.value >= gdp_p_c.value.loc['POL']].index.to_list()

gdp_p_c = gdp_p_c.sort_values('value')

#%% plot figure of tax rate according to gdp p c

fig,ax = plt.subplots(figsize=(20,12))

bars1 = ax.bar(x=gdp_p_c.loc[poor_countries].sort_values('value').index,
       height=gdp_p_c.loc[poor_countries].sort_values('value').value,
       label='Poor countries\nTax rate = $25 per tCO2eq.',
       color='r')
bars2 = ax.bar(x=gdp_p_c.loc[emerging_countries].sort_values('value').index,
       height=gdp_p_c.loc[emerging_countries].sort_values('value').value,
       label='Emerging countries\nTax rate = $50 per tCO2eq.',
       color='b')
bars3 = ax.bar(x=gdp_p_c.loc[rich_countries].sort_values('value').index
       ,height=gdp_p_c.loc[rich_countries].sort_values('value').value,
       label='Developed countries\nTax rate = $75 per tCO2eq.',
       color='g')

ax.bar_label(bars1,
              labels=gdp_p_c.loc[poor_countries].sort_values('value').index,
              rotation=90,
              label_type = 'edge',
              padding=5,
              fontsize=15,
              # color=colors[g],
              zorder=99)
ax.bar_label(bars2,
              labels=gdp_p_c.loc[emerging_countries].sort_values('value').index,
              rotation=90,
              label_type = 'edge',
              padding=5,
              fontsize=15,
              # color=colors[g],
              zorder=99)
ax.bar_label(bars3,
              labels=gdp_p_c.loc[rich_countries].sort_values('value').index,
              rotation=90,
              label_type = 'edge',
              padding=5,
              fontsize=15,
              # color=colors[g],
              zorder=99)

ax.set_xticklabels([''])

ax.set_ylabel('GDP per capita (Mio. US$)')

plt.legend()
plt.show()

#%% load IMF solution

imf_run = pd.read_csv('results/2018_101/runs.csv').iloc[0]

imf_sol = t.sol(imf_run,results_path,data_path)
imf_sol.compute_solution(baseline,inplace=True)

#%% load equivalent uniform tax

uniform_tax_runs = pd.read_csv('results/2018_50/runs.csv')
eq_uniform_tax_run = uniform_tax_runs.loc[np.argmin(np.abs(uniform_tax_runs.emissions-imf_run.emissions))]

eq_uniform_tax_sol = t.sol(eq_uniform_tax_run,results_path,data_path)
eq_uniform_tax_sol.compute_solution(baseline,inplace=True)

#%% Is it more equal ?

fig,ax = plt.subplots(figsize=(20,12))

va = baseline.va.groupby('col_country').sum()
labor = baseline.labor.set_index('country').rename_axis('col_country')['2018'].to_frame()
labor.columns = ['value']

gdp_per = va/labor

ax.scatter(gdp_per.value,eq_uniform_tax_sol.utility*100-100,label='Uniform tax')
ax.scatter(gdp_per.value,imf_sol.utility*100-100,label='IMF tax')

texts = [ax.text(gdp_per.loc[country,'value'], 
        imf_sol.utility.loc[country,'hat']*100-100, 
        country,
        size=14,
        # color=colors[g],
        rotation = 0,ha='center',va='top') 
        for i,country in enumerate(gdp_per.index)]

ax.axhline(y=eq_uniform_tax_run.utility*100-100,color=sns.color_palette()[0],label='Global real income change\nUniform tax')
ax.axhline(y=imf_run.utility*100-100,color=sns.color_palette()[1],label='Global real income change\nIMF tax')

# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))

ax.set_ylabel('Real income change (%)')
ax.set_xlabel('GDP per capita (Mio. US$)')

plt.legend()
plt.show()

#%% Is it more equal ?

fig,ax = plt.subplots(figsize=(20,12))

va = baseline.va.groupby('col_country').sum()
labor = baseline.labor.set_index('country').rename_axis('col_country')['2018'].to_frame()
labor.columns = ['value']

gdp_per = va/labor

# ax.scatter(gdp_per.value,eq_uniform_tax_sol.utility,label='Uniform tax')
# ax.scatter(gdp_per.value,imf_sol.utility,label='IMF tax')
ax.scatter(gdp_per.value,imf_sol.utility/eq_uniform_tax_sol.utility)
# ax.scatter(gdp_per.value,imf_sol.utility,label='IMF tax')

texts = [ax.text(gdp_per.loc[country,'value'], 
        (imf_sol.utility/eq_uniform_tax_sol.utility).loc[country,'hat'], 
        country,
        size=12,
        # color=colors[g],
        rotation = 0,ha='center',va='top') 
        for i,country in enumerate(gdp_per.index)]

# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))

ax.set_ylabel('Real income change from Uniform tax scenario to IMF tax scenario')
ax.set_xlabel('GDP per capita (Mio. US$)')

plt.legend()
plt.show()

#%% Who would be against  it ?

va = baseline.va.groupby('col_country').sum()
labor = baseline.labor.set_index('country').rename_axis('col_country')['2018'].to_frame()
labor.columns = ['value']

gdp_p_c = va/labor

poor_countries = gdp_p_c.loc[gdp_p_c.value < gdp_p_c.value.loc['BRA']].index.to_list()
emerging_countries = gdp_p_c.loc[(gdp_p_c.value >= gdp_p_c.value.loc['BRA']) & 
                                 (gdp_p_c.value < gdp_p_c.value.loc['TWN'])].index.to_list()
rich_countries = gdp_p_c.loc[gdp_p_c.value >= gdp_p_c.value.loc['TWN']].index.to_list()

gdp_p_c = gdp_p_c.sort_values('value')

y = (imf_sol.utility/eq_uniform_tax_sol.utility).hat*100-100

fig,ax = plt.subplots(figsize=(20,12))

bars1 = ax.bar(x=gdp_p_c.loc[poor_countries].sort_values('value').index,
       height=y.loc[gdp_p_c.loc[poor_countries].sort_values('value').index.to_list()],
       label='Poor countries\nTax rate = $25 per tCO2eq.',
       color='r')
bars2 = ax.bar(x=gdp_p_c.loc[emerging_countries].sort_values('value').index,
       height=y.loc[gdp_p_c.loc[emerging_countries].sort_values('value').index.to_list()],
       label='Emerging countries\nTax rate = $50 per tCO2eq.',
       color='b')
bars3 = ax.bar(x=gdp_p_c.loc[rich_countries].sort_values('value').index
       ,height=y.loc[gdp_p_c.loc[rich_countries].sort_values('value').index.to_list()],
       label='Developed countries\nTax rate = $75 per tCO2eq.',
       color='g')

ax.bar_label(bars1,
              labels=gdp_p_c.loc[poor_countries].sort_values('value').index,
              rotation=90,
              label_type = 'edge',
              padding=5,
              fontsize=15,
              # color=colors[g],
              zorder=99)
ax.bar_label(bars2,
              labels=gdp_p_c.loc[emerging_countries].sort_values('value').index,
              rotation=90,
              label_type = 'edge',
              padding=5,
              fontsize=15,
              # color=colors[g],
              zorder=99)
ax.bar_label(bars3,
              labels=gdp_p_c.loc[rich_countries].sort_values('value').index,
              rotation=90,
              label_type = 'edge',
              padding=5,
              fontsize=15,
              # color=colors[g],
              zorder=99)

ax.set_xticklabels([''])

ax.set_ylabel('Real income change (%)\nfrom Uniform tax scenario to IMF tax scenario ')

plt.legend()
plt.show()

#%% collect utilities in both scenarios

df = pd.DataFrame()

df['IMF scenario'] = (imf_sol.utility.hat*100-100).describe()
df['Equivalent uniform scenario'] = (eq_uniform_tax_sol.utility.hat*100-100).describe()
df = df.drop('count')

# df.round(3).to_csv('temp_imf.csv')

#%% load transfers

imf_transfers = pd.read_csv('results/2018_102/4_results_contrib.csv')
eq_uniform_tax_transfers = pd.read_csv('results/2018_102/2_results_contrib.csv')

EU = d.countries_from_fta('EU')
EU_dic = {c:'EU' for c in EU}

labor = baseline.labor.set_index('country').rename_axis('country')['2018'].to_frame()
labor.columns = ['value']

labor = labor.reset_index()
labor = labor.replace({'country':EU_dic}).groupby('country').sum()

imf_transfers = imf_transfers.replace({'country':EU_dic}).groupby('country').sum()*1e6/labor
eq_uniform_tax_transfers = eq_uniform_tax_transfers.replace({'country':EU_dic}
                                                            ).groupby('country').sum()*1e6/labor

fig,ax = plt.subplots(figsize=(20,12))

N = len(imf_transfers.sort_values('value').index.to_list())

bars1 = ax.bar(x=np.arange(N)*1.5-0.25,
       height=eq_uniform_tax_transfers.sort_values('value').value,
       label='Eq. uniform  tax scheme transfers',
       # color='r',
       width = 0.5
       )


ax.bar_label(bars1,
              labels=eq_uniform_tax_transfers.sort_values('value').index.to_list(),
              rotation=90,
              label_type = 'edge',
              padding=2,
              fontsize=15,
              # color=colors[g],
              zorder=99)

bars2 = ax.bar(x=np.arange(N)*1.5+0.25,
       height=imf_transfers.loc[eq_uniform_tax_transfers.sort_values('value').index].value,
       label='IMF tax scheme transfers',
       # hatch='\\',
       # alpha=0.5,
       width = 0.5
       )

# ax.bar_label(bars2,
#               labels=imf_transfers.sort_values('value').index.to_list(),
#               rotation=90,
#               label_type = 'edge',
#               padding=5,
#               fontsize=15,
#               # color=colors[g],
#               zorder=99)

ax.set_xticklabels([''])

ax.set_ylabel('Transfers ($ per capita)')

plt.legend()
plt.yscale('symlog',linthresh=100)
plt.show()

df = pd.DataFrame()


df['Eq uniform tax scenario transfers'] = eq_uniform_tax_transfers.value
df['IMF tax scenario transfers'] = imf_transfers.value
df['relative diff percent'] = (df['IMF tax scenario transfers'] - df['Eq uniform tax scenario transfers'])*100/np.abs(df['Eq uniform tax scenario transfers'])
# df.round(0).to_csv('temp_imf_transfers.csv')

#%% load fair solution through carb price

fair_run = pd.read_csv('results/2018_103/runs.csv').iloc[2]

fair_sol = t.sol(fair_run,results_path,data_path)
fair_sol.compute_solution(baseline,inplace=True)

print(fair_sol.run.emissions)

uniform_tax_runs = pd.read_csv('results/2018_50/runs.csv')
eq_uniform_fair_tax_run = uniform_tax_runs.loc[np.argmin(np.abs(uniform_tax_runs.emissions-fair_run.emissions))]

eq_uniform_fair_tax_sol = t.sol(eq_uniform_fair_tax_run,results_path,data_path)
eq_uniform_fair_tax_sol.compute_solution(baseline,inplace=True)

#%% Is it more equal ?

fig,ax = plt.subplots(figsize=(20,12))

va = baseline.va.groupby('col_country').sum()
labor = baseline.labor.set_index('country').rename_axis('col_country')['2018'].to_frame()
labor.columns = ['value']

gdp_per = va/labor

ax.scatter(gdp_per.value,eq_uniform_tax_sol.utility,label='Uniform tax')
ax.scatter(gdp_per.value,fair_sol.utility,label='Custom tax')
# ax.scatter(gdp_per.value,fair_sol.utility/eq_uniform_tax_sol.utility)
# ax.scatter(gdp_per.value,fair_sol.utility,label='IMF tax')

texts = [ax.text(gdp_per.loc[country,'value'], 
        # (fair_sol.utility/eq_uniform_tax_sol.utility).loc[country,'hat'], 
        eq_uniform_tax_sol.utility.loc[country,'hat'], 
        country,
        size=12,
        # color=colors[g],
        rotation = 0,ha='center',va='top') 
        for i,country in enumerate(gdp_per.index)]

# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))

ax.set_ylabel('Real income change from Uniform tax scenario to Fair price scenario')
ax.set_xlabel('GDP per capita (Mio. US$)')

plt.legend()
plt.show()

#%% Who would be against  it ?

va = baseline.va.groupby('col_country').sum()
labor = baseline.labor.set_index('country').rename_axis('col_country')['2018'].to_frame()
labor.columns = ['value']

gdp_p_c = va/labor

poor_countries = gdp_p_c.loc[gdp_p_c.value < gdp_p_c.value.loc['BRA']].index.to_list()
emerging_countries = gdp_p_c.loc[(gdp_p_c.value >= gdp_p_c.value.loc['BRA']) & 
                                 (gdp_p_c.value < gdp_p_c.value.loc['TWN'])].index.to_list()
rich_countries = gdp_p_c.loc[gdp_p_c.value >= gdp_p_c.value.loc['TWN']].index.to_list()

gdp_p_c = gdp_p_c.sort_values('value')

y = (fair_sol.utility/eq_uniform_tax_sol.utility).hat*100-100

fig,ax = plt.subplots(figsize=(20,12))

bars1 = ax.bar(x=gdp_p_c.loc[poor_countries].sort_values('value').index,
       height=y.loc[gdp_p_c.loc[poor_countries].sort_values('value').index.to_list()],
       label='Poor countries\nTax rate = $25 per tCO2eq.',
       color='r')
bars2 = ax.bar(x=gdp_p_c.loc[emerging_countries].sort_values('value').index,
       height=y.loc[gdp_p_c.loc[emerging_countries].sort_values('value').index.to_list()],
       label='Emerging countries\nTax rate = $50 per tCO2eq.',
       color='b')
bars3 = ax.bar(x=gdp_p_c.loc[rich_countries].sort_values('value').index
       ,height=y.loc[gdp_p_c.loc[rich_countries].sort_values('value').index.to_list()],
       label='Developed countries\nTax rate = $75 per tCO2eq.',
       color='g')

ax.bar_label(bars1,
              labels=gdp_p_c.loc[poor_countries].sort_values('value').index,
              rotation=90,
              label_type = 'edge',
              padding=5,
              fontsize=15,
              # color=colors[g],
              zorder=99)
ax.bar_label(bars2,
              labels=gdp_p_c.loc[emerging_countries].sort_values('value').index,
              rotation=90,
              label_type = 'edge',
              padding=5,
              fontsize=15,
              # color=colors[g],
              zorder=99)
ax.bar_label(bars3,
              labels=gdp_p_c.loc[rich_countries].sort_values('value').index,
              rotation=90,
              label_type = 'edge',
              padding=5,
              fontsize=15,
              # color=colors[g],
              zorder=99)

ax.set_xticklabels([''])

ax.set_ylabel('Real income change (%)\nfrom Uniform tax scenario to Fair price scenario ')

plt.legend()
plt.show()

#%% collect utilities in both scenarios

df = pd.DataFrame()

df['Fair price scenario'] = (fair_sol.utility.hat*100-100).describe()
df['Equivalent uniform scenario'] = (eq_uniform_tax_sol.utility.hat*100-100).describe()
df = df.drop('count')

# df.round(3).to_csv('temp_imf.csv')

#%% load transfers

fair_price_transfers = pd.read_csv('results/2018_103/4_results_contrib.csv')
eq_uniform_tax_transfers = pd.read_csv('results/2018_103/7_results_contrib.csv')

EU = d.countries_from_fta('EU')
EU_dic = {c:'EU' for c in EU}

labor = baseline.labor.set_index('country').rename_axis('country')['2018'].to_frame()
labor.columns = ['value']

# labor = labor.reset_index()
# labor = labor.replace({'country':EU_dic}).groupby('country').sum()

# fair_price_transfers = fair_price_transfers.replace({'country':EU_dic}).groupby('country').sum()*1e6/labor
fair_price_transfers = fair_price_transfers.groupby('country').sum()*1e6/labor

# eq_uniform_tax_transfers = eq_uniform_tax_transfers.replace({'country':EU_dic}
#                                                             ).groupby('country').sum()*1e6/labor
eq_uniform_tax_transfers = eq_uniform_tax_transfers.groupby('country').sum()*1e6/labor

fig,ax = plt.subplots(figsize=(20,12))

N = len(fair_price_transfers.sort_values('value').index.to_list())

bars1 = ax.bar(x=np.arange(N)*1.5-0.25,
       height=eq_uniform_tax_transfers.sort_values('value').value,
       label='Eq. uniform  tax scheme transfers',
       # color='r',
       width = 0.5
       )


ax.bar_label(bars1,
              labels=eq_uniform_tax_transfers.sort_values('value').index.to_list(),
              rotation=90,
              label_type = 'edge',
              padding=2,
              fontsize=15,
              # color=colors[g],
              zorder=99)

bars2 = ax.bar(x=np.arange(N)*1.5+0.25,
       height=fair_price_transfers.loc[eq_uniform_tax_transfers.sort_values('value').index].value,
       label='Fair price scheme transfers',
       # hatch='\\',
       # alpha=0.5,
       width = 0.5
       )

# ax.bar_label(bars2,
#               labels=fair_price_transfers.sort_values('value').index.to_list(),
#               rotation=90,
#               label_type = 'edge',
#               padding=5,
#               fontsize=15,
#               # color=colors[g],
#               zorder=99)

ax.set_xticklabels([''])

ax.set_ylabel('Transfers ($ per capita)')

plt.legend()
plt.yscale('symlog',linthresh=100)
plt.show()

df = pd.DataFrame()


df['Eq uniform tax scenario transfers'] = eq_uniform_tax_transfers.value
df['Fair price scenario transfers'] = fair_price_transfers.value
df['relative diff percent'] = (df['Fair price scenario transfers'] - df['Eq uniform tax scenario transfers'])*100/np.abs(df['Eq uniform tax scenario transfers'])
# df.round(0).to_csv('temp_fair_price_transfers.csv')

#%%

plt.yscale('log')

plt.scatter(df['Fair price scenario transfers'],fair_sol.params.carb_cost_np.mean(axis=0).mean(axis=0))