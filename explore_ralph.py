#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 10:00:52 2023

@author: slepot
"""

#%% import libraries
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

#%% load solution

results_path = main_path+'results/'
data_path = main_path+'data/'

dir_num = [1]

eta_path = ['uniform_elasticities_4.csv']
sigma_path = ['uniform_elasticities_4.csv']

dol_adjust = pd.read_csv('/Users/slepot/Documents/taff/tax_model_gh/data/dollar_adjustment.csv',
                         sep=';', 
                         decimal=',',index_col=0)
carb_cost_list = [1e-4]

taxed_countries_list = [None]

taxing_countries_list = [None]
taxed_sectors_list = [None]
specific_taxing_list = [None]
fair_tax_list = [False]
pol_pay_tax_list = [False]

cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                      taxed_sectors_list,specific_taxing_list,fair_tax_list,pol_pay_tax_list)

years = [y for y in range(2018,2019)]

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

sol = sols[0]
b = baselines[2018]
y = 2018

#%%

print(y)
trade_bsl = b.iot.groupby(
    ['row_country','row_sector','col_country']
    ).sum()+b.cons
# trade_bsl = trade_bsl.query('row_country!=col_country')
trade_cf = sol.iot.groupby(
    ['row_country','row_sector','col_country']
    ).sum()+sol.cons
# trade_cf = trade_cf.query('row_country!=col_country')

#%% Overall plot

from scipy.interpolate import splrep, BSpline

sector_map = pd.read_csv(data_path+'industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')

data_base = pd.DataFrame()
data_base['y'] = (trade_bsl.groupby(['row_country','row_sector']).sum()/trade_bsl.groupby(['row_country']).sum())/\
    (trade_bsl.groupby(['row_sector']).sum()/trade_bsl.sum())
data_base['x'] = b.co2_intensity#/b.co2_intensity.mean()

data_base_cf = pd.DataFrame()
data_base_cf['y'] = (trade_cf.groupby(['row_country','row_sector']).sum()/trade_cf.groupby(['row_country']).sum())/\
    (trade_cf.groupby(['row_sector']).sum()/trade_cf.sum())
data_base_cf['x'] = b.co2_intensity#/b.co2_intensity.mean()
data_base['z'] = data_base_cf.y/data_base.y
# data_base = data_base.loc['RUS']

x = data_base.loc[data_base.x>0].sort_values('x').x.values
y = data_base.loc[data_base.x>0].sort_values('x').z.values

fig, ax = plt.subplots()

colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
texts = []

ax.scatter(x,y)


ax.set_xlabel('Carbon intensity (tCO2eq./Mio.$)',fontsize = 40)
ax.set_ylabel('RCA change',fontsize = 40, 
              rotation=90)
plt.legend(fontsize = 25)
plt.xscale('log')
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)

plt.show()

#%%

from scipy.interpolate import splrep, BSpline

# sector_map = pd.read_csv(data_path+'industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')

data_base = pd.DataFrame()
# data_base['y'] = (trade_bsl.groupby(['row_country','row_sector']).sum()/trade_bsl.groupby(['row_country']).sum())/\
#     (trade_bsl.groupby(['row_sector']).sum()/trade_bsl.sum())
# # data_base['x'] = b.co2_intensity/b.co2_intensity.groupby(['sector']).mean()
# data_base['x'] = b.co2_intensity/b.co2_intensity.groupby(['country']).mean()
# # data_base['x'] = (b.co2_prod/b.co2_prod.groupby(['country']).sum())/\
# #     (b.co2_prod.groupby(['sector']).sum()/b.co2_prod.sum())

# data_base_cf = pd.DataFrame()
# data_base_cf['y'] = (trade_cf.groupby(['row_country','row_sector']).sum()/trade_cf.groupby(['row_country']).sum())/\
#     (trade_cf.groupby(['row_sector']).sum()/trade_cf.sum())
# # data_base_cf['x'] = b.co2_intensity/b.co2_intensity.groupby(['sector']).mean()
# data_base_cf['x'] = b.co2_intensity/b.co2_intensity.groupby(['country']).mean()
# # data_base_cf['x'] = (sol.co2_prod/sol.co2_prod.groupby(['country']).sum())/\
# #     (sol.co2_prod.groupby(['sector']).sum()/sol.co2_prod.sum())
# data_base['z'] = data_base_cf.y/data_base.y
# data_base_2 = data_base.loc['RUS']
# data_base_1 = data_base.loc['USA']
def X(trade):
    res = np.einsum('isj->',
                  trade)
    return res

def alpha_s(trade):
    num = np.einsum('isj->s',trade)
    denom = trade.sum()
    return num/denom

def alpha_is(trade):
    num = np.einsum('isj->is',trade)
    denom = np.einsum('isj->s',trade)
    return np.einsum('is,s->is',num,1/denom)

def e_s(trade,e):
    return np.einsum('is,is->s',alpha_is(trade),e)

def e_scal(trade,e):
    return np.einsum('s,s->',alpha_s(trade),e_s(trade,e))

def epsilon_s(trade,e):
    return np.einsum('s,s,,->s',
                     np.einsum('isj->s',trade),
                     e_s(trade,e),
                     1/np.einsum('isj->',trade),
                     1/e_scal(trade,e))

def epsilon_is(trade,e):
    return np.einsum('is,is,,->is',
                     np.einsum('isj->is',trade),
                     e,
                     1/np.einsum('isj->',trade),
                     1/e_scal(trade,e))

def epsilon_is_tilde(trade,e):
    return np.einsum('is,is,s->is',
                     np.einsum('isj->is',trade),
                     e,
                     1/np.einsum('isj,is->s',trade,e))

# def epsilon

N = b.country_number
S = b.sector_number
e = b.co2_intensity.value.values.reshape((N,S))

# data_base['y'] = (alpha_is(trade_cf.values.reshape(64, 42, 64)
#                            )-alpha_is(trade_bsl.values.reshape(64, 42, 64))
#                   ).ravel()/alpha_is(trade_bsl.values.reshape(64, 42, 64)).ravel()
# data_base['y'] = alpha_is(trade_cf.values.reshape(64, 42, 64)).ravel()/alpha_is(trade_bsl.values.reshape(64, 42, 64)).ravel()
# data_base['x'] = epsilon_is(trade_bsl.values.reshape(64, 42, 64),e).ravel()
data_base['y'] = alpha_is(trade_cf.values.reshape(64, 42, 64)).ravel()/alpha_is(trade_bsl.values.reshape(64, 42, 64)).ravel()
data_base['x'] = epsilon_is(trade_bsl.values.reshape(64, 42, 64),e).ravel()
# data_base['y'] =  epsilon_is(trade_bsl.values.reshape(64, 42, 64),e).ravel()*alpha_is(trade_cf.values.reshape(64, 42, 64)).ravel()/alpha_is(trade_bsl.values.reshape(64, 42, 64)).ravel()
# data_base['x'] = epsilon_is(trade_bsl.values.reshape(64, 42, 64),e).ravel()

x = data_base.sort_values('x').x.values
y = data_base.sort_values('x').y.values
# x1 = data_base_1.sort_values('x').x.values
# y1 = data_base_1.sort_values('x').z.values
# x2 = data_base_2.sort_values('x').x.values
# y2 = data_base_2.sort_values('x').z.values

fig, ax = plt.subplots()

colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
texts = []

ax.scatter(x,y)
# ax.scatter(x1,y1,label='USA',color=sns.color_palette()[1])
# ax.scatter(x2,y2,label='RUS',color='red')

ax.set_xlabel('dEpsilon',fontsize = 40)
ax.set_ylabel('d(alpha)/alpha',fontsize = 40, 
              rotation=90)
plt.legend(fontsize = 25)
plt.xscale('log')
# plt.yscale('log')
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)

plt.show()

#%%

from scipy.interpolate import splrep, BSpline

sector_map = pd.read_csv(data_path+'industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')

data_base = pd.DataFrame()
data_base['y'] = (trade_bsl.groupby(['row_country','row_sector']).sum()/trade_bsl.groupby(['row_country']).sum())/\
    (trade_bsl.groupby(['row_sector']).sum()/trade_bsl.sum())
# data_base['x'] = b.co2_intensity/b.co2_intensity.groupby(['sector']).mean()
data_base['x'] = b.co2_intensity/b.co2_intensity.groupby(['country']).mean()
# data_base['x'] = (b.co2_prod/b.co2_prod.groupby(['country']).sum())/\
#     (b.co2_prod.groupby(['sector']).sum()/b.co2_prod.sum())

data_base_cf = pd.DataFrame()
data_base_cf['y'] = (trade_cf.groupby(['row_country','row_sector']).sum()/trade_cf.groupby(['row_country']).sum())/\
    (trade_cf.groupby(['row_sector']).sum()/trade_cf.sum())
# data_base_cf['x'] = b.co2_intensity/b.co2_intensity.groupby(['sector']).mean()
data_base_cf['x'] = b.co2_intensity/b.co2_intensity.groupby(['country']).mean()
# data_base_cf['x'] = (sol.co2_prod/sol.co2_prod.groupby(['country']).sum())/\
#     (sol.co2_prod.groupby(['sector']).sum()/sol.co2_prod.sum())
data_base['z'] = data_base_cf.y/data_base.y
# data_base_2 = data_base.loc['RUS']
data_base = data_base.loc['RUS']

x = data_base.sort_values('x').x.values
y = data_base.sort_values('x').z.values
x1 = data_base_1.sort_values('x').x.values
y1 = data_base_1.sort_values('x').z.values
x2 = data_base_2.sort_values('x').x.values
y2 = data_base_2.sort_values('x').z.values

fig, ax = plt.subplots()

colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
texts = []

ax.scatter(x,y)
# ax.scatter(x1,y1,label='USA',color=sns.color_palette()[1])
# ax.scatter(x2,y2,label='RUS',color='red')

ax.set_xlabel('Carbon intensity (tCO2eq./Mio.$)',fontsize = 40)
ax.set_ylabel('RCA change',fontsize = 40, 
              rotation=90)
plt.legend(fontsize = 25)
plt.xscale('log')
# plt.yscale('log')
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=20)

plt.show()

