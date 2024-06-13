#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:43:50 2023

@author: slepot
"""

#%% import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from adjustText import adjust_text
import matplotlib as mpl
from tqdm import tqdm
# from labellines import labelLines
# import treatment_funcs as t
import lib.data_funcs as d
import lib.treatment_funcs as t
import os
import matplotlib.patches as patches
import scipy.optimize as opt

sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 25})
plt.rcParams['text.usetex'] = False

#%% load data

data_path = 'data/'
results_path = 'results/'

save_all = False
save_format = 'eps'

carb_cost_list = np.linspace(0,1e-3,1001)
elast_path = 'cp_estimate_allyears.csv'

save_path = 'presentation_material/'+elast_path[:-4]+'/'

try:
    os.mkdir(save_path)
except:
    pass

eta_path = elast_path
sigma_path = elast_path

taxed_countries_list = [None]
taxing_countries_list = [None]
taxed_sectors_list = [None]
specific_taxing_list = [None]
fair_tax_list = [False]
y  = 2018
year = str(y)
years = [y]
dir_num = 30

cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                      taxed_sectors_list,specific_taxing_list,fair_tax_list,
                      pol_pay_tax_list=[False])

cases = [cas for cas in cases if cas['eta_path'] == cas['sigma_path']]

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

b = baselines[y]
carb_taxes = np.array([sol.params.carb_cost*1e6  for sol in sols])
t_index = np.argmin(np.abs(carb_taxes-100))
sol = sols[t_index]
N = b.country_number
S = b.sector_number

sector_map = pd.read_csv(data_path+'industry_labels_after_agg_expl_wgroup.csv')
sector_map['sector'] = sector_map['ind_code'].str.replace('D','')
sector_map.set_index('sector',inplace=True)

sol = sols[100]

save = False

#%% compute trade

trade = baselines[y].iot.groupby(level=[0,1,2]).sum()
trade['cons'] = baselines[y].cons.value
trade['baseline'] = trade.value + trade.cons
trade = trade[['baseline']]
print("Computing trade")
l_trade = []
for i,sol in tqdm(enumerate(sols)):
    l_trade.append( sol.iot.value.values.reshape((N,S,N,S)).sum(axis=-1).ravel()
                   + sol.cons.value.values )

trades = pd.concat([pd.Series(l_t) for l_t in l_trade],axis=1)
trades.index = trade.index
trade = pd.concat([trade,trades],axis=1)

print('Computing share of output traded')

iot_traded_unit = b.iot.copy()
iot_traded_unit['value'] = 1
iot_traded_unit.loc[iot_traded_unit.query("row_country == col_country").index, 'value'] = 0
cons_traded_unit = b.cons.copy()
cons_traded_unit['value'] = 1
cons_traded_unit.loc[cons_traded_unit.query("row_country == col_country").index, 'value'] = 0

share_traded = np.array([((sol.cons.value.to_numpy() * cons_traded_unit.value.to_numpy()).sum() + (sol.iot.value.to_numpy()  * iot_traded_unit.value.to_numpy()).sum())*100 /\
                         (sol.cons.value.to_numpy().sum()+sol.iot.value.to_numpy().sum()) for sol in sols])

total_output = np.array([sol.output.sum() for sol in sols])

#%% compute decomposition

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



l_term_1 = []
l_term_2 = []
l_term_3 = []
l_em_reduc = []
e = b.co2_intensity.value.values.reshape((N,S))
print("Computing decomposition")
for i in tqdm(range(len(sols)-1)):
    trade_baseline = trade[i].values.reshape((N,S,N))
    # trade_baseline = trade['baseline'].values.reshape((N,S,N))
    trade_cf = trade[i+1].values.reshape((N,S,N)) 
    
    term_1 = (X(trade_cf) - X(trade_baseline))/X(trade_baseline)

    term_2 = np.einsum('s,s,s->s',
                       epsilon_s(trade_baseline,e),
                       alpha_s(trade_cf)-alpha_s(trade_baseline),
                       1/alpha_s(trade_baseline))

    term_3 = np.einsum('is,is,is->is',
               epsilon_is(trade_baseline,e),
               alpha_is(trade_cf)-alpha_is(trade_baseline),
               1/alpha_is(trade_baseline))

    em_reduc = (np.einsum('isj,is->',
                         trade_cf,
                         e)-\
                np.einsum('isj,is->',
                          trade_baseline,
                          e))/np.einsum('isj,is->',
                                    trade_baseline,
                                    e)
                                        
    l_term_1.append(term_1)
    l_term_2.append(term_2)
    l_term_3.append(term_3)
    l_em_reduc.append(em_reduc)

l_term_1 = np.array(l_term_1)
l_term_2 = np.array(l_term_2)
l_term_3 = np.array(l_term_3)

l_em_reduc = np.array(l_em_reduc)
    
d_term = {
    'term_1':l_term_1,
    'term_2':l_term_2,
    'term_3':l_term_3,
          }

d_term_summed = {key: [v.sum() for v in l_term] for key, l_term in d_term.items()}

emiss_reduc_contrib = {}
for term, l_term in d_term_summed.items():
    emiss_reduc_contrib[term] = np.array([l_term[k]*sols[k].co2_prod.value.sum()/b.co2_prod.value.sum() 
                                 for k in range(len(l_term))])
    
l_em_incr = np.array([l_em_reduc[k]*sols[k].co2_prod.value.sum()/b.co2_prod.value.sum() 
             for k in range(len(l_em_reduc))])


term_labels = {
    'term_1':'Scale',
    'term_2':'Composition sectors',
    'term_3':'Composition countries'
    # 'em_reduc':l_em_reduc
          }

#%% Plot macro effects


print('Plotting welfare and GDP cost corresponding to a carbon tax')

carb_taxes = np.array([sol.params.carb_cost*1e6  for sol in sols])
norm_emissions = np.array([(sol.output.value.values*b.co2_intensity.value.values/1e6).sum()/baselines[y].co2_prod.value.sum() 
                           for sol in sols])
norm_emissions_real = np.array([sol.run.emissions/baselines[y].co2_prod.value.sum() for sol in sols])
norm_gdp = np.array([sol.va.value.sum()/baselines[y].va.value.sum() for sol in sols])
norm_real_income = np.array([sol.run.utility for sol in sols])
norm_total_output = np.array([sol.output.value.sum()/baselines[y].output.value.sum() for sol in sols])

fig, ax = plt.subplots(2,2,figsize=(16,12))

color = 'g'

# Upper left - Emissions
ax[0,0].plot(carb_taxes,norm_emissions,lw=4,color=color,label='Global emissions')
# ax[0,0].plot(carb_taxes,norm_emissions_real,lw=4,ls=':',color=color,label='Global emissions real')
ax[0,0].legend()
ax[0,0].set_xlabel('')
ax[0,0].tick_params(axis='x', which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax[0,0].set_xlim(0,1000)

y_100 = np.array(norm_emissions)[np.argmin(np.abs(carb_taxes-100))]
# y_0 = runs_low_carb_cost.iloc[0].emissions

ax[0,0].vlines(x=100,
            ymin=0,
            ymax=y_100,
            lw=3,
            ls = '--',
            color = color)

ax[0,0].hlines(y=y_100,
            xmin=0,
            xmax=100,
            lw=3,
            ls = '--',
            color = color)

ax[0,0].margins(y=0)

ax[0,0].annotate(str((100*(y_100-1)).round(0))+'%',
             xy=(100,y_100),
             xytext=(0,0),
             textcoords='offset points',color=color)

ax[0,0].set_ylim(norm_emissions.min(),norm_emissions.max()+0.05)

# Upper right - GDP
color = 'b'

ax[1,1].plot(carb_taxes,share_traded,lw=4)
ax[1,1].set_xlabel('')
ax[0,1].tick_params(axis='x', which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax[1,1].set_xlim(0,1000)
ax[1,1].legend(['Share of output traded (%)'])

# y_100 = share_traded[np.argmin(np.abs(carb_taxes-100))]
# 
# ax[0,1].vlines(x=100,
#             ymin=norm_gdp.min(),
#             ymax=y_100,
#             lw=3,
#             ls = '--',
#             color = color)

# ax[0,1].hlines(y=y_100,
#             xmin=0,
#             xmax=100,
#             lw=3,
#             ls = '--',
#             color = color)

ax[1,1].margins(y=0)

# ax[0,1].annotate(str(100*(1-y_100).round(1)),
#               xy=(100,y_100),
#               xytext=(0,0),
#               textcoords='offset points',color=color)

ax[1,1].set_ylim(10,20)

# Bottom left - Welfare
color = 'r'

ax[1,0].plot(carb_taxes,norm_real_income,lw=4,color='r')
ax[1,0].legend(['Real income'])
ax[1,0].set_xlabel('Carbon tax ($/ton of CO2eq.)')
ax[1,0].set_xlim(0,1000)
# ax[1,0].set_ylim(min(utility),1.001)

y_100 = norm_real_income[np.argmin(np.abs(carb_taxes-100))]
# y_0 = runs_low_carb_cost.iloc[0].emissions

ax[1,0].vlines(x=100,
            ymin=norm_real_income.min(),
            ymax=y_100,
            lw=3,
            ls = '--',
            color = color)

ax[1,0].hlines(y=y_100,
            xmin=0,
            xmax=100,
            lw=3,
            ls = '--',
            color = color)

ax[1,0].margins(y=0)

ax[1,0].set_ylim(norm_real_income.min(),1.005)

ax[1,0].annotate(str((100*(y_100-1)).round(1))+'%',
              xy=(100,y_100),
              xytext=(0,0),
              textcoords='offset points',color=color)

# Bottom right summary
color = 'k'

ax[0,1].plot(carb_taxes,norm_total_output,lw=4,color='k')
ax[0,1].legend(['World gross output'])
ax[1,1].set_xlabel('Carbon tax ($/ton of CO2eq.)')
ax[0,1].set_xlim(0,1000)
# ax[1,1].set_ylim(10,15)

y_100 = norm_total_output[np.argmin(np.abs(carb_taxes-100))]
# y_0 = runs_low_carb_cost.iloc[0].emissions

ax[0,1].vlines(x=100,
            ymin=norm_total_output.min(),
            ymax=y_100,
            lw=3,
            ls = '--',
            color = color)

ax[0,1].hlines(y=y_100,
            xmin=0,
            xmax=100,
            lw=3,
            ls = '--',
            color = color)

ax[0,1].margins(y=0)

ax[0,1].set_ylim(norm_total_output.min(),1.005)

ax[0,1].annotate(str((100*(y_100-1)).round(1))+'%',
              xy=(105,y_100),
              xytext=(0,0),
              textcoords='offset points',color=color)

plt.tight_layout()

if save or save_all:
    plt.savefig(save_path+'macro_effects.'+save_format,format=save_format)
plt.show()

#%% welfare changes wrt gdp per capita

welfare_change = sol.utility.copy()

labor = b.labor.set_index('country').rename_axis('col_country')['2018'].to_frame()
labor.columns = ['value']
palette = [sns.color_palette()[i] for i in [0,2,3]]
income_colors = {
    'Low-income' : sns.color_palette()[3],
    'Middle-income' : sns.color_palette()[0],
    'High-income' : sns.color_palette()[2],
                    }
country_list = b.country_list

welfare_change['gdp_p_c'] = (b.va.groupby('col_country').sum()/labor).rename_axis('country')
welfare_change = welfare_change.join(
    pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0)
    )
welfare_change['hat'] = (welfare_change['hat']-1)*100
try:
    welfare_change = welfare_change.drop(['TWN'],axis=0)
except:
    pass

fig, ax = plt.subplots(figsize=(16,12),constrained_layout=True)

colors = [income_colors[welfare_change.loc[country,'income_label']] for country in welfare_change.index]

ax.scatter(welfare_change['gdp_p_c'],
           welfare_change['hat'],
           color = colors,
           lw=5
           )

texts = [plt.text(welfare_change['gdp_p_c'].loc[country], 
                  welfare_change['hat'].loc[country], 
                  country,
                  size=20, 
                  # c = colors[i]) for i,country in enumerate(country_list)]
                  c = colors[i]) for i,country in enumerate(welfare_change.index)]

adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        )
            )
handles = [mpatches.Patch(color=income_colors[ind], label=ind) for ind in welfare_change['income_label'].drop_duplicates()]
ax.legend(handles=handles,
           fontsize=25,
           loc = 'lower right')
ax.set_xlabel('GDP per capita (millions US$)',fontsize = 25)
ax.set_ylabel('Real income change (%)',fontsize = 25)

ax.axhline(0, color='k')

plt.savefig(save_path+'welfare_change_by_gdp_by_country_by_income_group.'+save_format,format=save_format)
    
plt.show()

#%% plot decomposition

cumul_terms = {key: np.array([value[:i].sum()
                      for i in range(len(value))]) for key, value in emiss_reduc_contrib.items()}

fig,ax = plt.subplots(figsize = (16,12))

ax.stackplot(carb_taxes[1:],
              [term for term in cumul_terms.values()],
              labels=[term for term in cumul_terms.keys()])
ax.plot(carb_taxes[1:],[l_em_incr[:i].sum() for i in range(len(l_em_incr))], 
          label='Emissions',color='black'
          ,lw=3)

ax.legend(loc='lower left',fontsize = 20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_xlabel('Carbon tax',fontsize = 20)

if save or save_all:
    plt.savefig(save_path+'decomposition_stacked.'+save_format,format=save_format)

plt.show()

#%% plot decomposition diff labels

cumul_terms = {key: np.array([value[:i].sum()
                      for i in range(len(value))]) for key, value in emiss_reduc_contrib.items()}

fig,ax = plt.subplots(figsize = (16,16))

ax.stackplot(carb_taxes[1:],
              [term for term in cumul_terms.values()],
              labels=[term_labels[term] for term in cumul_terms.keys()])
ax.plot(carb_taxes[1:],[l_em_incr[:i].sum() for i in range(len(l_em_incr))], 
          label='Emissions',color='black'
          ,lw=3)
ax.legend(loc='lower left',fontsize = 30)
ax.tick_params(axis='both', which='major', labelsize=25)
ax.set_xlabel('Carbon tax',fontsize = 35)

if save or save_all:
    plt.savefig(save_path+'decomposition_stacked_diff_labels.'+save_format,format=save_format)
plt.show()

#%% plot decomposition normalized

cumul_terms = {key: np.array([value[:i].sum()
                      for i in range(len(value))]) for key, value in emiss_reduc_contrib.items()}

sum_terms = cumul_terms['term_1']+cumul_terms['term_2']+cumul_terms['term_3']

fig,ax = plt.subplots(figsize = (16,16))

ax.stackplot(carb_taxes[1:],
              [-term/sum_terms for term in cumul_terms.values()],
              labels=[term_labels[term] for term in cumul_terms.keys()])
offset = 0
for name,term in cumul_terms.items():
    loc = 50
    ax.text(carb_taxes[1:][loc], -(term[1:]/sum_terms[1:])[loc]/2+offset, term_labels[name]+' : '+str(((term[1:]/sum_terms[1:]).mean()*100).round(1))+'%',
            ha='center', va='center',color='white',fontsize = 35)
    offset = offset-(term[1:]/sum_terms[1:])[loc]

ax.tick_params(axis='both', which='major', labelsize=25)
ax.set_xlabel('Carbon tax',fontsize = 35)

if save or save_all:
    plt.savefig(save_path+'decomposition_stacked_norm.'+save_format,format=save_format)
plt.show()

#%% Sourcing term

trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[10].values.reshape((N,S,N)) 
sector_map = pd.read_csv(data_path+'industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')

fig, ax =plt.subplots(figsize=(12,8),constrained_layout=True)

colors = [sns.color_palette()[i] for i in [2,3,0]]
markers = ['x','o','^']
texts = []
data_base = sector_map.copy()

data_base['x'] = e_s(trade_baseline,e)
data_base['y'] = alpha_s(trade_cf)/alpha_s(trade_baseline)
ax.scatter(data_base.x,data_base.y*100-100)

texts_group = [plt.text(data_base.x.iloc[i], 
        data_base.y.iloc[i]*100-100, 
        industry,
        size=15,
        ) 
        for i,industry in enumerate(data_base.industry)] # For kernel density
texts = texts+texts_group

ax.set_xlabel('Emissions intensity (tCO2eq/Mio.$)',fontsize = 20)
ax.set_ylabel('Growth rate (%)',
              fontsize = 20,rotation=90)

plt.legend(fontsize = 20)
plt.xscale('log')

from matplotlib.ticker import StrMethodFormatter

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) 
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) 

adjust_text(texts, precision=1,
        # expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        # arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
        #                 )
        )
if save or save_all:
    plt.savefig(save_path+'growth_rate_em_intensity_term_2_with_labels.png',format='png')

plt.show()
