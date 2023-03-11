#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 21:41:15 2022

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
from tqdm import tqdm
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

#%% load data

save_path = 'presentation_material/uniform_elasticities/present_decomposition/'
data_path = 'data/'
results_path = 'results/'

try:
    os.mkdir(save_path)
except:
    pass


save_all = True
save_format = 'eps'

carb_cost_list = np.linspace(0,1e-3,1001)
# carb_cost_list = [1e-4]
# eta_path = ['elasticities_agg1.csv']
# sigma_path = ['elasticities_agg1.csv']
eta_path = ['uniform_elasticities_4.csv']
sigma_path = ['uniform_elasticities_4.csv']
taxed_countries_list = [None]
taxing_countries_list = [None]
taxed_sectors_list = [None]
specific_taxing_list = [None]
fair_tax_list = [False]
y  = 2018
year = str(y)
years = [y]
dir_num = 5

cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
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

b = baselines[y]
carb_taxes = np.array([sol.params.carb_cost*1e6  for sol in sols])
t_index = np.argmin(np.abs(carb_taxes-100))
sol = sols[t_index]
N = b.country_number
S = b.sector_number

sector_map = pd.read_csv(data_path+'industry_labels_after_agg_expl_wgroup.csv')
sector_map['sector'] = sector_map['ind_code'].str.replace('D','')
sector_map.set_index('sector',inplace=True)

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

def alpha_sj(trade):
    num = np.einsum('isj->sj',trade)
    denom = np.einsum('isj->s',trade)
    return np.einsum('sj,s->sj',num,1/denom)

def alpha_is(trade):
    num = np.einsum('isj->is',trade)
    denom = np.einsum('isj->s',trade)
    return np.einsum('is,s->is',num,1/denom)

def alpha_isj(trade):
    num = np.einsum('isj->isj',trade)
    denom = np.einsum('isj->sj',trade)
    return np.einsum('isj,sj->isj',num,1/denom)

def e_sj(trade,e):
    return np.einsum('isj,is->sj',alpha_isj(trade),e)

def e_is(trade,e):
    return np.einsum('isj,is->is',alpha_isj(trade),e)
    
def e_s(trade,e):
    return np.einsum('sj,sj->s',alpha_sj(trade),e_sj(trade,e))

def e_scal(trade,e):
    return np.einsum('s,s->',alpha_s(trade),e_s(trade,e))

def epsilon_s(trade,e):
    return np.einsum('s,s,,->s',
                     np.einsum('isj->s',trade),
                     e_s(trade,e),
                     1/np.einsum('isj->',trade),
                     1/e_scal(trade,e))

def epsilon_sj(trade,e):
    return np.einsum('sj,sj,,->sj',
                     np.einsum('isj->sj',trade),
                     e_sj(trade,e),
                     1/np.einsum('isj->',trade),
                     1/e_scal(trade,e))

def epsilon_is(trade,e):
    return np.einsum('is,is,,->is',
                     np.einsum('isj->is',trade),
                     e_is(trade,e),
                     1/np.einsum('isj->',trade),
                     1/e_scal(trade,e))

def epsilon_isj(trade,e):
    return np.einsum('isj,is,,->isj',
                     np.einsum('isj->isj',trade),
                     e,
                     1/np.einsum('isj->',trade),
                     1/e_scal(trade,e))



l_term_1 = []
l_term_2 = []
l_term_3 = []
l_term_4 = []
l_em_reduc = []
print("Computing decomposition")
for i in tqdm(range(len(sols)-1)):
    trade_baseline = trade[i].values.reshape((N,S,N))
    # trade_baseline = trade['baseline'].values.reshape((N,S,N))
    trade_cf = trade[i+1].values.reshape((N,S,N)) 
    e = b.co2_intensity.value.values.reshape((N,S))
    term_1 = (X(trade_cf) - X(trade_baseline))/X(trade_baseline)

    term_2 = np.einsum('s,s,s->s',
                       epsilon_s(trade_baseline,e),
                       alpha_s(trade_cf)-alpha_s(trade_baseline),
                       1/alpha_s(trade_baseline))

    term_3 = np.einsum('sj,sj,sj->sj',
               epsilon_sj(trade_baseline,e),
               alpha_sj(trade_cf)-alpha_sj(trade_baseline),
               1/alpha_sj(trade_baseline))

    term_4 = np.einsum('isj,isj,isj->isj',
               epsilon_isj(trade_baseline,e),
               alpha_isj(trade_cf)-alpha_isj(trade_baseline),
               np.divide(1, 
                         alpha_isj(trade_baseline), 
                         out = np.zeros_like(alpha_isj(trade_baseline)), 
                         where = alpha_isj(trade_baseline)!=0 )
               )
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
    l_term_4.append(term_4)
    l_em_reduc.append(em_reduc)

l_term_1 = np.array(l_term_1)
l_term_2 = np.array(l_term_2)
l_term_3 = np.array(l_term_3)
l_term_4 = np.array(l_term_4)
l_em_reduc = np.array(l_em_reduc)
    
d_term = {
    'term_1':l_term_1,
    'term_2':l_term_2,
    'term_3':l_term_3,
    'term_4':l_term_4,
    # 'em_reduc':l_em_reduc
          }

d_term_summed = {key: [v.sum() for v in l_term] for key, l_term in tqdm(d_term.items())}

emiss_reduc_contrib = {}
for term, l_term in d_term_summed.items():
    emiss_reduc_contrib[term] = np.array([l_term[k]*sols[k].co2_prod.value.sum()/b.co2_prod.value.sum() 
                                 for k in range(len(l_term))])
    
l_em_incr = np.array([l_em_reduc[k]*sols[k].co2_prod.value.sum()/b.co2_prod.value.sum() 
             for k in range(len(l_em_reduc))])


term_labels = {
    'term_1':'De-growth',
    'term_2':'Composition',
    'term_3':'Sourcing - between',
    'term_4':'Sourcing - within',
    # 'em_reduc':l_em_reduc
          }

#%% Plot macro effects

save = False

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

#%% plot decomposition

cumul_terms = {key: np.array([value[:i].sum()
                      for i in range(len(value))]) for key, value in emiss_reduc_contrib.items()}

fig,ax = plt.subplots(figsize = (14,10))

ax.stackplot(carb_taxes[1:],
              [term for term in cumul_terms.values()],
              labels=[term for term in cumul_terms.keys()])
ax.plot(carb_taxes[1:],[l_em_incr[:i].sum() for i in range(len(l_em_incr))], 
          label='Emissions',color='black'
          ,lw=3)
# ax.plot(carb_taxes,norm_emissions-1, 
#           label='Emissions real',color='y'
#           ,lw=3)
ax.legend(loc='lower left',fontsize = 20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_xlabel('Carbon tax',fontsize = 20)

# if save or save_all:
#     plt.savefig(save_path+'decomposition_stacked.'+save_format,format=save_format)
plt.show()

#%% plot decomposition diff labels

fig,ax = plt.subplots(figsize = (14,10))

ax.stackplot(carb_taxes[1:],
              [term for term in cumul_terms.values()],
              labels=[term_labels[term] for term in cumul_terms.keys()])
ax.plot(carb_taxes[1:],[l_em_incr[:i].sum() for i in range(len(l_em_incr))], 
          label='Emissions',color='black'
          ,lw=3)
ax.plot(carb_taxes,norm_emissions-1, 
          label='Emissions real',color='y'
          ,lw=3)
ax.legend(loc='lower left',fontsize = 20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_xlabel('Carbon tax',fontsize = 20)

if save or save_all:
    plt.savefig(save_path+'decomposition_stacked_diff_labels.'+save_format,format=save_format)
plt.show()

#%% plot decomposition normalized

cumul_terms = {key: np.array([value[:i].sum()
                      for i in range(len(value))]) for key, value in emiss_reduc_contrib.items()}

sum_terms = cumul_terms['term_1']+cumul_terms['term_2']+cumul_terms['term_3']+cumul_terms['term_4']

fig,ax = plt.subplots(figsize = (14,10))

ax.stackplot(carb_taxes[1:],
              [-term/sum_terms for term in cumul_terms.values()],
              labels=[term for term in cumul_terms.keys()])
# ax.plot(carb_taxes[1:],[l_em_incr[:i].sum() for i in range(len(l_em_incr))], 
#           label='Emissions',color='black'
#           ,lw=3)
# ax.plot(carb_taxes,norm_emissions-1, 
#           label='Emissions real',color='y'
#           ,lw=3)
# ax.legend(loc='lower left',fontsize = 20)
ax.legend(fontsize = 20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_xlabel('Carbon tax',fontsize = 20)

if save or save_all:
    plt.savefig(save_path+'decomposition_stacked_norm.'+save_format,format=save_format)
plt.show()

#%% plot composition term epsilon(alpha)

# trade_baseline = trade[100].values.reshape((N,S,N))
trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# ax2 = ax.twinx()
colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
texts = []
data_base = sector_map.copy()
data_base['y'] = epsilon_s(trade_baseline, e)
data_base['x'] = alpha_s(trade_baseline)
for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    data = data_base.loc[data_base['group_label'] == group_label]
    ax.scatter(data.x,data.y, label = group_label,color=colors[g])
    texts_group = [plt.text(data.x.iloc[i], 
            data.y.iloc[i], 
            industry,
            size=20,
            color=colors[g]) 
            for i,industry in enumerate(data.industry)]     # For kernel density
    texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_xlabel('Alpha',fontsize = 30)
ax.set_ylabel('Epsilon',fontsize = 30)
ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
# ax.set_xlim([1e-2,3e1])
plt.legend(fontsize = 25)
plt.xscale('log')
plt.yscale('log')


adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))
if save or save_all:
    plt.savefig(save_path+'term_2_eps_alpha.'+save_format,format=save_format)

plt.show()

#%% plot composition term d(alpha)/alpha func of epsilon

# trade_baseline = trade[100].values.reshape((N,S,N))
trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# ax2 = ax.twinx()
colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
texts = []
data_base = sector_map.copy()
data_base['x'] = epsilon_s(trade_baseline, e)
data_base['y'] = (alpha_s(trade_cf)-alpha_s(trade_baseline))/alpha_s(trade_baseline)
for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    data = data_base.loc[data_base['group_label'] == group_label]
    ax.scatter(data.x,data.y, label = group_label,color=colors[g])
    texts_group = [plt.text(data.x.iloc[i], 
            data.y.iloc[i], 
            industry,
            size=20,
            color=colors[g]) 
            for i,industry in enumerate(data.industry)]     # For kernel density
    texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_xlabel('epsilon',fontsize = 30)
ax.set_ylabel('d(alpha)/alpha',fontsize = 30)
# ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
# ax.set_xlim([1e-2,3e1])
plt.legend(fontsize = 25)
plt.xscale('log')
# plt.yscale('log')


# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))
if save or save_all:
    plt.savefig(save_path+'term_2_d_alpha_over_alpha_func_of_eps.'+save_format,format=save_format)

plt.show()

#%% plot composition term epsilon/alpha(alpha) schema

# trade_baseline = trade[100].values.reshape((N,S,N))
trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# ax2 = ax.twinx()
colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
texts = []
data_base = sector_map.copy()
data_base['y'] = epsilon_s(trade_baseline, e)/alpha_s(trade_baseline)
data_base['x'] = alpha_s(trade_baseline)
data_base = data_base.loc[data_base.industry.isin(['Basic metals','Construction'])]
# for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
data = data_base
ax.scatter(data.x,data.y)
texts_group = [plt.text(data.x.iloc[i], 
        data.y.iloc[i], 
        industry,
        size=20)
        # color=colors[g]) 
        for i,industry in enumerate(data.industry)]     # For kernel density
texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_xlabel('Alpha',fontsize = 30)
ax.set_ylabel('Epsilon/Alpha',fontsize = 30)

# plt.legend(fontsize = 25)
# plt.xscale('log')
# plt.yscale('log')

rect1 = patches.Rectangle((0, data.loc[data.industry == 'Basic metals'].iloc[0]['y']), 
                          data.loc[data.industry == 'Basic metals'].iloc[0]['x'], 
                          -data[data.industry == 'Basic metals'].iloc[0]['y'], 
                          alpha = 0.5,
                         # linewidth=1, 
                          edgecolor='r', 
                           facecolor=sns.color_palette()[0]
                         )

rect2 = patches.Rectangle((0, data.loc[data.industry == 'Construction'].iloc[0]['y']), 
                          data.loc[data.industry == 'Construction'].iloc[0]['x'], 
                          -data[data.industry == 'Construction'].iloc[0]['y'], 
                          alpha = 0.5,
                         # linewidth=1, 
                          edgecolor='r', 
                           facecolor=sns.color_palette()[1]
                         )

rect3 = patches.Rectangle((data.loc[data.industry == 'Basic metals'].iloc[0]['x'], data.loc[data.industry == 'Basic metals'].iloc[0]['y']), 
                          -0.001, 
                          -data[data.industry == 'Basic metals'].iloc[0]['y'], 
                          alpha = 0.5,
                         # linewidth=1, 
                          edgecolor='r', 
                          hatch='/',
                           facecolor=sns.color_palette()[0],
                           label = 'Emissions reduction from Basic metals'
                         )

rect4 = patches.Rectangle((data.loc[data.industry == 'Construction'].iloc[0]['x'], data.loc[data.industry == 'Construction'].iloc[0]['y']), 
                          0.001, 
                          -data[data.industry == 'Construction'].iloc[0]['y'], 
                          alpha = 0.5,
                         # linewidth=1, 
                          edgecolor='r', 
                          hatch='/',
                           facecolor=sns.color_palette()[1],
                           label = 'Additional emissions from Construction'
                         )

ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
plt.legend(fontsize = 25)

plt.text(data.loc[data.industry == 'Basic metals'].iloc[0]['x']-0.0005,
         -0.1,'d(alpha)',
         horizontalalignment='center')
plt.text(data.loc[data.industry == 'Construction'].iloc[0]['x']+0.0005,
         -0.1,'d(alpha)',
         horizontalalignment='center')

# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))
if save or save_all:
    plt.savefig(save_path+'term_2_eps_alpha_schema.pdf',format='pdf')

plt.show()

#%% plot composition term function of epsilon / alpha

# trade_baseline = trade[100].values.reshape((N,S,N))
trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# ax2 = ax.twinx()
colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
texts = []
data_base = sector_map.copy()
data_base['x'] = epsilon_s(trade_baseline, e)/alpha_s(trade_baseline)
data_base['y'] = alpha_s(trade_cf)-alpha_s(trade_baseline)
for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    data = data_base.loc[data_base['group_label'] == group_label]
    ax.scatter(data.x,data.y, label = group_label,color=colors[g])
    texts_group = [plt.text(data.x.iloc[i], 
            data.y.iloc[i], 
            industry,
            size=20,
            color=colors[g]) 
            for i,industry in enumerate(data.industry)]     # For kernel density
    texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_xlabel('Epsilon / alpha',fontsize = 30)
ax.set_ylabel('d(alpha)',fontsize = 30)
# ax.set_xlim([1e-2,3e1])
plt.legend(fontsize = 25)
plt.xscale('log')


adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))
if save or save_all:
    plt.savefig(save_path+'term_2_eps_over_alpha.'+save_format,format=save_format)

plt.show()

#%% plot composition term function of alpha

# trade_baseline = trade[100].values.reshape((N,S,N))
trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# ax2 = ax.twinx()
colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
texts = []
data_base = sector_map.copy()
data_base['x'] = alpha_s(trade_baseline)
data_base['y'] = alpha_s(trade_cf)-alpha_s(trade_baseline)
for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    data = data_base.loc[data_base['group_label'] == group_label]
    ax.scatter(data.x,data.y, label = group_label,color=colors[g])
    texts_group = [plt.text(data.x.iloc[i], 
            data.y.iloc[i], 
            industry,
            size=20,
            color=colors[g]) 
            for i,industry in enumerate(data.industry)]     # For kernel density
    texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_xlabel('Alpha',fontsize = 30)
ax.set_ylabel('d(alpha)',fontsize = 30)
# ax.set_xlim([1e-2,3e1])
plt.legend(fontsize = 25)
plt.xscale('log')
plt.title('Term 2', fontsize = 30)

adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))
if save or save_all:
    plt.savefig(save_path+'term_2_alpha.'+save_format,format=save_format)

plt.show()

#%% plot composition term function of alpha cons

# trade_baseline = trade[100].values.reshape((N,S,N))
trade_baseline = b.cons.value.values.reshape((N,S,N))
trade_cf = sol.cons.value.values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# ax2 = ax.twinx()
colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
texts = []
data_base = sector_map.copy()
data_base['x'] = alpha_s(trade_baseline)
data_base['y'] = alpha_s(trade_cf)-alpha_s(trade_baseline)
for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    data = data_base.loc[data_base['group_label'] == group_label]
    ax.scatter(data.x,data.y, label = group_label,color=colors[g])
    texts_group = [plt.text(data.x.iloc[i], 
            data.y.iloc[i], 
            industry,
            size=20,
            color=colors[g]) 
            for i,industry in enumerate(data.industry)]     # For kernel density
    texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_xlabel('Alpha',fontsize = 30)
ax.set_ylabel('d(alpha)',fontsize = 30)
# ax.set_xlim([1e-2,3e1])
plt.legend(fontsize = 25)
plt.xscale('log')
plt.title('Term 2 final consumption only', fontsize = 30)

adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))
if save or save_all:
    plt.savefig(save_path+'term_2_alpha_cons.'+save_format,format=save_format)

plt.show()

#%% plot composition term function of alpha intermediate input

# trade_baseline = trade[100].values.reshape((N,S,N))
trade_baseline = b.iot.value.values.reshape((N,S,N,S)).sum(axis=-1)
trade_cf = sol.iot.value.values.reshape((N,S,N,S)).sum(axis=-1)

fig, ax = plt.subplots(figsize=(25,15))
# ax2 = ax.twinx()
colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
texts = []
data_base = sector_map.copy()
data_base['x'] = alpha_s(trade_baseline)
data_base['y'] = alpha_s(trade_cf)-alpha_s(trade_baseline)
for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    data = data_base.loc[data_base['group_label'] == group_label]
    ax.scatter(data.x,data.y, label = group_label,color=colors[g])
    texts_group = [plt.text(data.x.iloc[i], 
            data.y.iloc[i], 
            industry,
            size=20,
            color=colors[g]) 
            for i,industry in enumerate(data.industry)]     # For kernel density
    texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_xlabel('Alpha',fontsize = 30)
ax.set_ylabel('d(alpha)',fontsize = 30)
# ax.set_xlim([1e-2,3e1])
plt.legend(fontsize = 25)
plt.xscale('log')
plt.title('Term 2 intermediate input only', fontsize = 30)

adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))
if save or save_all:
    plt.savefig(save_path+'term_2_alpha_iot.'+save_format,format=save_format)

plt.show()

#%% plot composition term

trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
ax2 = ax.twinx()
colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
texts = []
data_base = sector_map.copy()
data_base['y'] = np.einsum('s,s,s->s',
                   epsilon_s(trade_baseline,e),
                   alpha_s(trade_cf)-alpha_s(trade_baseline),
                   1/alpha_s(trade_baseline))
data_base['y2'] = alpha_s(trade_cf)-alpha_s(trade_baseline)
ax.bar(0,0,color='grey',label = 'change in emissions\nassociated')
for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    data = data_base.loc[data_base['group_label'] == group_label]
    bars = ax.bar(data.industry,data.y, label = group_label,color=colors[g])
    ax.bar_label(bars,
                 labels=data.industry,
                 rotation=90,
                  label_type = 'edge',
                  padding=2,
                  # color=colors[g],
                  zorder=10)
    ax2.scatter(data.industry,data.y2,color=colors[g],
                edgecolors='k')
    # texts_group = [plt.text(data.x.iloc[i], 
    #         data.y.iloc[i], 
    #         industry,
    #         size=20,
    #         color=colors[g]) 
    #         for i,industry in enumerate(data.industry)]     # For kernel density
    # texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
ax2.grid([])
ax.scatter([],[],color='grey',label = 'change in share')

# ax.set_xlabel('Sector',fontsize = 30)
ax.set_ylabel('Change in emissions associated',fontsize = 30)
ax2.set_ylabel('d(alpha)',fontsize = 30)
ax.set_ylim([-0.075,0.075])
ax2.set_ylim([-0.0075,0.0075])
ax.legend(fontsize = 20)
# plt.xscale('log')
plt.title('Term 2', fontsize = 30)
ax.set_xticklabels(['']
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)


# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))
# if save or save_all:
#     plt.savefig(save_path+'term_2.'+save_format,format=save_format)

plt.show()

#%% plot term 3 epsilon(alpha) all terms

trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
# texts = []
countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
                                                              b.sector_list],
                                                            names = ['country','sector'])).reset_index()
# data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
# data_base = countries.copy()

data_base['y'] = np.einsum('sj->sj',epsilon_sj(trade_baseline,e)).ravel()
data_base['x'] = np.einsum('sj->sj',alpha_sj(trade_baseline)).ravel()
# for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
# data = data_base.loc[data_base['group_label'] == group_label]
data= data_base
ax.scatter(data.x,data.y
           # , label = group_label,color=colors[g]
           )
# texts_group = [plt.text(data.x.iloc[i], 
#         data.y.iloc[i], 
#         data.iloc[i].country+','+data.iloc[i].sector,
#         size=10,
#         # color=colors[g]
#         ) 
#         for i in data_base.index]     # For kernel density
# texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_xlabel('Alpha',fontsize = 30)
ax.set_ylabel('Epsilon',fontsize = 30)
ax.plot([0,data_base.x.max()],[0,data_base.x.max()/S],ls='--',color='k')
# ax.set_xlim([1e-2,3e1])
# plt.legend(fontsize = 25)
plt.xscale('log')
plt.yscale('log')
plt.title('Term 3')


# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))
if save or save_all:
    plt.savefig(save_path+'term_3_eps_alpha.'+save_format,format=save_format)

plt.show()

#%% plot term 3 function of epsilon over alpha all terms

trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
# texts = []
countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
                                                              b.sector_list],
                                                            names = ['country','sector'])).reset_index()
# data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
# data_base = countries.copy()

data_base['x'] = np.einsum('sj,sj->sj',
                           epsilon_sj(trade_baseline,e),
                           1/alpha_sj(trade_baseline)).ravel()
data_base['y'] = np.einsum('sj->sj',alpha_sj(trade_cf)-alpha_sj(trade_baseline)).ravel()
# for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
# data = data_base.loc[data_base['group_label'] == group_label]
data= data_base
ax.scatter(data.x,data.y
           # , label = group_label,color=colors[g]
           )
# texts_group = [plt.text(data.x.iloc[i], 
#         data.y.iloc[i], 
#         data.iloc[i].country+','+data.iloc[i].sector,
#         size=10,
#         # color=colors[g]
#         ) 
#         for i in data_base.index]     # For kernel density
# texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_xlabel('Epsilon/Alpha',fontsize = 30)
ax.set_ylabel('d(alpha)',fontsize = 30)
# ax.plot([0,data_base.x.max()],[0,data_base.x.max()/S/N],ls='--',color='k')
# ax.set_xlim([1e-2,3e1])
# plt.legend(fontsize = 25)
plt.xscale('log')
# plt.yscale('log')
plt.title('Term 3')


# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))
if save or save_all:
    plt.savefig(save_path+'term_3_eps_over_alpha.'+save_format,format=save_format)

plt.show()

#%% plot term 3 function of epsilon over alpha by sector all terms

trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 


# colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
# texts = []
countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
                                                              b.sector_list],
                                                            names = ['country','sector'])).reset_index()
# data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
# data_base = countries.copy()

data_base['x'] = np.einsum('sj,sj->sj',
                           epsilon_sj(trade_baseline,e),
                           1/alpha_sj(trade_baseline)).ravel()
data_base['y'] = np.einsum('sj->sj',alpha_sj(trade_cf)-alpha_sj(trade_baseline)).ravel()
data_base = pd.merge(data_base,sector_map.reset_index(),on='sector')
# for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
# data = data_base.loc[data_base['group_label'] == group_label]
for sector in data_base.sector.drop_duplicates():   
    fig, ax = plt.subplots(figsize=(25,15))
    data= data_base.loc[data_base.sector == sector]
    ax.scatter(data.x,data.y
           # , label = group_label,color=colors[g]
           )
    texts_group = [plt.text(data.x.loc[i], 
            data.y.loc[i], 
            data.loc[i].country,
            size=20,
            # color=colors[g]
            ) 
            for i in data.index]     # For kernel density
# texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
    ax.set_xlabel('Epsilon/Alpha',fontsize = 30)
    ax.set_ylabel('d(alpha)',fontsize = 30)
# ax.plot([0,data_base.x.max()],[0,data_base.x.max()/S/N],ls='--',color='k')
# ax.set_xlim([1e-2,3e1])
# plt.legend(fontsize = 25)
    plt.xscale('log')
    # plt.yscale('log')
    plt.title('Term 3 '+sector_map.loc[sector].industry)


# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))
# if save or save_all:
#     plt.savefig(save_path+'term_3_eps_over_alpha.'+save_format,format=save_format)

    plt.show()

#%% plot term 3 epsilon(alpha) j specific

trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
# texts = []
countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
# data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
#                                                               b.sector_list],
#                                                             names = ['country','sector'])).reset_index()
# data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
data_base = countries.copy()

data_base['y'] = np.einsum('sj->j',epsilon_sj(trade_baseline,e)).ravel()
data_base['x'] = np.einsum('sj->j',alpha_sj(trade_baseline)).ravel()/S
# for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
# data = data_base.loc[data_base['group_label'] == group_label]
data= data_base
ax.scatter(data.x,data.y
           # , label = group_label,color=colors[g]
           )
texts = [plt.text(data.x.loc[i], 
        data.y.loc[i], 
        data.loc[i].country_name,
        size=15,
        # color=colors[g]
        ) 
        for i in data_base.index]     # For kernel density
# texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_xlabel('Alpha',fontsize = 30)
ax.set_ylabel('Epsilon',fontsize = 30)
ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
# ax.set_xlim([1e-2,3e1])
# plt.legend(fontsize = 25)
plt.xscale('log')
plt.yscale('log')
plt.title('Term 3')

adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))
if save or save_all:
    plt.savefig(save_path+'term_3_eps_alpha_j.'+save_format,format=save_format)

plt.show()

#%% plot term 3 epsilon(alpha) cons only j specific

trade_baseline = b.cons.value.values.reshape((N,S,N))
trade_cf = sol.cons.value.values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
# texts = []
countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
# data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
#                                                               b.sector_list],
#                                                             names = ['country','sector'])).reset_index()
# data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
data_base = countries.copy()

data_base['y'] = np.einsum('sj->j',epsilon_sj(trade_baseline,e)).ravel()
data_base['x'] = np.einsum('sj->j',alpha_sj(trade_baseline)).ravel()/S
# for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
# data = data_base.loc[data_base['group_label'] == group_label]
data= data_base
ax.scatter(data.x,data.y
           # , label = group_label,color=colors[g]
           )
texts = [plt.text(data.x.loc[i], 
        data.y.loc[i], 
        data.loc[i].country_name,
        size=15,
        # color=colors[g]
        ) 
        for i in data_base.index]     # For kernel density
# texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_xlabel('Alpha',fontsize = 30)
ax.set_ylabel('Epsilon',fontsize = 30)
ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
# ax.set_xlim([1e-2,3e1])
# plt.legend(fontsize = 25)
plt.xscale('log')
plt.yscale('log')
plt.title('Term 3 cons only')


adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))
if save or save_all:
    plt.savefig(save_path+'term_3_eps_alpha_j_cons_only.'+save_format,format=save_format)

plt.show()

#%% plot term 3
trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
# texts = []
ax2 = ax.twinx()
countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
# data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
#                                                              b.sector_list],
#                                                             names = ['country','sector']))
# data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
data_base = countries.copy()
data_base['y'] = np.einsum('sj,sj,sj->j',
           epsilon_sj(trade_baseline,e),
           alpha_sj(trade_cf)-alpha_sj(trade_baseline),
           1/alpha_sj(trade_baseline))
data_base['y2'] = np.einsum('sj->j',alpha_sj(trade_cf)-alpha_sj(trade_baseline)).ravel()
data = data_base
bars = ax.bar(data.country_name,data.y)
ax.bar_label(bars,
             labels=data.country_name,
             rotation=90,
              label_type = 'edge',
              padding=2,
              # color=colors[g],
              zorder=10)
ax2.scatter(data.country_name,data.y2,edgecolors='k',color='g')
ax2.grid([])
ax2.set_ylim([-0.25,0.25])
ax.set_ylim([-0.025,0.025])
ax2.scatter([],[],color='grey',label = 'change in share')
ax2.bar(0,0, label = 'change in emissions\nassociated', color = 'grey')
ax.set_xticklabels(['']
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)
ax.set_ylabel('Change in emissions associated')
ax2.set_ylabel('Change in share')
plt.title('Term 3')
plt.legend(fontsize = 20)

if save or save_all:
    plt.savefig(save_path+'term_3_j.'+save_format,format=save_format)

plt.show()

#%% plot term 3 d(alpha) function of epsilon/alpha j specific

trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
# texts = []
countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
# data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
#                                                               b.sector_list],
#                                                             names = ['country','sector'])).reset_index()
# data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
data_base = countries.copy()

data_base['x'] = np.einsum('sj,sj->j',epsilon_sj(trade_baseline,e),1/alpha_sj(trade_baseline)).ravel()
data_base['y'] = np.einsum('sj->j',alpha_sj(trade_cf)-alpha_sj(trade_baseline)).ravel()
# for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
# data = data_base.loc[data_base['group_label'] == group_label]
data= data_base
ax.scatter(data.x,data.y
           # , label = group_label,color=colors[g]
           )
texts = [plt.text(data.x.loc[i], 
        data.y.loc[i], 
        data.loc[i].country_name,
        size=15,
        # color=colors[g]
        ) 
        for i in data_base.index]     # For kernel density
# texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_ylabel('d(alpha)',fontsize = 30)
ax.set_xlabel('Epsilon/alpha',fontsize = 30)
# ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
# ax.set_xlim([1e-2,3e1])
# plt.legend(fontsize = 25)
plt.xscale('log')
# plt.yscale('log')
plt.title('Term 3')

adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))
if save or save_all:
    plt.savefig(save_path+'term_3_eps_over_alpha_j.'+save_format,format=save_format)

plt.show()

#%% plot term 3 d(alpha) function of epsilon/alpha j specific cons only

trade_baseline = b.cons.value.values.reshape((N,S,N))
trade_cf = sol.cons.value.values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
# texts = []
countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
# data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
#                                                               b.sector_list],
#                                                             names = ['country','sector'])).reset_index()
# data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
data_base = countries.copy()

data_base['x'] = np.einsum('sj,sj->j',epsilon_sj(trade_baseline,e),1/alpha_sj(trade_baseline)).ravel()
data_base['y'] = np.einsum('sj->j',alpha_sj(trade_cf)-alpha_sj(trade_baseline)).ravel()
# for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
# data = data_base.loc[data_base['group_label'] == group_label]
data= data_base
ax.scatter(data.x,data.y
           # , label = group_label,color=colors[g]
           )
texts = [plt.text(data.x.loc[i], 
        data.y.loc[i], 
        data.loc[i].country_name,
        size=15,
        # color=colors[g]
        ) 
        for i in data_base.index]     # For kernel density
# texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_ylabel('d(alpha)',fontsize = 30)
ax.set_xlabel('Epsilon/alpha',fontsize = 30)
# ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
# ax.set_xlim([1e-2,3e1])
# plt.legend(fontsize = 25)
plt.xscale('log')
# plt.yscale('log')
plt.title('Term 3 cons only')

adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))
if save or save_all:
    plt.savefig(save_path+'term_3_eps_over_alpha_j_cons_only.'+save_format,format=save_format)

plt.show()

#%% plot term 3 d(alpha) function of alpha j specific

trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
# texts = []
countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
# data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
#                                                               b.sector_list],
#                                                             names = ['country','sector'])).reset_index()
# data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
data_base = countries.copy()

data_base['x'] = np.einsum('sj->j',alpha_sj(trade_baseline)).ravel()
data_base['y'] = np.einsum('sj->j',alpha_sj(trade_cf)-alpha_sj(trade_baseline)).ravel()
# for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
# data = data_base.loc[data_base['group_label'] == group_label]
data= data_base
ax.scatter(data.x,data.y
           # , label = group_label,color=colors[g]
           )
texts = [plt.text(data.x.loc[i], 
        data.y.loc[i], 
        data.loc[i].country_name,
        size=15,
        # color=colors[g]
        ) 
        for i in data_base.index]     # For kernel density
# texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_ylabel('d(alpha)',fontsize = 30)
ax.set_xlabel('Alpha',fontsize = 30)
# ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
# ax.set_xlim([1e-2,3e1])
# plt.legend(fontsize = 25)
plt.xscale('log')
# plt.yscale('log')
plt.title('Term 3')

adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))
if save or save_all:
    plt.savefig(save_path+'term_3_alpha_j.'+save_format,format=save_format)

plt.show()

#%% plot term 4 epsilon(alpha) all terms

trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
# texts = []
countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
                                                              b.sector_list,
                                                              b.country_list],
                                                            names = ['row_country','row_sector','col_country'])).reset_index()
# data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
# data_base = countries.copy()

data_base['y'] = np.einsum('isj->isj',epsilon_isj(trade_baseline,e)).ravel()
data_base['x'] = np.einsum('isj->isj',alpha_isj(trade_baseline)).ravel()
# for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
# data = data_base.loc[data_base['group_label'] == group_label]
data= data_base
ax.scatter(data.x,data.y
           # , label = group_label,color=colors[g]
           )
# texts_group = [plt.text(data.x.iloc[i], 
#         data.y.iloc[i], 
#         data.iloc[i].row_country+','+data.iloc[i].row_sector+','+data.iloc[i].row_sector,
#         size=10,
#         # color=colors[g]
#         ) 
#         for i in data_base.index]     # For kernel density
# texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_xlabel('Alpha',fontsize = 30)
ax.set_ylabel('Epsilon',fontsize = 30)
ax.plot([0,data_base.x.max()],[0,data_base.x.max()/S/N],ls='--',color='k')
# ax.set_xlim([1e-2,3e1])
# plt.legend(fontsize = 25)
plt.xscale('log')
plt.yscale('log')
plt.title('Term 4')


# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))
if save or save_all:
    plt.savefig(save_path+'term_4_eps_alpha.'+save_format,format=save_format)

plt.show()

#%% plot term 4 function of epsilon/alpha all terms

trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
# texts = []
countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
                                                              b.sector_list,
                                                              b.country_list],
                                                            names = ['row_country','row_sector','col_country'])).reset_index()
# data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
# data_base = countries.copy()

data_base['x'] = np.einsum('isj,isj->isj',
                           epsilon_isj(trade_baseline,e),
                           np.divide(1, 
                                     alpha_isj(trade_baseline), 
                                     out = np.zeros_like(alpha_isj(trade_baseline)), 
                                     where = alpha_isj(trade_baseline)!=0 )
                           ).ravel()
data_base['y'] = np.einsum('isj->isj',alpha_isj(trade_cf)-alpha_isj(trade_baseline)).ravel()
# for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
# data = data_base.loc[data_base['group_label'] == group_label]
data= data_base
ax.scatter(data.x,data.y
           # , label = group_label,color=colors[g]
           )
# texts_group = [plt.text(data.x.iloc[i], 
#         data.y.iloc[i], 
#         data.iloc[i].row_country+','+data.iloc[i].row_sector+','+data.iloc[i].row_sector,
#         size=10,
#         # color=colors[g]
#         ) 
#         for i in data_base.index]     # For kernel density
# texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_xlabel('Epsilon/Alpha',fontsize = 30)
ax.set_ylabel('d(alpha)',fontsize = 30)
ax.plot([0,data_base.x.max()],[0,data_base.x.max()/S/N],ls='--',color='k')
# ax.set_xlim([1e-2,3e1])
# plt.legend(fontsize = 25)
plt.xscale('log')
# plt.yscale('log')
plt.title('Term 4')


# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))
if save or save_all:
    plt.savefig(save_path+'term_4_eps_over_alpha_all_terms.'+save_format,format=save_format)

plt.show()

#%% plot term 4 function of epsilon/alpha all terms

trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 


# colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
# texts = []
countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
                                                              b.sector_list,
                                                              b.country_list],
                                                            names = ['row_country','row_sector','col_country'])).reset_index()
# data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
# data_base = countries.copy()

data_base['x'] = np.einsum('isj,isj->isj',
                           epsilon_isj(trade_baseline,e),
                           np.divide(1, 
                                     alpha_isj(trade_baseline), 
                                     out = np.zeros_like(alpha_isj(trade_baseline)), 
                                     where = alpha_isj(trade_baseline)!=0 )
                           ).ravel()
data_base['y'] = np.einsum('isj->isj',alpha_isj(trade_cf)-alpha_isj(trade_baseline)).ravel()
# for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
# data = data_base.loc[data_base['group_label'] == group_label]
# data_base = pd.merge
for sector in data_base.row_sector.drop_duplicates():
    fig, ax = plt.subplots(figsize=(25,15))
    data= data_base.loc[data_base.row_sector == sector]
    ax.scatter(data.x,data.y
               , label = sector
               )
    plt.xscale('log')
    plt.show()
    
# texts_group = [plt.text(data.x.iloc[i], 
#         data.y.iloc[i], 
#         data.iloc[i].row_country+','+data.iloc[i].row_sector+','+data.iloc[i].row_sector,
#         size=10,
#         # color=colors[g]
#         ) 
#         for i in data_base.index]     # For kernel density
# texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_xlabel('Epsilon/Alpha',fontsize = 30)
ax.set_ylabel('d(alpha)',fontsize = 30)
ax.plot([0,data_base.x.max()],[0,data_base.x.max()/S/N],ls='--',color='k')
# ax.set_xlim([1e-2,3e1])
plt.legend(fontsize = 25)
plt.xscale('log')
# plt.yscale('log')
plt.title('Term 4')


# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))
# if save or save_all:
#     plt.savefig(save_path+'term_4_eps_over_alpha_all_terms.'+save_format,format=save_format)

plt.show()


#%% plot term 4 epsilon(alpha) i specific

trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
# texts = []
countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
# data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
#                                                               b.sector_list],
#                                                             names = ['country','sector'])).reset_index()
# data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
data_base = countries.copy()

data_base['y'] = np.einsum('isj->i',epsilon_isj(trade_baseline,e)).ravel()
data_base['x'] = np.einsum('isj->i',alpha_isj(trade_baseline)).ravel()/N/S
# for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
# data = data_base.loc[data_base['group_label'] == group_label]
data= data_base
ax.scatter(data.x,data.y
           # , label = group_label,color=colors[g]
           )
texts = [plt.text(data.x.loc[i], 
        data.y.loc[i], 
        data.loc[i].country_name,
        size=15,
        # color=colors[g]
        ) 
        for i in data_base.index]     # For kernel density
# texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_xlabel('Alpha',fontsize = 30)
ax.set_ylabel('Epsilon',fontsize = 30)
ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
# ax.set_xlim([1e-2,3e1])
# plt.legend(fontsize = 25)
plt.xscale('log')
plt.yscale('log')
plt.title('Term 4')

# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))
if save or save_all:
    plt.savefig(save_path+'term_4_eps_alpha_i.'+save_format,format=save_format)

plt.show()

#%% plot term 4 epsilon(alpha) cons only i specific

trade_baseline = b.cons.value.values.reshape((N,S,N))
trade_cf = sol.cons.value.values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
# texts = []
countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
# data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
#                                                               b.sector_list],
#                                                             names = ['country','sector'])).reset_index()
# data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
data_base = countries.copy()

data_base['y'] = np.einsum('isj->i',epsilon_isj(trade_baseline,e)).ravel()
data_base['x'] = np.einsum('isj->i',alpha_isj(trade_baseline)).ravel()/N/S
# for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
# data = data_base.loc[data_base['group_label'] == group_label]
data= data_base
ax.scatter(data.x,data.y
           # , label = group_label,color=colors[g]
           )
texts = [plt.text(data.x.loc[i], 
        data.y.loc[i], 
        data.loc[i].country_name,
        size=15,
        # color=colors[g]
        ) 
        for i in data_base.index]     # For kernel density
# texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_xlabel('Alpha',fontsize = 30)
ax.set_ylabel('Epsilon',fontsize = 30)
ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
# ax.set_xlim([1e-2,3e1])
# plt.legend(fontsize = 25)
plt.xscale('log')
plt.yscale('log')
plt.title('Term 4 cons only')


adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))
if save or save_all:
    plt.savefig(save_path+'term_4_eps_alpha_i_cons_only.'+save_format,format=save_format)

plt.show()

#%% plot term 4
trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
# texts = []
ax2 = ax.twinx()
countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
# data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
#                                                              b.sector_list],
#                                                             names = ['country','sector']))
# data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
data_base = countries.copy()
data_base['y'] = np.einsum('isj,isj,isj->i',
           epsilon_isj(trade_baseline,e),
           alpha_isj(trade_cf)-alpha_isj(trade_baseline),
           np.divide(1, 
                     alpha_isj(trade_baseline), 
                     out = np.zeros_like(alpha_isj(trade_baseline)), 
                     where = alpha_isj(trade_baseline)!=0 ))
data_base['y2'] = np.einsum('isj->i',alpha_isj(trade_cf)-alpha_isj(trade_baseline)).ravel()
data = data_base
bars = ax.bar(data.country_name,data.y)
ax.bar_label(bars,
             labels=data.country_name,
             rotation=90,
              label_type = 'edge',
              padding=2,
              # color=colors[g],
              zorder=10)
ax2.scatter(data.country_name,data.y2,edgecolors='k',color='g')
ax2.grid([])
ax2.set_ylim([-5,5])
ax.set_ylim([-0.008,0.008])
ax2.scatter([],[],color='grey',label = 'change in share')
ax2.bar(0,0, label = 'change in emissions\nassociated', color = 'grey')
ax.set_xticklabels(['']
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)
ax.set_ylabel('Change in emissions associated')
ax2.set_ylabel('Change in share')
plt.title('Term 4')
plt.legend(fontsize = 20)

if save or save_all:
    plt.savefig(save_path+'term_4_i.'+save_format,format=save_format)

plt.show()

#%% plot term 4 d(alpha) function of epsilon/alpha i specific

trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
# texts = []
countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
# data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
#                                                               b.sector_list],
#                                                             names = ['country','sector'])).reset_index()
# data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
data_base = countries.copy()

data_base['x'] = np.einsum('isj,isj->i',epsilon_isj(trade_baseline,e),
                           np.divide(1, 
                                     alpha_isj(trade_baseline), 
                                     out = np.zeros_like(alpha_isj(trade_baseline)), 
                                     where = alpha_isj(trade_baseline)!=0 )).ravel()
data_base['y'] = np.einsum('isj->i',alpha_isj(trade_cf)-alpha_isj(trade_baseline)).ravel()
# for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
# data = data_base.loc[data_base['group_label'] == group_label]
data= data_base
ax.scatter(data.x,data.y
           # , label = group_label,color=colors[g]
           )
texts = [plt.text(data.x.loc[i], 
        data.y.loc[i], 
        data.loc[i].country_name,
        size=15,
        # color=colors[g]
        ) 
        for i in data_base.index]     # For kernel density
# texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_ylabel('d(alpha)',fontsize = 30)
ax.set_xlabel('Epsilon/alpha',fontsize = 30)
# ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
# ax.set_xlim([1e-2,3e1])
# plt.legend(fontsize = 25)
plt.xscale('log')
# plt.yscale('log')
plt.title('Term 4')

adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))
if save or save_all:
    plt.savefig(save_path+'term_4_eps_over_alpha_i.'+save_format,format=save_format)

plt.show()

#%% plot term 4 d(alpha) function of epsilon/alpha i specific cons only

trade_baseline = b.cons.value.values.reshape((N,S,N))
trade_cf = sol.cons.value.values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
# texts = []
countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
# data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
#                                                               b.sector_list],
#                                                             names = ['country','sector'])).reset_index()
# data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
data_base = countries.copy()

data_base['x'] = np.einsum('isj,isj->i',epsilon_isj(trade_baseline,e),
                           np.divide(1, 
                                     alpha_isj(trade_baseline), 
                                     out = np.zeros_like(alpha_isj(trade_baseline)), 
                                     where = alpha_isj(trade_baseline)!=0 )).ravel()
data_base['y'] = np.einsum('isj->i',alpha_isj(trade_cf)-alpha_isj(trade_baseline)).ravel()
# for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
# data = data_base.loc[data_base['group_label'] == group_label]
data= data_base
ax.scatter(data.x,data.y
           # , label = group_label,color=colors[g]
           )
texts = [plt.text(data.x.loc[i], 
        data.y.loc[i], 
        data.loc[i].country_name,
        size=15,
        # color=colors[g]
        ) 
        for i in data_base.index]     # For kernel density
# texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_ylabel('d(alpha)',fontsize = 30)
ax.set_xlabel('Epsilon/alpha',fontsize = 30)
# ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
# ax.set_xlim([1e-2,3e1])
# plt.legend(fontsize = 25)
plt.xscale('log')
# plt.yscale('log')
plt.title('Term 4 cons only')

adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))
if save or save_all:
    plt.savefig(save_path+'term_4_eps_over_alpha_i_cons_only.'+save_format,format=save_format)

plt.show()

#%% plot term 4 d(alpha) function of alpha i specific

trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

fig, ax = plt.subplots(figsize=(25,15))
# colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
# texts = []
countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
# data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
#                                                               b.sector_list],
#                                                             names = ['country','sector'])).reset_index()
# data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
data_base = countries.copy()

data_base['x'] = np.einsum('isj->i',alpha_isj(trade_baseline)).ravel()
data_base['y'] = np.einsum('isj->i',alpha_isj(trade_cf)-alpha_isj(trade_baseline)).ravel()
# for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
# data = data_base.loc[data_base['group_label'] == group_label]
data= data_base
ax.scatter(data.x,data.y
           # , label = group_label,color=colors[g]
           )
texts = [plt.text(data.x.loc[i], 
        data.y.loc[i], 
        data.loc[i].country_name,
        size=15,
        # color=colors[g]
        ) 
        for i in data_base.index]     # For kernel density
# texts = texts+texts_group
# ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
# ax2.grid([])
ax.set_ylabel('d(alpha)',fontsize = 30)
ax.set_xlabel('Alpha',fontsize = 30)
# ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
# ax.set_xlim([1e-2,3e1])
# plt.legend(fontsize = 25)
plt.xscale('log')
# plt.yscale('log')
plt.title('Term 4')

adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))
if save or save_all:
    plt.savefig(save_path+'term_4_alpha_i.'+save_format,format=save_format)

plt.show()

#%% stacked all terms
from matplotlib import cm
cmap = cm.get_cmap('Spectral')


trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

data_base = pd.DataFrame(columns=['term','spec1','spec2','value']).set_index(['term','spec1','spec2'])

data_base.loc[('term 1','',''),'value'] = (X(trade_cf) - X(trade_baseline))/X(trade_baseline)

term2 = np.einsum('s,s,s->s',
                   epsilon_s(trade_baseline,e),
                   alpha_s(trade_cf)-alpha_s(trade_baseline),
                   1/alpha_s(trade_baseline))

for i,sector in enumerate(b.sector_list):
    data_base.loc[('term 2 reduction',sector_map.loc[sector,'industry'],''),'value'] = term2[i]
    
term3 = np.einsum('sj,sj,sj->sj',
           epsilon_sj(trade_baseline,e),
           alpha_sj(trade_cf)-alpha_sj(trade_baseline),
           1/alpha_sj(trade_baseline))

for i,country in enumerate(b.country_list):
    for j,sector in enumerate(b.sector_list):
        data_base.loc[('term 3 reduction',sector_map.loc[sector,'industry'],country),'value'] = term3[j,i]
    
term4 = np.einsum('isj,isj,isj->is',
           epsilon_isj(trade_baseline,e),
           alpha_isj(trade_cf)-alpha_isj(trade_baseline),
           np.divide(1, 
                     alpha_isj(trade_baseline), 
                     out = np.zeros_like(alpha_isj(trade_baseline)), 
                     where = alpha_isj(trade_baseline)!=0 ))

for i,exp in tqdm(enumerate(b.country_list)):
    for j,sector in enumerate(b.sector_list):
        data_base.loc[('term 4 reduction',sector_map.loc[sector,'industry'],exp),'value'] = term4[i,j]
# index = pd.MultiIndex.from_product([['term 4'],b.country_list,sector_map.industry.to_list(),b.country_list])
# for j,i in tqdm(enumerate(index)):
#     data_base.loc[i,'value'] = term4.ravel()[j]
# data_base.value = np.abs(data_base.value)
# data_base1 = data_base.loc[data_base.value<0].copy()
# data_base2 = data_base.loc[data_base.value>0].copy()

data_base.reset_index(inplace = True)
data_base.loc[data_base.value>0, 'term'] = data_base.loc[data_base.value>0, 'term'].str.replace('reduction','increase')
data_base.set_index(['term','spec1','spec2'],inplace=True)
data_base.sort_index(inplace=True)
data_base.value = np.abs(data_base.value)
# data_base.loc[('untouched emissions','',''),'value'] = 1-data_base.value.sum()
data_base.reset_index(inplace = True)
data_base = data_base.replace('',None)

# data_base2.value = np.abs(data_base2.value)
# data_base2.reset_index(inplace = True)
# data_base2 = data_base2.replace('',None)


# fig, ax = plt.subplots(figsize=(25,15))
# print('here')
# ax.stackplot(data_base.value.to_list(),
#               labels=data_base.spec.to_list() )
# ax.plot(carb_taxes[1:],[l_em_incr[:i].sum() for i in range(len(l_em_incr))], 
#           label='Emissions',color='black'
#           ,lw=3)
# ax.plot(carb_taxes,norm_emissions-1, 
#           label='Emissions real',color='y'
#           ,lw=3)
# ax.legend(loc='lower left',fontsize = 20)
# ax.tick_params(axis='both', which='major', labelsize=15)
# ax.set_xlabel('Carbon tax',fontsize = 20)

# ax = data_base.T.plot(kind='bar', stacked=True,legend=False, cmap=cmap, linewidth=0.5,edgecolor='grey'
#                  , figsize=(20, 10))
# for c in ax.containers:

#     # Optional: if the segment is small or 0, customize the labels
#     # labels = [v.get_height() if v.get_height() > 0 else '' for v in c]
#     ax.bar_label(c, labels=data_base.index.get_level_values(1).to_list(), label_type='center')

# plt.show()
#%%
import plotly.io as pio
import plotly.express as px
# import plotly.graph_objects as go
pio.renderers.default='browser'
color_discrete_map = {
    'term 1':sns.color_palette("Paired").as_hex()[1],
    'term 2 reduction':sns.color_palette("Paired").as_hex()[3],
    'term 2 increase':sns.color_palette("Paired").as_hex()[2],
    'term 3 reduction':sns.color_palette("Paired").as_hex()[5],
    'term 3 increase':sns.color_palette("Paired").as_hex()[4],
    'term 4 reduction':sns.color_palette("Paired").as_hex()[7],
    'term 4 increase':sns.color_palette("Paired").as_hex()[6],
    # 'untouched emissions':sns.color_palette("Paired").as_hex()[8]
    }
# color_discrete_map = {
#     'term 1':sns.color_palette().as_hex()[0],
#     'term 2':sns.color_palette().as_hex()[1],
#     'term 2 positive':sns.color_palette().as_hex()[1],
#     'term 3':sns.color_palette().as_hex()[2],
#     'term 3 positive':sns.color_palette().as_hex()[2],
#     'term 4':sns.color_palette().as_hex()[3],
#     'term 4 positive':sns.color_palette().as_hex()[3],
#     'untouched emissions':sns.color_palette("Paired").as_hex()[8]
#     }

# color_discrete_map = {term:sns.color_palette("Paired").as_hex() for i,term in enumerate(data_base.reset_index().term.drop_duplicates())}

fig1 = px.sunburst(data_base, path=['term', 'spec1','spec2'], values='value', color='term',
                    color_discrete_map=color_discrete_map)
fig1.update_traces(sort=False, selector=dict(type='sunburst')) 
# fig1 = go.Figure()
# fig1.add_trace(go.Sunburst(ids=data_base1.spec2, values=data_base1.value))
# fig2.update_layout(title_text="Two-level Sunburst Diagram", font_size=10)
fig1.show()
# fig2 = px.sunburst(data_base2, path=['term', 'spec1','spec2'], values='value', color='term',
#                    color_discrete_map=color_discrete_map)
# # fig2.update_layout(title_text="Two-level Sunburst Diagram", font_size=10)
# fig2.show()

#%% stacked all terms
from matplotlib import cm
cmap = cm.get_cmap('Spectral')


trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

data_base = pd.DataFrame(columns=['term','spec1','spec2','value']).set_index(['term','spec1','spec2'])

data_base.loc[('term 1','',''),'value'] = (X(trade_cf) - X(trade_baseline))/X(trade_baseline)

term2 = np.einsum('s->s',
                   alpha_s(trade_cf)-alpha_s(trade_baseline))

for i,sector in enumerate(b.sector_list):
    data_base.loc[('term 2 reduction',sector_map.loc[sector,'industry'],''),'value'] = term2[i]
    
term3 = np.einsum('sj,->sj',
           alpha_sj(trade_cf)-alpha_sj(trade_baseline))

for i,country in enumerate(b.country_list):
    for j,sector in enumerate(b.sector_list):
        data_base.loc[('term 3 reduction',sector_map.loc[sector,'industry'],country),'value'] = term3[j,i]
    
term4 = np.einsum('isj,->is',
           alpha_isj(trade_cf)-alpha_isj(trade_baseline))

for i,exp in tqdm(enumerate(b.country_list)):
    for j,sector in enumerate(b.sector_list):
        data_base.loc[('term 4 reduction',sector_map.loc[sector,'industry'],exp),'value'] = term4[i,j]
# index = pd.MultiIndex.from_product([['term 4'],b.country_list,sector_map.industry.to_list(),b.country_list])
# for j,i in tqdm(enumerate(index)):
#     data_base.loc[i,'value'] = term4.ravel()[j]
# data_base.value = np.abs(data_base.value)
# data_base1 = data_base.loc[data_base.value<0].copy()
# data_base2 = data_base.loc[data_base.value>0].copy()

data_base.reset_index(inplace = True)
data_base.loc[data_base.value>0, 'term'] = data_base.loc[data_base.value>0, 'term'].str.replace('reduction','increase')
data_base.set_index(['term','spec1','spec2'],inplace=True)
data_base.sort_index(inplace=True)
data_base.value = np.abs(data_base.value)
# data_base.loc[('untouched emissions','',''),'value'] = 1-data_base.value.sum()
data_base.reset_index(inplace = True)
data_base = data_base.replace('',None)

# data_base2.value = np.abs(data_base2.value)
# data_base2.reset_index(inplace = True)
# data_base2 = data_base2.replace('',None)


# fig, ax = plt.subplots(figsize=(25,15))
# print('here')
# ax.stackplot(data_base.value.to_list(),
#               labels=data_base.spec.to_list() )
# ax.plot(carb_taxes[1:],[l_em_incr[:i].sum() for i in range(len(l_em_incr))], 
#           label='Emissions',color='black'
#           ,lw=3)
# ax.plot(carb_taxes,norm_emissions-1, 
#           label='Emissions real',color='y'
#           ,lw=3)
# ax.legend(loc='lower left',fontsize = 20)
# ax.tick_params(axis='both', which='major', labelsize=15)
# ax.set_xlabel('Carbon tax',fontsize = 20)

# ax = data_base.T.plot(kind='bar', stacked=True,legend=False, cmap=cmap, linewidth=0.5,edgecolor='grey'
#                  , figsize=(20, 10))
# for c in ax.containers:

#     # Optional: if the segment is small or 0, customize the labels
#     # labels = [v.get_height() if v.get_height() > 0 else '' for v in c]
#     ax.bar_label(c, labels=data_base.index.get_level_values(1).to_list(), label_type='center')

# plt.show()

import plotly.io as pio
import plotly.express as px
# import plotly.graph_objects as go
pio.renderers.default='browser'
color_discrete_map = {
    'term 1':sns.color_palette("Paired").as_hex()[1],
    'term 2 reduction':sns.color_palette("Paired").as_hex()[3],
    'term 2 increase':sns.color_palette("Paired").as_hex()[2],
    'term 3 reduction':sns.color_palette("Paired").as_hex()[5],
    'term 3 increase':sns.color_palette("Paired").as_hex()[4],
    'term 4 reduction':sns.color_palette("Paired").as_hex()[7],
    'term 4 increase':sns.color_palette("Paired").as_hex()[6],
    # 'untouched emissions':sns.color_palette("Paired").as_hex()[8]
    }
# color_discrete_map = {
#     'term 1':sns.color_palette().as_hex()[0],
#     'term 2':sns.color_palette().as_hex()[1],
#     'term 2 positive':sns.color_palette().as_hex()[1],
#     'term 3':sns.color_palette().as_hex()[2],
#     'term 3 positive':sns.color_palette().as_hex()[2],
#     'term 4':sns.color_palette().as_hex()[3],
#     'term 4 positive':sns.color_palette().as_hex()[3],
#     'untouched emissions':sns.color_palette("Paired").as_hex()[8]
#     }

# color_discrete_map = {term:sns.color_palette("Paired").as_hex() for i,term in enumerate(data_base.reset_index().term.drop_duplicates())}

fig1 = px.sunburst(data_base, path=['term', 'spec1','spec2'], values='value', color='term',
                    color_discrete_map=color_discrete_map)
fig1.update_traces(sort=False, selector=dict(type='sunburst')) 
# fig1 = go.Figure()
# fig1.add_trace(go.Sunburst(ids=data_base1.spec2, values=data_base1.value))
# fig2.update_layout(title_text="Two-level Sunburst Diagram", font_size=10)
fig1.show()
# fig2 = px.sunburst(data_base2, path=['term', 'spec1','spec2'], values='value', color='term',
#                    color_discrete_map=color_discrete_map)
# # fig2.update_layout(title_text="Two-level Sunburst Diagram", font_size=10)
# fig2.show()
