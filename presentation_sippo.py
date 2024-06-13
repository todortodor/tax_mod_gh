#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 09:36:47 2024

@author: slepot
"""

import pandas as pd
import lib.data_funcs as d
import lib.treatment_funcs as t
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from adjustText import adjust_text
from tqdm import tqdm
import os

sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 25})

#%% load data

save_path = 'presentation_material/present_sippo/'
data_path = 'data/'
results_path = 'results/'

save_all = False
# save_format = 'pdf'
save_format = 'pdf'

carb_cost_list = np.linspace(0,1e-3,1001)

elast_path = 'cp_estimate_allyears.csv'

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
pol_pay_tax_list=[False]
tau_factor_list = [1]
tax_scheme_list = ['producer']
y  = 2018
year = str(y)
years = [y]
dir_num = [50,60]

cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                      taxed_sectors_list,specific_taxing_list,fair_tax_list,
                      pol_pay_tax_list,tax_scheme_list,tau_factor_list)

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
# l_term_4 = []
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
    'term_3':'Sourcing countries'
    # 'em_reduc':l_em_reduc
          }

cumul_terms = {key: np.array([value[:i].sum()
                      for i in range(len(value))]) for key, value in emiss_reduc_contrib.items()}

#%% plot decomposition diff labels
save=True
cumul_terms = {key: np.array([value[:i].sum()
                      for i in range(len(value))]) for key, value in emiss_reduc_contrib.items()}

fig,ax = plt.subplots(figsize = (16,16))

ax.stackplot(carb_taxes[:-1],
              [term for term in cumul_terms.values()],
              labels=[term_labels[term] for term in cumul_terms.keys()])
ax.plot(carb_taxes[:-1],[l_em_incr[:i].sum() for i in range(len(l_em_incr))], 
          label='Emissions',color='black'
          ,lw=3)
# ax.plot(carb_taxes,norm_emissions-1, 
#           label='Emissions real',color='y'
#           ,lw=3)
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
    loc = 145
    ax.text(carb_taxes[:-1][loc], -(term[1:]/sum_terms[1:])[loc]/2+offset, term_labels[name]+' : '+str(((term[1:]/sum_terms[1:]).mean()*100).round(1))+'%',
            ha='center', va='center',color='white',fontsize = 35)
    offset = offset-(term[1:]/sum_terms[1:])[loc]

ax.tick_params(axis='both', which='major', labelsize=25)
ax.set_xlabel('Carbon tax',fontsize = 35)

if save or save_all:
    plt.savefig(save_path+'decomposition_stacked_norm.'+save_format,format=save_format)
plt.show()

#%%

country_list = b.country_list
C = len(country_list)
sector_list = b.sector_list

sector_map = pd.read_csv(data_path+'industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')

trade_baseline = trade['baseline'].values.reshape((N,S,N))
trade_cf = trade[100].values.reshape((N,S,N)) 

data_base = pd.DataFrame(columns=['term','sector','country','value']
                         ).set_index(['term','sector','country'])

term3 = np.einsum('is->is',
                   alpha_is(trade_cf)-alpha_is(trade_baseline))

for i,sector in enumerate(b.sector_list):
    for j, country in enumerate(b.country_list):
        data_base.loc[('Expenditure share change',sector_map.loc['D'+sector,'industry'],country),'value'
                      ] = term3[j,i]*100
    

total = data_base.loc[data_base.value>0, 'value'].sum()

data_base.reset_index(inplace = True)
data_base['d(alpha)'] = data_base['value']
# data_base.loc[data_base.value>0, 'term'] = data_base.loc[data_base.value>0, 'term'].str.replace('decrease','increase')
data_base.set_index(['term','sector','country'],inplace=True)
data_base.sort_index(inplace=True)
data_base.value = np.abs(data_base.value)
data_base.reset_index(inplace = True)
data_base = data_base.replace('',None)
data_base['value'] = data_base['value']*total/data_base['value'].sum()
data_base = data_base.set_index(['sector','country'])

sippo_sectors = {
    'D03':['TUN','MAR','IDN'],
    'D10T12':['PER','COL','TUN','ZAF','MAR'],
    'D16':['IDN','VNM'],
    'D13T15':['MAR','VNM'],
    'D55T56':['COL','PER'],
    'D01T02':['COL','IDN','MAR','PER','TUN','VNM','ZAF']
    }

data = data_base.loc[[(sector_map.loc[key,'industry'],v) for key,value in sippo_sectors.items() for v in value]]

# data_red = data_base.loc[(data_base.spec1.isin())]
#%%
save = True

colors_dic = {
    'Fishing':0, 
    'Food products':1, 
    'Wood':5, 
    'Textiles':3, 
    'Tourism':4,
    'Agriculture':2
    }

colors = [sns.color_palette()[colors_dic[sector]] for sector in data.index.get_level_values(0)]
data['colors'] = colors

fig, ax = plt.subplots(figsize=(12,10),dpi=288)

# data.plot.barh(y='d(alpha)',ax=ax,legend=False,colors='colors')
ax.barh(y=[' '.join(map(str,i)) for i in data.index.tolist()],
        width=data['d(alpha)'],
        color=data['colors']
        )
# ax.set_xscale('symlog',linthresh=0.01)
# plt.legend()
ax.set_ylabel('')

# fig, ax = plt.subplots(1,2,figsize=(12,10),dpi=288, width_ratios=[2, 1])

# # data.plot.barh(y='d(alpha)',ax=ax,legend=False,colors='colors')
# ax[0].barh(y=[' '.join(map(str,i)) for i in data.index.tolist()],
#         width=data['d(alpha)'],
#         color=data['colors']
#         )
# # ax.set_xscale('symlog',linthresh=0.01)
# # plt.legend()
# ax[0].set_ylabel('')

# ax[1].barh(y=[' '.join(map(str,i)) for i in data.index.tolist()],
#         width=data['d(alpha)'],
#         color=data['colors']
#         )

ax.set_xlim([-0.25,0.11])
# ax[1].set_yticklabels([])

plt.tight_layout()

if save or save_all:
    plt.savefig(save_path+'sippo_country_sectors_gca.'+save_format,format=save_format)

plt.show()