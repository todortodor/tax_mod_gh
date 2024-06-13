#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 16:10:23 2024

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
from ast import literal_eval
from tqdm import tqdm
from scipy.spatial.distance import pdist

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

save_path = '/Users/slepot/Dropbox/Green Logistics/KIS/KIS_03_24/'
save_formats = ['eps','png','pdf']

EU = d.countries_from_fta('EU')
EU_ETS = d.countries_from_fta('EU_ETS')

data_path = main_path+'data/'
results_path = 'results/'
      
y=2018         
year=str(y)

baseline = d.baseline(year, data_path)

#%% load a set of solutions with incremental countries taxed on consumption

dir_num = 55
path = results_path+baseline.year+'_'+str(dir_num)
runs_path = path+'/runs.csv'
    
runs = pd.read_csv(runs_path,index_col=0)
sols = [t.sol(run[1],results_path,data_path) for run in tqdm(runs.iterrows())]

#%% sort the solutions in a dictionary with keys being the countries taxing

sols_by_taxing_countries = {}

for sol in tqdm(sols):
    taxing_countries = str(sol.params.taxing_countries)
    if taxing_countries in sols_by_taxing_countries.keys():
        sols_by_taxing_countries[taxing_countries].append(sol)
    else:
        sols_by_taxing_countries[taxing_countries] = [sol]
        
#%% calculate the decomposition for each countries taxed

import treatment_funcs as t

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

N = baseline.country_number
S = baseline.sector_number

e = baseline.co2_intensity.value.values.reshape((N,S))
    
def compute_decomposition_at_100_for_list_of_sols(sols_by_carb_cost,baseline):
    l_term_1 = []
    l_term_2 = []
    l_term_3 = []
    for i in tqdm(range(1,len(sols_by_carb_cost))):
        sol_b = sols_by_carb_cost[i-1].compute_solution(baseline,with_np_arrays=True)
        sol_cf = sols_by_carb_cost[i].compute_solution(baseline,with_np_arrays=True)
        trade_baseline = sol_b.trade_np
        trade_cf = sol_cf.trade_np
        
        term_1 = (X(trade_cf) - X(trade_baseline))/X(trade_baseline)

        term_2 = np.einsum('s,s,s->s',
                           epsilon_s(trade_baseline,e),
                           alpha_s(trade_cf)-alpha_s(trade_baseline),
                           1/alpha_s(trade_baseline))

        term_3 = np.einsum('is,is,is->is',
                   epsilon_is(trade_baseline,e),
                   alpha_is(trade_cf)-alpha_is(trade_baseline),
                   1/alpha_is(trade_baseline))

        l_term_1.append(term_1)
        l_term_2.append(term_2)
        l_term_3.append(term_3)
        
    l_term_1 = np.array(l_term_1)
    l_term_2 = np.array(l_term_2)
    l_term_3 = np.array(l_term_3)
    
    d_term = {
        'term_1':l_term_1,
        'term_2':l_term_2,
        'term_3':l_term_3,
              }
    
    d_term_summed = {key: [v.sum() for v in l_term] for key, l_term in d_term.items()}
    
    emiss_reduc_contrib = {}
    for term, l_term in d_term_summed.items():
        emiss_reduc_contrib[term] = np.array([l_term[k]*sols_by_carb_cost[k].run.emissions/baseline.co2_prod.value.sum() 
                                      for k in range(len(l_term))])
    
    cumul_terms = {key: np.array([value[:i].sum()
                          for i in range(len(value))]) for key, value in emiss_reduc_contrib.items()}
    
    return cumul_terms

decompositions_by_taxing_countries = {}

for k,sols_by_carb_cost in tqdm(sols_by_taxing_countries.items()):    
    
    decompositions_by_taxing_countries[k] = compute_decomposition_at_100_for_list_of_sols(sols_by_carb_cost,baseline)

#%% put the values in a dataframe and save it

countries = ['EU']

countries = []

for i,count_list in enumerate(pd.Series(decompositions_by_taxing_countries.keys()).apply(literal_eval)):
    if i == 0:
        countries.append('EU')
    else:
        for c in count_list:
            if c not in countries and c not in EU:
                countries.append(c)

df = pd.DataFrame(index=countries,columns=['term_1','term_2','term_3','em_reduc'])

idx_cc = -1
baseline_em = baseline.co2_prod.value.sum()
l_em_reduc = -(np.array([sols_by_taxing_countries[k][idx_cc].run.emissions for k in sols_by_taxing_countries]) - baseline_em)*100/baseline_em

for j,k in enumerate(decompositions_by_taxing_countries.keys()):
    sum_terms = decompositions_by_taxing_countries[k]['term_1'][idx_cc]+decompositions_by_taxing_countries[k]['term_2'][idx_cc]+\
        decompositions_by_taxing_countries[k]['term_3'][idx_cc]
    df.loc[countries[j],'term_1'] = decompositions_by_taxing_countries[k]['term_1'][idx_cc]
    df.loc[countries[j],'term_2'] = decompositions_by_taxing_countries[k]['term_2'][idx_cc]
    df.loc[countries[j],'term_3'] = decompositions_by_taxing_countries[k]['term_3'][idx_cc]
    df.loc[countries[j],'em_reduc'] = l_em_reduc[j]

df.to_csv(save_path+'incremental_tax_countries.csv')

#%% Plot

df = pd.read_csv(save_path+'incremental_tax_countries.csv',index_col=0)

countries = df.index.get_level_values(0)

t1 = df['term_1']*df['em_reduc']/(df['term_1']+df['term_2']+df['term_3'])
t2 = df['term_2']*df['em_reduc']/(df['term_1']+df['term_2']+df['term_3'])
t3 = df['term_3']*df['em_reduc']/(df['term_1']+df['term_2']+df['term_3'])
sum_t = t1+t2+t3

fig,ax = plt.subplots(figsize = (24,12),dpi = 288)

bars1 = ax.barh(countries,
                t1,
                label = 'Scale effect')
bars2 = ax.barh(countries,t2,left=t1, label = 'Composition effect')
bars3 = ax.barh(countries,t3,left=t1+t2, label = 'Sourcing effect')

ax.bar_label(bars3,
              labels=countries,
              label_type = 'edge',
              padding=5,
                fontsize=12,
              zorder=99)

ax.bar_label(bars1,
              labels = [round(t*100/sum_t[i],1) for i,t in enumerate(t1)],
              label_type = 'center',
              padding = 5,
              color = 'k',
              fontsize = 12,
              zorder = 99)
ax.bar_label(bars2,
              labels = [round(t*100/sum_t[i],1) for i,t in enumerate(t2)],
              label_type = 'center',
              padding = 5,
              color = 'k',
              fontsize = 12,
              zorder = 99)
ax.bar_label(bars3,
              labels = [round(t*100/sum_t[i],1) for i,t in enumerate(t3)],
              label_type = 'center',
              padding = 5,
              color = 'k',
              fontsize = 12,
              zorder = 99)

ax.axvline(x=df['em_reduc'].iloc[-1],
            color='red')
ax.set_xlabel('Emissions reduction (%)')

ax.legend(loc=(0,-0.2),title='Decomposition')

plt.tight_layout()

# for save_format in ['eps','png','pdf']:
#     plt.savefig(save_path+'incremental_tax_countries.'+save_format,
#                 format=save_format)

plt.show()

#%% Load EU solution with and without CBAM

elast_path = 'cp_estimate_allyears.csv'

eta_path = elast_path
sigma_path = elast_path

taxed_countries_list = [None]
taxing_countries_list = [EU_ETS]
taxed_sectors_list = [None]
specific_taxing_list = [None]
fair_tax_list = [False]
pol_pay_tax_list=[False]
tax_scheme_list = ['consumer','producer','eu_style']
y  = 2018
year = str(y)
years = [y]
dir_num = [55,56,57,65,66,75,76]
carb_cost_list = [1e-4]

cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                      taxed_sectors_list,specific_taxing_list,fair_tax_list,
                      pol_pay_tax_list,tax_scheme_list,tau_factor_list=[1])

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

df = pd.read_csv(save_path+'incremental_tax_countries.csv',index_col=0)
countries = df.index.get_level_values(0)
countries = countries.to_list()

#%% Plot

welfare_prod_tax = sols[1].utility
welfare_prod_tax_cba = sols[2].utility

x = b.va.groupby(['col_country']).sum()

fig,ax = plt.subplots(figsize = (24,12),dpi = 288)

bars = ax.bar(EU_ETS+countries[1:],welfare_prod_tax.loc[EU_ETS+countries[1:]].hat*100-100,label='Without BCA')
ax.scatter(EU_ETS+countries[1:],welfare_prod_tax_cba.loc[EU_ETS+countries[1:]].hat*100-100,label='With BCA')

ax.bar_label(bars,
              labels=EU_ETS+countries[1:],
              rotation=90,
              label_type = 'edge',
              padding=5,
              fontsize=12,
              zorder=99)
ax.set_xticklabels(['']
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)
ax.vlines(x=len(EU_ETS)-0.5,
            ymin=-1,
            ymax=1,
            lw=3,
            ls = '--',
            color = 'r',label='EU')
ax.set_ylabel('Welfare change')

plt.title('EU taxing only')
plt.legend()

for save_format in ['eps','png','pdf']:
    plt.savefig(save_path+'eu_cbam_welfares.'+save_format,
                format=save_format)

plt.show()

df = pd.DataFrame(index = EU_ETS+countries[1:],columns=['No CBAM','With CBAM'])
df['No CBAM'] = welfare_prod_tax.loc[EU_ETS+countries[1:]].hat*100-100
df['With CBAM'] = welfare_prod_tax_cba.loc[EU_ETS+countries[1:]].hat*100-100
df['Diff pp'] = df['With CBAM'] - df['No CBAM']

df.to_csv(save_path+'eu_cbam_welfares.csv')

#%% load a set of solutions with incremental countries taxed with CBAM

dir_num = 75
data_path = main_path+'data/'
results_path = 'results/'

numeraire_type = 'wage'
numeraire_country = 'WLD'

EEA = d.countries_from_fta('EEA')
EU = d.countries_from_fta('EU')
NAFTA = d.countries_from_fta('NAFTA')
ASEAN = d.countries_from_fta('ASEAN')
AANZFTA = d.countries_from_fta('AANZFTA')
APTA = d.countries_from_fta('APTA')
MERCOSUR = d.countries_from_fta('MERCOSUR')

carb_cost_list = [1e-4]
eta_path = ["cp_estimate_allyears.csv"]
sigma_path = ["cp_estimate_allyears.csv"]
taxed_countries_list = [None]
test_taxed_countries_list = [None]
taxing_countries_list = [[]]
taxed_sectors_list = [None]
specific_taxing_list = [None]
fair_tax_list = [False]
pol_pay_tax_list = [False]
      
y=2018         
year=str(y)

baseline = d.baseline(year, data_path)

results_path = 'results/'
path = results_path+baseline.year+'_'+str(dir_num)
runs_path = path+'/runs.csv'
    
runs = pd.read_csv(runs_path,index_col=0)
sols = [t.sol(run[1],results_path,data_path) for run in tqdm(runs.iterrows())]

#%% sort the solutions in a dictionary with keys being the countries taxing

sols_by_taxing_countries = {}

for sol in tqdm(sols):
    taxing_countries = str(sol.params.taxing_countries)
    if taxing_countries in sols_by_taxing_countries.keys():
        sols_by_taxing_countries[taxing_countries].append(sol)
    else:
        sols_by_taxing_countries[taxing_countries] = [sol]
        
#%% calculate the decomposition for each countries taxed

import treatment_funcs as t

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

N = baseline.country_number
S = baseline.sector_number

e = baseline.co2_intensity.value.values.reshape((N,S))
    
def compute_decomposition_at_100_for_list_of_sols(sols_by_carb_cost,baseline):
    l_term_1 = []
    l_term_2 = []
    l_term_3 = []
    for i in tqdm(range(1,len(sols_by_carb_cost))):
        sol_b = sols_by_carb_cost[i-1].compute_solution(baseline,with_np_arrays=True)
        sol_cf = sols_by_carb_cost[i].compute_solution(baseline,with_np_arrays=True)
        trade_baseline = sol_b.trade_np
        trade_cf = sol_cf.trade_np
        
        term_1 = (X(trade_cf) - X(trade_baseline))/X(trade_baseline)

        term_2 = np.einsum('s,s,s->s',
                           epsilon_s(trade_baseline,e),
                           alpha_s(trade_cf)-alpha_s(trade_baseline),
                           1/alpha_s(trade_baseline))

        term_3 = np.einsum('is,is,is->is',
                   epsilon_is(trade_baseline,e),
                   alpha_is(trade_cf)-alpha_is(trade_baseline),
                   1/alpha_is(trade_baseline))

        l_term_1.append(term_1)
        l_term_2.append(term_2)
        l_term_3.append(term_3)
        
    l_term_1 = np.array(l_term_1)
    l_term_2 = np.array(l_term_2)
    l_term_3 = np.array(l_term_3)
    
    d_term = {
        'term_1':l_term_1,
        'term_2':l_term_2,
        'term_3':l_term_3,
              }
    
    d_term_summed = {key: [v.sum() for v in l_term] for key, l_term in d_term.items()}
    
    emiss_reduc_contrib = {}
    for term, l_term in d_term_summed.items():
        emiss_reduc_contrib[term] = np.array([l_term[k]*sols_by_carb_cost[k].run.emissions/baseline.co2_prod.value.sum() 
                                      for k in range(len(l_term))])
    
    cumul_terms = {key: np.array([value[:i].sum()
                          for i in range(len(value))]) for key, value in emiss_reduc_contrib.items()}
    
    return cumul_terms

decompositions_by_taxing_countries = {}

for k,sols_by_carb_cost in tqdm(sols_by_taxing_countries.items()):    
    
    decompositions_by_taxing_countries[k] = compute_decomposition_at_100_for_list_of_sols(sols_by_carb_cost,baseline)

#%% put the values in a dataframe and save it

countries = ['EU']

countries = []

for i,count_list in enumerate(pd.Series(decompositions_by_taxing_countries.keys()).apply(literal_eval)):
    if i == 0:
        countries.append('EU')
    else:
        for c in count_list:
            if c not in countries and c not in EU:
                countries.append(c)

df = pd.DataFrame(index=countries,columns=['term_1','term_2','term_3','em_reduc'])

idx_cc = -1
baseline_em = baseline.co2_prod.value.sum()
l_em_reduc = -(np.array([sols_by_taxing_countries[k][idx_cc].run.emissions for k in sols_by_taxing_countries]) - baseline_em)*100/baseline_em

for j,k in enumerate(decompositions_by_taxing_countries.keys()):
    sum_terms = decompositions_by_taxing_countries[k]['term_1'][idx_cc]+decompositions_by_taxing_countries[k]['term_2'][idx_cc]+\
        decompositions_by_taxing_countries[k]['term_3'][idx_cc]
    df.loc[countries[j],'term_1'] = decompositions_by_taxing_countries[k]['term_1'][idx_cc]
    df.loc[countries[j],'term_2'] = decompositions_by_taxing_countries[k]['term_2'][idx_cc]
    df.loc[countries[j],'term_3'] = decompositions_by_taxing_countries[k]['term_3'][idx_cc]
    df.loc[countries[j],'em_reduc'] = l_em_reduc[j]

df.to_csv(save_path+'cbam_incremental_tax_countries.csv')

#%% Plot

df = pd.read_csv(save_path+'cbam_incremental_tax_countries.csv',index_col=0)

countries = df.index.get_level_values(0)

t1 = df['term_1']*df['em_reduc']/(df['term_1']+df['term_2']+df['term_3'])
t2 = df['term_2']*df['em_reduc']/(df['term_1']+df['term_2']+df['term_3'])
t3 = df['term_3']*df['em_reduc']/(df['term_1']+df['term_2']+df['term_3'])
sum_t = t1+t2+t3

fig,ax = plt.subplots(figsize = (24,12),dpi = 288)

bars1 = ax.barh(countries,
                t1,
                label = 'Scale effect')
bars2 = ax.barh(countries,t2,left=t1, label = 'Composition effect')
bars3 = ax.barh(countries,t3,left=t1+t2, label = 'Sourcing effect')

ax.bar_label(bars3,
              labels=countries,
              label_type = 'edge',
              padding=5,
                fontsize=12,
              zorder=99)

ax.bar_label(bars1,
              labels = [round(t*100/sum_t[i],1) for i,t in enumerate(t1)],
              label_type = 'center',
              padding = 5,
              color = 'k',
              fontsize = 12,
              zorder = 99)
ax.bar_label(bars2,
              labels = [round(t*100/sum_t[i],1) for i,t in enumerate(t2)],
              label_type = 'center',
              padding = 5,
              color = 'k',
              fontsize = 12,
              zorder = 99)
ax.bar_label(bars3,
              labels = [round(t*100/sum_t[i],1) for i,t in enumerate(t3)],
              label_type = 'center',
              padding = 5,
              color = 'k',
              fontsize = 12,
              zorder = 99)

ax.axvline(x=df['em_reduc'].iloc[-1],
            color='red')
ax.set_xlabel('Emissions reduction (%)')

ax.legend(loc=(0,-0.2),title='Decomposition')

plt.tight_layout()

for save_format in ['eps','png','pdf']:
    plt.savefig(save_path+'cbam_incremental_tax_countries.'+save_format,
                format=save_format)

plt.show()

#%% Load df of terms with and without CBAM

df = pd.read_csv(save_path+'incremental_tax_countries.csv',index_col=0)[['term_1','term_2','term_3']]
sum_terms = df['term_1']+df['term_2']+df['term_3']
for col in df.columns:
    df[col] = df[col]*100/(sum_terms)
df_cba = pd.read_csv(save_path+'cbam_incremental_tax_countries.csv',index_col=0)[['term_1','term_2','term_3']]
sum_terms = df_cba['term_1']+df_cba['term_2']+df_cba['term_3']
for col in df_cba.columns:
    df_cba[col] = df_cba[col]*100/(sum_terms)
    
fig,ax = plt.subplots(figsize = (24,12),dpi = 288)

ax.plot(df.index,df_cba['term_3']-df['term_3']+df['term_3'].iloc[-1]-df_cba['term_3'].iloc[-1],label='Climate club with CBAM')

ax.set_ylabel('Sourcing effect')
ax.axhline(0,color='k',lw=1.5,ls='--')

plt.title('Deviation from baseline tax')
plt.legend()

for save_format in ['eps','png','pdf']:
    plt.savefig(save_path+'comparison_term_3_cbam_no_cbam.'+save_format,
                format=save_format)

plt.show()

#%% load a set of solutions with IMF scheme - compute - save a dataframe


data_path = main_path+'data/'
results_path = 'results/'

y=2018         
year=str(y)

configs = [
    {'dir_num':115,
     'description':'ICPF on all countries',
     'fair_tax_run':215},
    {'dir_num':116,
     'description':'ICPF on G20 countries',
     'fair_tax_run':216},
    {'dir_num':117,
     'description':'ICPF on IMF restricted club',
     'fair_tax_run':217},
    {'dir_num':119,
     'description':'ICPF on key players club',
     'fair_tax_run':219},
    ]

# df = pd.DataFrame(columns=['term_1','term_2','term_3','em_reduc','transfers'])
df = pd.read_csv(save_path+'summary_imf_configurations.csv',index_col = 0)
# df.index = [config['description'] for config in configs]
baseline_em = baseline.co2_prod.value.sum()

for config in configs:
    dir_num = config['dir_num']
    print(config['description'])

    # dir_num = 115
    path = results_path+year+'_'+str(dir_num)
    runs_path = path+'/runs.csv'
        
    runs = pd.read_csv(runs_path,index_col=0)
    sols = [t.sol(run[1],results_path,data_path) for run in tqdm(runs.iterrows())]
    # sols = [t.sol(runs.iloc[-11],results_path,data_path)]
    
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
    
    N = baseline.country_number
    S = baseline.sector_number
    
    e = baseline.co2_intensity.value.values.reshape((N,S))
        
    def compute_decomposition_at_100_for_list_of_sols(sols_by_carb_cost,baseline):
        l_term_1 = []
        l_term_2 = []
        l_term_3 = []
        for i in tqdm(range(1,len(sols_by_carb_cost))):
            sol_b = sols_by_carb_cost[i-1].compute_solution(baseline,with_np_arrays=True)
            sol_cf = sols_by_carb_cost[i].compute_solution(baseline,with_np_arrays=True)
            trade_baseline = sol_b.trade_np
            trade_cf = sol_cf.trade_np
            #here
            
            term_1 = (X(trade_cf) - X(trade_baseline))/X(trade_baseline)
    
            term_2 = np.einsum('s,s,s->s',
                               epsilon_s(trade_baseline,e),
                               alpha_s(trade_cf)-alpha_s(trade_baseline),
                               1/alpha_s(trade_baseline))
    
            term_3 = np.einsum('is,is,is->is',
                       epsilon_is(trade_baseline,e),
                       alpha_is(trade_cf)-alpha_is(trade_baseline),
                       1/alpha_is(trade_baseline))
    
            l_term_1.append(term_1)
            l_term_2.append(term_2)
            l_term_3.append(term_3)
            
        l_term_1 = np.array(l_term_1)
        l_term_2 = np.array(l_term_2)
        l_term_3 = np.array(l_term_3)
        
        d_term = {
            'term_1':l_term_1,
            'term_2':l_term_2,
            'term_3':l_term_3,
                  }
        
        d_term_summed = {key: [v.sum() for v in l_term] for key, l_term in d_term.items()}
        
        emiss_reduc_contrib = {}
        for term, l_term in d_term_summed.items():
            emiss_reduc_contrib[term] = np.array([l_term[k]*sols_by_carb_cost[k].run.emissions/baseline.co2_prod.value.sum() 
                                          for k in range(len(l_term))])
        
        cumul_terms = {key: np.array([value[:i].sum()
                              for i in range(len(value))]) for key, value in emiss_reduc_contrib.items()}
        
        return cumul_terms
    
    decomposition = compute_decomposition_at_100_for_list_of_sols(sols,baseline)
    
    l_em_reduc = (sols[-1].run.emissions- baseline_em)*100/baseline_em
    
    df.loc[config['description'],'term_1'] = decomposition['term_1'][-1]
    df.loc[config['description'],'term_2'] = decomposition['term_2'][-1]
    df.loc[config['description'],'term_3'] = decomposition['term_3'][-1]
    df.loc[config['description'],'em_reduc'] = l_em_reduc
    
    dir_num_transfers = config['fair_tax_run']
    path_transfers = results_path+year+'_'+str(dir_num_transfers)
    runs_path_transfers = path_transfers+'/runs.csv'
    runs_transfers = pd.read_csv(runs_path_transfers,index_col=0)
    sol_transfers = t.sol(runs_transfers.loc[runs_transfers.fair_tax].iloc[0],results_path,data_path)
    transfers = sol_transfers.contrib[sol_transfers.contrib>0].sum().value
    df.loc[config['description'],'transfers'] = transfers
    
    uniform_tax_runs = pd.read_csv('results/2018_50/runs.csv')
    eq_uniform_tax_run = uniform_tax_runs.loc[np.argmin(np.abs(uniform_tax_runs.emissions-sols[-1].run.emissions))]

    eq_uniform_tax_sol = t.sol(eq_uniform_tax_run,results_path,data_path)
    eq_uniform_tax_sol.compute_solution(baseline,inplace=True)
    
    df.loc[config['description'],'additional real income cost'] = 100*(sols[-1].run.utility-eq_uniform_tax_sol.run.utility)/(eq_uniform_tax_sol.run.utility-1)
    
    gsi = pdist([baseline.output.value, sols[-1].compute_solution(baseline).output.value] , 
                      metric = 'correlation')
    
    df.loc[config['description'],'distance with baseline'] = gsi*1e3
    
    gsi = pdist([baseline.output.value, eq_uniform_tax_sol.output.value] , 
                      metric = 'correlation')

    df.loc[config['description'],'distance baseline - eq uniform tax'] = gsi*1e3
    
    gsi = pdist([eq_uniform_tax_sol.output.value, sols[-1].compute_solution(baseline).output.value] , 
                      metric = 'correlation')
    
    df.loc[config['description'],'distance with eq uniform tax'] = gsi*1e3

    uniform_tax_runs_fair = pd.read_csv('results/2018_150/runs.csv')
    eq_uniform_tax_run_fair = uniform_tax_runs_fair.loc[np.argmin(np.abs(uniform_tax_runs_fair.emissions-sols[-1].run.emissions))]

    eq_uniform_tax_sol_fair = t.sol(eq_uniform_tax_run_fair,results_path,data_path)
    eq_uniform_tax_sol_fair.compute_solution(baseline,inplace=True)
    
    df.loc[config['description'],'additional transfers'] = 100*(df.loc[config['description'],'transfers']
                                                                -eq_uniform_tax_sol_fair.contrib[eq_uniform_tax_sol_fair.contrib>0].sum().value
                                                                )/(eq_uniform_tax_sol_fair.contrib[eq_uniform_tax_sol_fair.contrib>0].sum().value)     
    
    df.loc[config['description'],'tax equivalent'] = eq_uniform_tax_run.carb_cost*1e6
    df.loc[config['description'],'fair tax equivalent'] = eq_uniform_tax_run_fair.carb_cost*1e6
    
    df.loc[config['description'],'equivalent fair tax transfers'] = eq_uniform_tax_sol_fair.contrib[eq_uniform_tax_sol_fair.contrib>0].sum().value
    
    print(df)
    
df.to_csv(save_path+'summary_imf_configurations.csv')
    
#%% Plot

import matplotlib.patches as patches

df = pd.read_csv(save_path+'summary_imf_configurations.csv',index_col = 0)

t1 = -df['term_1']*df['em_reduc']/(df['term_1']+df['term_2']+df['term_3'])
t2 = -df['term_2']*df['em_reduc']/(df['term_1']+df['term_2']+df['term_3'])
t3 = -df['term_3']*df['em_reduc']/(df['term_1']+df['term_2']+df['term_3'])
sum_t = t1+t2+t3

fig,ax = plt.subplots(1,2,figsize = (24,12),dpi = 288)

bars1 = ax[0].barh(df.index.get_level_values(0),
                t1,
                label = 'Scale effect')
bars2 = ax[0].barh(df.index.get_level_values(0),t2,left=t1, label = 'Composition effect')
bars3 = ax[0].barh(df.index.get_level_values(0),t3,left=t1+t2, label = 'Sourcing effect')

# ax[0].bar_label(bars3,
#               labels=df.index.get_level_values(0),
#               label_type = 'edge',
#               padding=5,
#                 fontsize=12,
#               zorder=99)

ax[0].bar_label(bars1,
              labels = [round(t*100/sum_t[i],1) for i,t in enumerate(t1)],
              label_type = 'center',
              padding = 5,
              color = 'k',
              fontsize = 12,
              zorder = 99)
ax[0].bar_label(bars2,
              labels = [round(t*100/sum_t[i],1) for i,t in enumerate(t2)],
              label_type = 'center',
              padding = 5,
              color = 'k',
              fontsize = 12,
              zorder = 99)
ax[0].bar_label(bars3,
              labels = [round(t*100/sum_t[i],1) for i,t in enumerate(t3)],
              label_type = 'center',
              padding = 5,
              color = 'k',
              fontsize = 12,
              zorder = 99)

ax[0].set_xlabel('Emissions reduction (%)')

ax[0].legend(loc=(0,-0.2),title='Decomposition')

# economic_cost = ax[1].barh(
#                 df.index.get_level_values(0),
#                 df['additional real income cost'],
#                 edgecolor='blue',  # Set the color of the edge here
#                 linewidth=1.5,     # Set the thickness of the edge here
#                 color='none',      # This makes the inside of the bars transparent
#                 )

for index, value in enumerate(df['additional real income cost']):
    y = index  # The y-position of the bar
    
    # Calculate the x positions for the left and right edges of each bar
    right_edge_x = value
    if index == 0:
        label1 = 'Relative increase in economic cost (%)'
    else:
        label1 = None
    
    # The height of the 'line' is actually the thickness of the bar. Adjust as needed.
    bar_height = ax[1].barh(df.index.get_level_values(0), df['additional real income cost'],
                            label = label1).patches[0].get_height()
    
    # Draw the right edge as a rectangle
    ax[1].add_patch(
        patches.Rectangle((right_edge_x, y - bar_height / 2), 1, bar_height, 
                          edgecolor=sns.color_palette()[0], facecolor=sns.color_palette()[0]),
        
    )
    
    right_edge_x_2 = df.iloc[index]['additional transfers']
    if index == 0:
        label2 = 'Relative increase in North-South aid (%)'
    else:
        label2 = None
    
    bar_height_2 = ax[1].barh(df.index.get_level_values(0), df['additional transfers'],
                            label = label2).patches[0].get_height()
    
    # Draw the right edge as a rectangle
    ax[1].add_patch(
        patches.Rectangle((right_edge_x_2, y - bar_height_2 / 2), 1, bar_height_2, 
                          edgecolor=sns.color_palette()[1], facecolor=sns.color_palette()[1]),
        
    )

# Remove the original bar objects since we only need the edges
for container in ax[1].containers:
    for bar in container:
        bar.remove()

ax[1].legend(loc=(0,-0.2))
ax[1].tick_params(labelleft=False) 
ax[1].set_xlabel('Comparison with equivalent uniform tax') 
ax[1].axvline(0,color='k',ls='--',lw=1.5)

plt.tight_layout()

for save_format in ['eps','png','pdf']:
    plt.savefig(save_path+'summary_imf_configurations.'+save_format,
                format=save_format)

plt.show()

#%% load a set of solutions with CBAM - compute - save a dataframe


data_path = main_path+'data/'
results_path = 'results/'

y=2018         
year=str(y)

configs = [
    {'dir_num':125,
     'description':'EU tax with CBAM',
     'fair_tax_run':225},
    {'dir_num':127,
     'description':'Key players club with CBAM',
     'fair_tax_run':227},
    ]

df = pd.DataFrame(columns=['term_1','term_2','term_3','em_reduc','transfers'])
# df = pd.read_csv(save_path+'summary_cbam_configurations.csv',index_col = 0)
# df.index = [config['description'] for config in configs]
baseline_em = baseline.co2_prod.value.sum()

for config in configs:
    dir_num = config['dir_num']
    print(config['description'])

    # dir_num = 115
    path = results_path+year+'_'+str(dir_num)
    runs_path = path+'/runs.csv'
        
    runs = pd.read_csv(runs_path,index_col=0)
    sols = [t.sol(run[1],results_path,data_path) for run in tqdm(runs.iterrows())]
    # sols = [t.sol(runs.iloc[-11],results_path,data_path)]
    
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
    
    N = baseline.country_number
    S = baseline.sector_number
    
    e = baseline.co2_intensity.value.values.reshape((N,S))
        
    def compute_decomposition_at_100_for_list_of_sols(sols_by_carb_cost,baseline):
        l_term_1 = []
        l_term_2 = []
        l_term_3 = []
        for i in tqdm(range(1,len(sols_by_carb_cost))):
            sol_b = sols_by_carb_cost[i-1].compute_solution(baseline,with_np_arrays=True)
            sol_cf = sols_by_carb_cost[i].compute_solution(baseline,with_np_arrays=True)
            trade_baseline = sol_b.trade_np
            trade_cf = sol_cf.trade_np
            #here
            
            term_1 = (X(trade_cf) - X(trade_baseline))/X(trade_baseline)
    
            term_2 = np.einsum('s,s,s->s',
                               epsilon_s(trade_baseline,e),
                               alpha_s(trade_cf)-alpha_s(trade_baseline),
                               1/alpha_s(trade_baseline))
    
            term_3 = np.einsum('is,is,is->is',
                       epsilon_is(trade_baseline,e),
                       alpha_is(trade_cf)-alpha_is(trade_baseline),
                       1/alpha_is(trade_baseline))
    
            l_term_1.append(term_1)
            l_term_2.append(term_2)
            l_term_3.append(term_3)
            
        l_term_1 = np.array(l_term_1)
        l_term_2 = np.array(l_term_2)
        l_term_3 = np.array(l_term_3)
        
        d_term = {
            'term_1':l_term_1,
            'term_2':l_term_2,
            'term_3':l_term_3,
                  }
        
        d_term_summed = {key: [v.sum() for v in l_term] for key, l_term in d_term.items()}
        
        emiss_reduc_contrib = {}
        for term, l_term in d_term_summed.items():
            emiss_reduc_contrib[term] = np.array([l_term[k]*sols_by_carb_cost[k].run.emissions/baseline.co2_prod.value.sum() 
                                          for k in range(len(l_term))])
        
        cumul_terms = {key: np.array([value[:i].sum()
                              for i in range(len(value))]) for key, value in emiss_reduc_contrib.items()}
        
        return cumul_terms
    
    decomposition = compute_decomposition_at_100_for_list_of_sols(sols,baseline)
    
    l_em_reduc = (sols[-1].run.emissions- baseline_em)*100/baseline_em
    
    df.loc[config['description'],'term_1'] = decomposition['term_1'][-1]
    df.loc[config['description'],'term_2'] = decomposition['term_2'][-1]
    df.loc[config['description'],'term_3'] = decomposition['term_3'][-1]
    df.loc[config['description'],'em_reduc'] = l_em_reduc
    
    dir_num_transfers = config['fair_tax_run']
    path_transfers = results_path+year+'_'+str(dir_num_transfers)
    runs_path_transfers = path_transfers+'/runs.csv'
    runs_transfers = pd.read_csv(runs_path_transfers,index_col=0)
    sol_transfers = t.sol(runs_transfers.loc[runs_transfers.fair_tax].iloc[0],results_path,data_path)
    transfers = sol_transfers.contrib[sol_transfers.contrib>0].sum().value
    df.loc[config['description'],'transfers'] = transfers
    
    uniform_tax_runs = pd.read_csv('results/2018_50/runs.csv')
    eq_uniform_tax_run = uniform_tax_runs.loc[np.argmin(np.abs(uniform_tax_runs.emissions-sols[-1].run.emissions))]

    eq_uniform_tax_sol = t.sol(eq_uniform_tax_run,results_path,data_path)
    eq_uniform_tax_sol.compute_solution(baseline,inplace=True)
    
    df.loc[config['description'],'additional real income cost'] = 100*(sols[-1].run.utility-eq_uniform_tax_sol.run.utility)/(eq_uniform_tax_sol.run.utility-1)
    
    gsi = pdist([baseline.output.value, sols[-1].compute_solution(baseline).output.value] , 
                      metric = 'correlation')
    
    df.loc[config['description'],'distance with baseline'] = gsi*1e3
    
    gsi = pdist([baseline.output.value, eq_uniform_tax_sol.output.value] , 
                      metric = 'correlation')

    df.loc[config['description'],'distance baseline - eq uniform tax'] = gsi*1e3
    
    gsi = pdist([eq_uniform_tax_sol.output.value, sols[-1].compute_solution(baseline).output.value] , 
                      metric = 'correlation')
    
    df.loc[config['description'],'distance with eq uniform tax'] = gsi*1e3

    uniform_tax_runs_fair = pd.read_csv('results/2018_150/runs.csv')
    eq_uniform_tax_run_fair = uniform_tax_runs_fair.loc[np.argmin(np.abs(uniform_tax_runs_fair.emissions-sols[-1].run.emissions))]

    eq_uniform_tax_sol_fair = t.sol(eq_uniform_tax_run_fair,results_path,data_path)
    eq_uniform_tax_sol_fair.compute_solution(baseline,inplace=True)
    
    df.loc[config['description'],'additional transfers'] = 100*(df.loc[config['description'],'transfers']
                                                                       -eq_uniform_tax_sol_fair.contrib[eq_uniform_tax_sol_fair.contrib>0].sum().value
                                                                       )/(eq_uniform_tax_sol_fair.contrib[eq_uniform_tax_sol_fair.contrib>0].sum().value)    
    
    df.loc[config['description'],'tax equivalent'] = eq_uniform_tax_run.carb_cost*1e6
    df.loc[config['description'],'fair tax equivalent'] = eq_uniform_tax_run_fair.carb_cost*1e6
    
    transfers = sol_transfers.contrib[sol_transfers.contrib>0].sum().value
    df.loc[config['description'],'equivalent fair tax transfers'] = eq_uniform_tax_sol_fair.contrib[eq_uniform_tax_sol_fair.contrib>0].sum().value
    
    print(df)
    
df.to_csv(save_path+'summary_cbam_configurations.csv')
    
#%% Plot

import matplotlib.patches as patches

df = pd.read_csv(save_path+'summary_cbam_configurations.csv',index_col = 0)

df = df.drop('EU tax with CBAM')

t1 = -df['term_1']*df['em_reduc']/(df['term_1']+df['term_2']+df['term_3'])
t2 = -df['term_2']*df['em_reduc']/(df['term_1']+df['term_2']+df['term_3'])
t3 = -df['term_3']*df['em_reduc']/(df['term_1']+df['term_2']+df['term_3'])
sum_t = t1+t2+t3

fig,ax = plt.subplots(1,2,figsize = (24,12),dpi = 288)

bars1 = ax[0].barh(df.index.get_level_values(0),
                t1,
                label = 'Scale effect')
bars2 = ax[0].barh(df.index.get_level_values(0),t2,left=t1, label = 'Composition effect')
bars3 = ax[0].barh(df.index.get_level_values(0),t3,left=t1+t2, label = 'Sourcing effect')

# ax[0].bar_label(bars3,
#               labels=df.index.get_level_values(0),
#               label_type = 'edge',
#               padding=5,
#                 fontsize=12,
#               zorder=99)

ax[0].bar_label(bars1,
              labels = [round(t*100/sum_t[i],1) for i,t in enumerate(t1)],
              label_type = 'center',
              padding = 5,
              color = 'k',
              fontsize = 12,
              zorder = 99)
ax[0].bar_label(bars2,
              labels = [round(t*100/sum_t[i],1) for i,t in enumerate(t2)],
              label_type = 'center',
              padding = 5,
              color = 'k',
              fontsize = 12,
              zorder = 99)
ax[0].bar_label(bars3,
              labels = [round(t*100/sum_t[i],1) for i,t in enumerate(t3)],
              label_type = 'center',
              padding = 5,
              color = 'k',
              fontsize = 12,
              zorder = 99)

ax[0].set_xlabel('Emissions reduction (%)')

ax[0].legend(loc=(0,-0.2),title='Decomposition')

# economic_cost = ax[1].barh(
#                 df.index.get_level_values(0),
#                 df['additional real income cost'],
#                 edgecolor='blue',  # Set the color of the edge here
#                 linewidth=1.5,     # Set the thickness of the edge here
#                 color='none',      # This makes the inside of the bars transparent
#                 )

for index, value in enumerate(df['additional real income cost']):
    y = index  # The y-position of the bar
    
    # Calculate the x positions for the left and right edges of each bar
    right_edge_x = value
    if index == 0:
        label1 = 'Relative increase in economic cost (%)'
    else:
        label1 = None
    
    # The height of the 'line' is actually the thickness of the bar. Adjust as needed.
    bar_height = ax[1].barh(df.index.get_level_values(0), df['additional real income cost'],
                            label = label1).patches[0].get_height()
    
    # Draw the right edge as a rectangle
    ax[1].add_patch(
        patches.Rectangle((right_edge_x, y - bar_height / 2), 1, bar_height, 
                          edgecolor=sns.color_palette()[0], facecolor=sns.color_palette()[0]),
        
    )
    
    right_edge_x_2 = df.iloc[index]['additional transfers']
    if index == 0:
        label2 = 'Relative increase in North-South aid (%)'
    else:
        label2 = None
    
    bar_height_2 = ax[1].barh(df.index.get_level_values(0), df['additional transfers'],
                            label = label2).patches[0].get_height()
    
    # Draw the right edge as a rectangle
    ax[1].add_patch(
        patches.Rectangle((right_edge_x_2, y - bar_height_2 / 2), 1, bar_height_2, 
                          edgecolor=sns.color_palette()[1], facecolor=sns.color_palette()[1]),
        
    )

# Remove the original bar objects since we only need the edges
for container in ax[1].containers:
    for bar in container:
        bar.remove()

ax[1].legend(loc=(0,-0.2))
ax[1].tick_params(labelleft=False) 
ax[1].set_xlabel('Comparison with equivalent uniform tax') 
ax[1].axvline(0,color='k',ls='--',lw=1.5)

plt.tight_layout()

for save_format in ['eps','png','pdf']:
    plt.savefig(save_path+'summary_cbam_configurations.'+save_format,
                format=save_format)

plt.show()

#%% detailed graphs of welfare and transfers
import os

labor = baseline.labor.set_index('country').rename_axis('col_country')['2018'].to_frame()
labor.columns = ['value']
country_list = baseline.iot.index.get_level_values(0).drop_duplicates().to_list()

detail_save_path = save_path+'details/'

try:
    os.mkdir(detail_save_path)
except:
    pass

income_colors = {
    'Low-income' : sns.color_palette()[3],
    'Middle-income' : sns.color_palette()[0],
    'High-income' : sns.color_palette()[2],
                    }

configs = [
    {'dir_num':115,
     'description':'ICPF on all countries',
     'fair_tax_run':215},
    {'dir_num':116,
     'description':'ICPF on G20 countries',
     'fair_tax_run':216},
    {'dir_num':117,
     'description':'ICPF on IMF restricted club',
     'fair_tax_run':217},
    {'dir_num':119,
     'description':'ICPF on key players club',
     'fair_tax_run':219},
    ]

year = str(2018)

for config in configs:
    dir_num = config['dir_num']
    print(config['description'])

    # dir_num = 115
    path = results_path+year+'_'+str(dir_num)
    runs_path = path+'/runs.csv'
        
    runs = pd.read_csv(runs_path,index_col=0)
    run = runs.iloc[-1]
    sol = t.sol(run,results_path,data_path)
    sol.compute_solution(baseline,inplace=True)
    
    dir_num_transfers = config['fair_tax_run']
    path_transfers = results_path+year+'_'+str(dir_num_transfers)
    runs_path_transfers = path_transfers+'/runs.csv'
    runs_transfers = pd.read_csv(runs_path_transfers,index_col=0)
    sol_transfers = t.sol(runs_transfers.loc[runs_transfers.fair_tax].iloc[0],results_path,data_path)
    transfers = sol_transfers.contrib[sol_transfers.contrib>0].sum().value
    
    uniform_tax_runs = pd.read_csv('results/2018_50/runs.csv')
    eq_uniform_tax_run = uniform_tax_runs.loc[np.argmin(np.abs(uniform_tax_runs.emissions-sol.run.emissions))]
    eq_uniform_tax_sol = t.sol(eq_uniform_tax_run,results_path,data_path)
    eq_uniform_tax_sol.compute_solution(baseline,inplace=True)
    
    uniform_tax_runs_fair = pd.read_csv('results/2018_150/runs.csv')
    eq_uniform_tax_run_fair = uniform_tax_runs_fair.loc[np.argmin(np.abs(uniform_tax_runs_fair.emissions-sol.run.emissions))]
    eq_uniform_tax_sol_fair = t.sol(eq_uniform_tax_run_fair,results_path,data_path)
    eq_uniform_tax_sol_fair.compute_solution(baseline,inplace=True)
    
    df = pd.DataFrame(index = pd.Index(country_list,name='country'))
    
    fig, ax = plt.subplots(1,2,figsize=(20,10),constrained_layout = True)
    
    contrib = sol_transfers.contrib.copy()
    contrib['per_capita'] = contrib['value']*1e6/labor['value']
    contrib = contrib.join(
        pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0)
        ).sort_values('per_capita')

    colors = [income_colors[contrib.loc[country,'income_label']] for country in contrib.index]

    ax[0].bar(contrib.index,
           contrib['per_capita'],
           color=colors
           )
    ax[0].set_xticklabels([''])
    ax[0].bar_label(ax[0].containers[0],
                 labels=contrib.index.get_level_values(0), 
                 rotation=90,
                  label_type = 'edge',
                  padding=5,zorder=10,fontsize=10)
    handles = [mpatches.Patch(color=income_colors[ind], label=ind) for ind in contrib['income_label'].drop_duplicates()]
    ax[0].legend(handles=handles,fontsize=25)
    ax[0].grid(axis='x')
    ax[0].set_ylabel('Monetary transfer per capita (US$)',fontsize=25)
    ax[0].set_title(config['description'])
    
    y_max = contrib.per_capita.max()
    y_min = contrib.per_capita.min()
    
    df['transfers'] = contrib.per_capita
    
    contrib = eq_uniform_tax_sol_fair.contrib.copy()
    contrib['per_capita'] = contrib['value']*1e6/labor['value']
    contrib = contrib.join(
        pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0)
        ).sort_values('per_capita')
    
    colors = [income_colors[contrib.loc[country,'income_label']] for country in contrib.index]

    ax[1].bar(contrib.index,
           contrib['per_capita'],
           color=colors
           )
    ax[1].set_xticklabels([''])
    ax[1].bar_label(ax[1].containers[0],
                 labels=contrib.index.get_level_values(0), 
                 rotation=90,
                  label_type = 'edge',
                  padding=5,zorder=10,fontsize=10)
    handles = [mpatches.Patch(color=income_colors[ind], label=ind) for ind in contrib['income_label'].drop_duplicates()]
    ax[1].legend(handles=handles,fontsize=25)
    ax[1].grid(axis='x')
    ax[1].set_ylabel('Monetary transfer per capita (US$)',fontsize=25)
    ax[1].set_title('Equivalent uniform tax')
    
    y_max = np.max([y_max,contrib.per_capita.max()])
    y_min = np.min([y_min,contrib.per_capita.min()])
    
    ax[0].set_ylim((y_min*1.1,y_max*1.1))
    ax[1].set_ylim((y_min*1.1,y_max*1.1))
    
    df['transfers eq uniform tax'] = contrib.per_capita
    
    for save_format in ['eps','png','pdf']:
        plt.savefig(detail_save_path+f'fair_tax_transfers_by_country_by_income_group_{config["description"]}.'+save_format,format=save_format)
        
    plt.show()
    
    
    fig, ax = plt.subplots(1,2,figsize=(20,10),constrained_layout=True)
    
    welfare_change = sol.utility.copy()

    welfare_change['gdp_p_c'] = (baseline.va.groupby('col_country').sum()/labor).rename_axis('country')
    welfare_change = welfare_change.join(
        pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0)
        )
    welfare_change['hat'] = (welfare_change['hat']-1)*100
    
    colors = [income_colors[welfare_change.loc[country,'income_label']] for country in welfare_change.index]

    ax[0].scatter(welfare_change['gdp_p_c'],
               welfare_change['hat'],
               color = colors,
               lw=5
               )

    texts = [ax[0].text(welfare_change['gdp_p_c'].loc[country], 
                      welfare_change['hat'].loc[country], 
                      country,
                      size=20, 
                      c = colors[i]) for i,country in enumerate(country_list)]

    # adjust_text(texts, precision=0.001,
    #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    #         arrowprops=dict(arrowstyle='-', color='k', alpha=.5)
    #             )
    handles = [mpatches.Patch(color=income_colors[ind], label=ind) for ind in welfare_change['income_label'].drop_duplicates()]
    ax[0].legend(handles=handles,
               fontsize=25,
               loc = 'lower right')
    ax[0].set_xlabel('GDP per capita (millions US$)',fontsize = 25)
    ax[0].set_ylabel('Real income change (%)',fontsize = 25)
    ax[0].set_title(config['description'])
    
    ax[0].axhline(0, color='k')
    
    y_max = welfare_change.hat.max()
    y_min = welfare_change.hat.min()
    
    df['real income change'] = welfare_change.hat
    df['gdp per capita'] = welfare_change['gdp_p_c']
    
    welfare_change = eq_uniform_tax_sol.utility.copy()

    welfare_change['gdp_p_c'] = (baseline.va.groupby('col_country').sum()/labor).rename_axis('country')
    welfare_change = welfare_change.join(
        pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0)
        )
    welfare_change['hat'] = (welfare_change['hat']-1)*100
    
    colors = [income_colors[welfare_change.loc[country,'income_label']] for country in welfare_change.index]

    ax[1].scatter(welfare_change['gdp_p_c'],
               welfare_change['hat'],
               color = colors,
               lw=5
               )

    texts = [ax[1].text(welfare_change['gdp_p_c'].loc[country], 
                      welfare_change['hat'].loc[country], 
                      country,
                      size=20, 
                      c = colors[i]) for i,country in enumerate(country_list)]

    # adjust_text(texts, precision=0.001,
    #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    #         arrowprops=dict(arrowstyle='-', color='k', alpha=.5)
    #             )
    handles = [mpatches.Patch(color=income_colors[ind], label=ind) for ind in welfare_change['income_label'].drop_duplicates()]
    ax[1].legend(handles=handles,
               fontsize=25,
               loc = 'lower right')
    ax[1].set_xlabel('GDP per capita (millions US$)',fontsize = 25)
    ax[1].set_ylabel('Real income change (%)',fontsize = 25)
    ax[1].set_title('Equivalent uniform tax')
    
    ax[1].axhline(0, color='k')
    
    y_max = np.max([y_max,welfare_change.hat.max()])
    y_min = np.min([y_min,welfare_change.hat.min()])
    
    ax[0].set_ylim((y_min*1.1,y_max*1.1))
    ax[1].set_ylim((y_min*1.1,y_max*1.1))
    
    df['real income change equivalent uniform tax'] = welfare_change.hat

    for save_format in ['eps','png','pdf']:
        plt.savefig(detail_save_path+f'welfare_change_by_gdp_by_country_by_income_group_{config["description"]}.'+save_format,format=save_format)
    
    df.to_csv(detail_save_path+f'{config["description"]}.csv')
    
    plt.show()
