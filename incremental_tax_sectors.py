#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:05:27 2023

@author: slepot
"""

main_path = './'
import sys
sys.path.append(main_path+'lib/')
import solver_funcs as s
import data_funcs as d
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
from time import perf_counter

dir_num = 13
data_path = main_path+'data/'
results_path = 'results/'

numeraire_type = 'wage'
numeraire_country = 'USA'

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
taxing_countries_list = [None]
taxed_sectors_list = [[]]
# spec_tax = pd.DataFrame(index = pd.MultiIndex.from_product([d.get_country_list(),
#                                                             d.get_sector_list(),
#                                                             d.get_country_list()],
#                                                             names = ['row_country',
#                                                                     'row_sector',
#                                                                     'col_country']),
#                         columns = ['value'])
# spec_tax['value'] = 0
# # spec_tax.loc[spec_tax.query("row_country != col_country").index, 'value'] = 1
# specific_taxing_list = [spec_tax]
specific_taxing_list = [None]
fair_tax_list = [False]
pol_pay_tax_list = [False]
      
y=2018         
year=str(y)

baseline = d.baseline(year, data_path)

baseline.num_scale(numeraire_type, numeraire_country, inplace = True)

baseline.make_np_arrays(inplace = True)

baseline.compute_shares_and_gammas(inplace = True)

#%%

sectors_to_test = baseline.sector_list.copy()
sector_emissions = baseline.co2_prod_np.sum()


while len(sectors_to_test)>0:
    winning_sector = sectors_to_test[0]
    sector_emissions = baseline.co2_prod_np.sum()
    for sector in tqdm(sectors_to_test):
        test_taxed_sectors_list = [taxed_sectors_list[0] + [sector]]
        
        cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                              test_taxed_sectors_list,specific_taxing_list,fair_tax_list,pol_pay_tax_list,
                              same_elasticities=True)
        
        for simulation_case in cases:
            params = d.params(data_path, **simulation_case)
            params.num_scale_carb_cost(baseline.num, inplace = True)
            
            if not params.fair_tax and not params.pol_pay_tax:
                results = s.solve_E_p(params, baseline)
            
            if params.fair_tax:
                results = s.solve_fair_tax(params, baseline)
                
            if params.pol_pay_tax:
                results = s.solve_pol_pay_tax(params, baseline)
            
            #compute some aggregated solution quantities to write directly in runs report
            emissions_sol, utility, utility_countries = s.compute_emissions_utility(results, params, baseline)
            
            if emissions_sol<sector_emissions:
                print('new winning sector ',sector,', emissions ',emissions_sol)
                sector_emissions = emissions_sol
                winning_sector = sector
    
    print('overall winning sector ',winning_sector)
    # taxed_sectors_list.append(winning_sector)
    taxed_sectors_list = [taxed_sectors_list[0] + [winning_sector]]
    sectors_to_test.remove(winning_sector)
    
    cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                          taxed_sectors_list,specific_taxing_list,fair_tax_list,pol_pay_tax_list,
                          same_elasticities=True)
    
    for simulation_case in cases:
        params = d.params(data_path, **simulation_case)
        params.num_scale_carb_cost(baseline.num, inplace = True)
        
        if not params.fair_tax and not params.pol_pay_tax:
            results = s.solve_E_p(params, baseline)
        
        if params.fair_tax:
            results = s.solve_fair_tax(params, baseline)
            
        if params.pol_pay_tax:
            results = s.solve_pol_pay_tax(params, baseline)
        
        #compute some aggregated solution quantities to write directly in runs report
        emissions_sol, utility, utility_countries = s.compute_emissions_utility(results, params, baseline)
    
    d.write_solution_csv(results,results_path,dir_num,emissions_sol,utility,params,baseline)
        
    
#%%

dir_num = 13
results_path = 'results/'
path = results_path+'2018'+'_'+str(dir_num)
runs_path = path+'/runs.csv'
    
runs = pd.read_csv(runs_path)

import matplotlib.pyplot as plt

import treatment_funcs as t

sols = [t.sol(run[1],results_path,data_path) for run in runs.iterrows()]

taxed_sectors_list = []

for sol in sols:
    for sector in sol.params.taxed_sectors:
        if len(taxed_sectors_list) == 0:
            taxed_sectors_list = [[sector]]
        if sector not in taxed_sectors_list[-1]:
            taxed_sectors_list.append(taxed_sectors_list[-1]+[sector])

main_path = './'
import sys
sys.path.append(main_path+'lib/')
import solver_funcs as s
import data_funcs as d
from tqdm import tqdm
import numpy as np
# from deco import *
import pandas as pd
import time
from time import perf_counter
# from multiprocessing import Pool
# from multiprocessing import Manager

dir_num = 62
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

# carb_cost_list = [1e-4]
carb_cost_list = np.linspace(0,1e-4,101)
# carb_cost_list = np.array([0,5e-5,1e-4])
eta_path = ["cp_estimate_allyears.csv"]
sigma_path = ["cp_estimate_allyears.csv"]
taxed_countries_list = [None]
taxing_countries_list = [None]
# taxed_sectors_list = [None]
specific_taxing_list = [None]
fair_tax_list = [False]
pol_pay_tax_list = [False]
tax_scheme_list = ['producer']
tau_factor_list = [1]

cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                      taxed_sectors_list,specific_taxing_list,fair_tax_list,pol_pay_tax_list,
                      tax_scheme_list,tau_factor_list,
                      same_elasticities=True)
exclude_direct_emissions = False

years = [2018]
      
for y in years:
    # y=2018         
    year=str(y)
    
    baseline = d.baseline(year, data_path, exclude_direct_emissions=exclude_direct_emissions)
    
    baseline.num_scale(numeraire_type, numeraire_country, inplace = True)
    
    baseline.make_np_arrays(inplace = True)
    
    baseline.compute_shares_and_gammas(inplace = True)
    
    for simulation_case in tqdm(cases):
        if simulation_case['eta_path'] == simulation_case['sigma_path']:

            params = d.params(data_path, **simulation_case)
            params.num_scale_carb_cost(baseline.num, inplace = True)
            
            if not params.fair_tax and not params.pol_pay_tax:
                results = s.solve_E_p(params, baseline)
            
            if params.fair_tax:
                results = s.solve_fair_tax(params, baseline)
                
            if params.pol_pay_tax:
                results = s.solve_pol_pay_tax(params, baseline)
            
            #compute some aggregated solution quantities to write directly in runs report
            emissions_sol, utility, utility_countries = s.compute_emissions_utility(results, params, baseline)
            
            d.write_solution_csv(results,results_path,dir_num,emissions_sol,utility,params,baseline)

#%%

sector_map = pd.read_csv(data_path+'industry_labels_after_agg_expl_wgroup.csv')
sector_map['sector'] = sector_map['ind_code'].str.replace('D','')
sector_map.set_index('sector',inplace=True)

sectors = []
emissions = []

for i,sol in enumerate(sols):
    if i == 0:
        sectors.append(sector_map.loc[sol.params.carb_cost_df[sol.params.carb_cost_df.value != 0
                                               ].index.get_level_values(1).drop_duplicates(
                                                   )[0],'industry'])
    else:
        for s in sol.params.carb_cost_df[
                sol.params.carb_cost_df.value != 0
                ].index.get_level_values(1).drop_duplicates():
            if sector_map.loc[s,'industry'] not in sectors:
                sectors.append(sector_map.loc[s,'industry'])
    emissions.append(sol.run.emissions)
           
emissions = np.array(emissions)

#%%
import seaborn as sns

sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 25})
plt.rcParams['text.usetex'] = False

#%%

fig,ax = plt.subplots(figsize = (16,12))

bars = ax.barh(sectors,(baseline.co2_prod_np.sum() - emissions)*100/baseline.co2_prod_np.sum())

ax.bar_label(bars,
              labels=sectors,
              # rotation=90,
              label_type = 'edge',
              padding=5,
                # color='red',
                fontsize=15,
              zorder=99)

ax.axvline(x=(baseline.co2_prod_np.sum() - emissions[-1])*100/baseline.co2_prod_np.sum(),
           color='red')
ax.set_xlabel('Emissions reduction (%)')
ax2 = ax.twiny()

ax2.scatter(x=runs.utility,y=range(runs.shape[0]),
            color=sns.color_palette()[1],label='Real income')
ax2.grid(None)
ax2.legend(loc='upper center')

plt.show()

#%%

fig,ax = plt.subplots(figsize = (16,12))

bars = ax.barh(sectors,(baseline.co2_prod_np.sum() - emissions)*100/baseline.co2_prod_np.sum())

ax.bar_label(bars,
              labels=sectors,
              # rotation=90,
              label_type = 'edge',
              padding=5,
                # color='red',
                fontsize=15,
              zorder=99)

ax.axvline(x=(baseline.co2_prod_np.sum() - emissions[-1])*100/baseline.co2_prod_np.sum(),
           color='red')
ax.set_xlim([27,28])
ax.set_xlabel('Emissions reduction (%)')

plt.show()


#%% compute trade

b = baseline
N = b.country_number
S = b.sector_number

trade = baseline.iot.groupby(level=[0,1,2]).sum()
trade['cons'] = baseline.cons.value
trade['baseline'] = trade.value + trade.cons
trade = trade[['baseline']]
print("Computing trade")
l_trade = []
for i,sol in tqdm(enumerate(sols)):
    sol.compute_solution(baseline,inplace=True)
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

#%% 

dir_num = 63
results_path = 'results/'
path = results_path+baseline.year+'_'+str(dir_num)
runs_path = path+'/runs.csv'
    
runs = pd.read_csv(runs_path)

import treatment_funcs as t

sols = [t.sol(run[1],results_path,data_path) for run in tqdm(runs.iterrows())]
baseline = d.baseline(2018, data_path)

#%%

sols_by_taxed_sectors = {}

for sol in tqdm(sols):
    # sol.compute_solution(baseline,inplace=True)
    taxed_sectors = str(sol.params.taxed_sectors)
    if taxed_sectors in sols_by_taxed_sectors.keys():
        sols_by_taxed_sectors[taxed_sectors].append(sol)
    else:
        sols_by_taxed_sectors[taxed_sectors] = [sol]


#%%

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
    # l_E = []
    # l_em_reduc = []
    # l_em_reduc_real = []
    for i in tqdm(range(1,len(sols_by_carb_cost))):
    # for i in tqdm(range(1,2)):
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
        
        # em_reduc = (np.einsum('isj,is->',
        #                      trade_cf,
        #                      e)-\
        #             np.einsum('isj,is->',
        #                       trade_baseline,
        #                       e))/np.einsum('isj,is->',
        #                                 trade_baseline,
        #                                 e)

        # em_reduc = (np.einsum('isj,is->',
        #                      trade_cf,
        #                      e)-\
        #             np.einsum('isj,is->',
        #                       trade_baseline,
        #                       e))/np.einsum('isj,is->',
        #                                 trade_baseline,
        #                                 e)
        
        # print(em_reduc)

        l_term_1.append(term_1)
        l_term_2.append(term_2)
        l_term_3.append(term_3)
        # l_E.append( (sol_b.output.value.values*e.ravel() ).sum()/1e6 )
        
        # l_em_reduc.append(em_reduc)
        # l_em_reduc_real.append(sol_cf.co2_prod.value.sum())
        
    l_term_1 = np.array(l_term_1)
    l_term_2 = np.array(l_term_2)
    l_term_3 = np.array(l_term_3)
    # l_E = np.array(l_E)
    # l_em_reduc = np.array(l_em_reduc)
    
    d_term = {
        'term_1':l_term_1,
        'term_2':l_term_2,
        'term_3':l_term_3,
              }
    
    d_term_summed = {key: [v.sum() for v in l_term] for key, l_term in d_term.items()}
    
    # sol_last = sols_by_carb_cost[-1].compute_solution(baseline,with_np_arrays=True)
    
    # delta_E_R = sol_last.co2_prod.value.sum() - baseline.co2_prod.value.sum()
    # delta_E = ( sol_last.output.value.values*e.ravel() ).sum()/1e6 - baseline.co2_prod.value.sum()
    
    # # print(delta_E_R)
    # # print(delta_E)
    
    # # pi = {key: np.einsum('c,c->',
    # #                  np.array(l_term),
    # #                  l_E
    # #                  )*delta_E_R/delta_E/baseline.co2_prod.value.sum() for key, l_term in d_term_summed.items()
    # #       }
    # pi = {key: np.einsum('c,c->',
    #                   np.array(l_term),
    #                   l_E
    #                   ) for key, l_term in d_term_summed.items()
    #       }
    
    # pi_1 = np.einsum('c,c->',
    #                   np.array(d_term_summed['term_1']),
    #                   np.array([(sol.output.value.values*e.ravel()).sum() for sol in sols_by_carb_cost[:-1]])
    #                   )*delta_E_R/delta_E/baseline.co2_prod.value.sum()
    # pi_2 = 0
    # pi_3 = 0 
    
    # return pi
    
    emiss_reduc_contrib = {}
    for term, l_term in d_term_summed.items():
        emiss_reduc_contrib[term] = np.array([l_term[k]*sols_by_carb_cost[k].run.emissions/baseline.co2_prod.value.sum() 
                                      for k in range(len(l_term))])
        
    # l_em_incr = np.array([l_em_reduc[k]*sols_by_carb_cost[k].run.emissions/baseline.co2_prod.value.sum() 
    #              for k in range(len(l_em_reduc))])
    
    cumul_terms = {key: np.array([value[:i].sum()
                          for i in range(len(value))]) for key, value in emiss_reduc_contrib.items()}
    
    return cumul_terms
    

# #%%

# test_count = str(EU+['USA'])

# test = compute_decomposition_at_100_for_list_of_sols(
#     sols_by_taxing_countries[test_count]
#     ,baseline)

# # emiss_reduc_contrib_new_version = test[0]
# # cumul_terms_new_version = test[1]
# # d_term_summed_new_version = test[2]

#%%

decompositions_by_taxed_sectors = {}

for k,sols_by_carb_cost in tqdm(sols_by_taxed_sectors.items()):    
    
    decompositions_by_taxed_sectors[k] = compute_decomposition_at_100_for_list_of_sols(sols_by_carb_cost,baseline)


#%%


from ast import literal_eval

sectors = []

# for i,sect_list in enumerate(pd.Series(decompositions_by_taxed_sectors.keys()).apply(literal_eval)):
for i,sect_list in enumerate(pd.Series(decompositions_by_taxed_sectors.keys()).apply(literal_eval)):
    if i == 0:
        sectors = [sect_list[0]]
    else:
        for s in sect_list:
            if s not in sectors:
                sectors.append(s)

sector_names = [sector_map.loc[s,'industry'] for s in sectors]

t1 = np.zeros(len(sectors))
t2 = np.zeros(len(sectors))
t3 = np.zeros(len(sectors))

idx_cc = -1
baseline_em = baseline.co2_prod.value.sum()
l_em_reduc= -(np.array([sols_by_taxed_sectors[k][idx_cc].run.emissions for k in sols_by_taxed_sectors]) - baseline_em)*100/baseline_em
# l_em_reduc = np.ones(len(list(sols_by_taxed_sectors.keys())))*100

for j,k in enumerate(decompositions_by_taxed_sectors.keys()):
    sum_terms = decompositions_by_taxed_sectors[k]['term_1'][idx_cc]+decompositions_by_taxed_sectors[k]['term_2'][idx_cc]+\
        decompositions_by_taxed_sectors[k]['term_3'][idx_cc]
    t1[j] = decompositions_by_taxed_sectors[k]['term_1'][idx_cc]*l_em_reduc[j]/sum_terms
    t2[j] = decompositions_by_taxed_sectors[k]['term_2'][idx_cc]*l_em_reduc[j]/sum_terms
    t3[j] = decompositions_by_taxed_sectors[k]['term_3'][idx_cc]*l_em_reduc[j]/sum_terms

sum_t = t1 + t2 + t3
    
fig,ax = plt.subplots(figsize = (16,12),dpi=288)


# sum_terms = np.array(d_term_summed['term_1'])+np.array(d_term_summed['term_2'])+np.array(d_term_summed['term_3'])

# t1 = np.array(d_term_summed['term_1'])*(baseline_em - emissions)*100/baseline_em/(sum_terms)
# t2 = np.array(d_term_summed['term_2'])*(baseline_em - emissions)*100/baseline_em/(sum_terms)
# t3 = np.array(d_term_summed['term_3'])*(baseline_em - emissions)*100/baseline_em/(sum_terms)

# bars = ax.barh(sectors,(baseline.co2_prod_np.sum() - emissions)*100/baseline.co2_prod_np.sum())
bars1 = ax.barh(sector_names,t1,label = 'Scale')
bars2 = ax.barh(sector_names,t2,left=t1, label = 'Composition sectors')
bars3 = ax.barh(sector_names,t3,left=t1+t2, label = 'Composition countries')

ax.bar_label(bars3,
              labels=sector_names,
              # rotation=90,
              label_type = 'edge',
              padding=5,
                # color='red',
                fontsize=15,
              zorder=99)

ax.bar_label(bars1,
              labels = [round(t*100/sum_t[i],1) for i,t in enumerate(t1)],
              # rotation=90,
              label_type = 'center',
              padding = 5,
              # color = sns.color_palette()[0],
              color = 'k',
              fontsize = 15,
              zorder = 99)
ax.bar_label(bars2,
              labels = [round(t*100/sum_t[i],1) for i,t in enumerate(t2)],
              # rotation=90,
              label_type = 'center',
              padding = 5,
              # color = sns.color_palette()[0],
              color = 'k',
              fontsize = 15,
              zorder = 99)
ax.bar_label(bars3,
              labels = [round(t*100/sum_t[i],1) for i,t in enumerate(t3)],
              # rotation=90,
              label_type = 'center',
              padding = 5,
              # color = sns.color_palette()[0],
              color = 'k',
              fontsize = 15,
              zorder = 99)

# ax.axvline(x=(baseline.co2_prod.value.sum() - 
#               np.array([sols_by_taxing_countries[k][idx_cc].run.emissions for k in sols_by_taxing_countries])[-1]
#               )*100/baseline.co2_prod.value.sum(),
#             color='red')
ax.axvline(x=l_em_reduc[-1],
            color='red')
ax.set_xlabel('Emissions reduction (%)')
# ax2 = ax.twiny()

# x = runs.utility.loc[runs.carb_cost == 1e-4]
# ax2.scatter(x=x,y=range(x.shape[0]),
#             color=sns.color_palette()[7],label='Real income')
# ax2.grid(None)
# ax2.legend(loc='upper center')
ax.legend(loc=(0,-0.2),title='Decomposition')

plt.tight_layout()

for save_format in ['eps','png','pdf']:
    plt.savefig('presentation_material/cp_estimate_allyears_world_va_prod_tax/incremental_tax_sectors.'+save_format,
                format=save_format)

plt.show()