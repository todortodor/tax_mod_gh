#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 18:10:29 2023

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
import matplotlib.pyplot as plt

dir_num = 33
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
# taxed_countries_list = [None]
# taxing_countries_list = [None]
taxed_countries_list = [None]
test_taxed_countries_list = [None]
taxing_countries_list = [[]]
taxed_sectors_list = [None]
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

countries_to_test = baseline.country_list.copy()
# countries_to_test = ['ARG', 'AUS',
#  'AUT', 'BEL', 'BGR', 'BRA', 'BRN', 'CAN', 'CHE', 'CHL',
#  'COL', 'CRI', 'CYP', 'CZE', 'DEU', 'DNK', 'ESP', 'EST',
#  'FIN', 'FRA', 'GBR', 'GRC', 'HRV', 'HUN', 'IDN', 'IRL',
#  'ISL', 'ISR', 'ITA', 'JPN', 'KAZ', 'KHM', 'KOR', 'LAO',
#  'LTU', 'LVA', 'MAR', 'MEX', 'MLT', 'MMR', 'NLD', 'NOR',
#  'NZL', 'PER', 'PHL', 'POL', 'PRT', 'ROU', 'RUS', 'SAU',
#  'SGP', 'SVK', 'SVN', 'SWE', 'THA', 'TUN', 'TUR','TWN',
#  'VNM', 'ZAF']
country_emissions = baseline.co2_prod_np.sum()

while len(countries_to_test)>0:
    winning_country = countries_to_test[0]
    country_emissions = baseline.co2_prod_np.sum()
    for country in tqdm(countries_to_test):
        test_taxing_countries_list = [taxing_countries_list[0] + [country]]
            
        cases = d.build_cases(eta_path,sigma_path,carb_cost_list,test_taxed_countries_list,
                              test_taxing_countries_list,
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
            
            if emissions_sol<country_emissions:
                print('new winning country ',country,', emissions ',emissions_sol)
                country_emissions = emissions_sol
                winning_country = country
    
    print('overall winning country ',winning_country)
    # taxed_sectors_list.append(winning_sector)
    taxing_countries_list = [taxing_countries_list[0] + [winning_country]]
    countries_to_test.remove(winning_country)
    
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

dir_num = 33
results_path = 'results/'
path = results_path+baseline.year+'_'+str(dir_num)
runs_path = path+'/runs.csv'
    
runs = pd.read_csv(runs_path,index_col=0)

import matplotlib.pyplot as plt

plt.bar(x = runs.index, height = 1-runs['emissions']/baseline.co2_prod_np.sum())
# plt.yscale('log')

#%%
import treatment_funcs as t

sols = [t.sol(run[1],results_path,data_path) for run in runs.iterrows()]
from ast import literal_eval


#%%

countries = []
emissions = []

for i,count_list in enumerate(runs.taxing_countries.apply(literal_eval)):
    if i == 0:
        countries.append(count_list[0])
    else:
        for c in count_list:
            if c not in countries:
                countries.append(c)
           
emissions = np.array(runs.emissions)

#%%
import seaborn as sns
fig,ax = plt.subplots(figsize = (15,20))

bars = ax.barh(countries,(baseline.co2_prod_np.sum() - emissions)*100/baseline.co2_prod_np.sum())

ax.bar_label(bars,
              labels=countries,
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

fig,ax = plt.subplots(figsize = (15,20))

bars = ax.barh(countries,(baseline.co2_prod_np.sum() - emissions)*100/baseline.co2_prod_np.sum())

ax.bar_label(bars,
              labels=countries,
              # rotation=90,
              label_type = 'edge',
              padding=5,
                # color='red',
                fontsize=13,
              zorder=99)

ax.axvline(x=(baseline.co2_prod_np.sum() - emissions[-1])*100/baseline.co2_prod_np.sum(),
           color='red')
ax.set_xlim([27,28])
ax.set_xlabel('Emissions reduction (%)')

plt.show()

#%%
taxing_countries_list = []
taxing_countries_list.append(EU)
taxing_countries_list.append(taxing_countries_list[-1]+['USA'])

for sol in sols:
    for country in sol.params.taxing_countries:
        if country not in taxing_countries_list[-1] and country != 'ROW':
            taxing_countries_list.append(taxing_countries_list[-1]+[country])

taxing_countries_list.append(taxing_countries_list[-1]+['ROW'])

#%%

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

dir_num = 34
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
# taxing_countries_list = taxing_countries_list
taxed_sectors_list = [None]
specific_taxing_list = [None]
fair_tax_list = [False]
pol_pay_tax_list = [False]

cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                      taxed_sectors_list,specific_taxing_list,fair_tax_list,pol_pay_tax_list,
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

dir_num = 34
results_path = 'results/'
path = results_path+baseline.year+'_'+str(dir_num)
runs_path = path+'/runs.csv'
    
runs = pd.read_csv(runs_path)

import matplotlib.pyplot as plt

# plt.bar(x = runs.index, height = 1-runs['emissions']/baseline.co2_prod_np.sum())
# plt.yscale('log')

#%%
import treatment_funcs as t

sols = [t.sol(run[1],results_path,data_path) for run in runs.iterrows()]
from ast import literal_eval


#%%

countries = []
emissions = []

for i,count_list in enumerate(runs.taxing_countries.apply(literal_eval)):
    if i == 0:
        countries.append('EU')
    else:
        for c in count_list:
            if c not in countries and c not in EU:
                countries.append(c)
           
emissions = np.array(runs.emissions)

#%%
import seaborn as sns
fig,ax = plt.subplots(figsize = (15,20))

bars = ax.barh(countries,(baseline.co2_prod_np.sum() - emissions)*100/baseline.co2_prod_np.sum())

ax.bar_label(bars,
              labels=countries,
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

fig,ax = plt.subplots(figsize = (15,20))

bars = ax.barh(countries,(baseline.co2_prod_np.sum() - emissions)*100/baseline.co2_prod_np.sum())

ax.bar_label(bars,
              labels=countries,
              # rotation=90,
              label_type = 'edge',
              padding=5,
                # color='red',
                fontsize=13,
              zorder=99)

ax.axvline(x=(baseline.co2_prod_np.sum() - emissions[-1])*100/baseline.co2_prod_np.sum(),
           color='red')
ax.set_xlim([27,28])
ax.set_xlabel('Emissions reduction (%)')

plt.show()

#%%

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

dir_num = 18
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
# taxed_countries_list = [None]
taxing_countries_list = [None]
taxed_sectors_list = [None]
specific_taxing_list = [None]
fair_tax_list = [False]
pol_pay_tax_list = [False]

cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                      taxed_sectors_list,specific_taxing_list,fair_tax_list,pol_pay_tax_list,
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

dir_num = 33
results_path = 'results/'
path = results_path+baseline.year+'_'+str(dir_num)
runs_path = path+'/runs.csv'
    
runs = pd.read_csv(runs_path)

import matplotlib.pyplot as plt

# plt.bar(x = runs.index, height = 1-runs['emissions']/baseline.co2_prod_np.sum())
# plt.yscale('log')

#%%
import treatment_funcs as t

sols = [t.sol(run[1],results_path,data_path) for run in runs.iterrows()]
from ast import literal_eval


#%%

countries = []
emissions = []

for i,count_list in enumerate(runs.taxed_countries.apply(literal_eval)):
    if i == 0:
        countries.append('EU')
    else:
        for c in count_list:
            if c not in countries and c not in EU:
                countries.append(c)
           
emissions = np.array(runs.emissions)

#%%
import seaborn as sns
fig,ax = plt.subplots(figsize = (15,20))

bars = ax.barh(countries,(baseline.co2_prod_np.sum() - emissions)*100/baseline.co2_prod_np.sum())

ax.bar_label(bars,
              labels=countries,
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

fig,ax = plt.subplots(figsize = (15,20))

bars = ax.barh(countries,(baseline.co2_prod_np.sum() - emissions)*100/baseline.co2_prod_np.sum())

ax.bar_label(bars,
              labels=countries,
              # rotation=90,
              label_type = 'edge',
              padding=5,
                # color='red',
                fontsize=13,
              zorder=99)

ax.axvline(x=(baseline.co2_prod_np.sum() - emissions[-1])*100/baseline.co2_prod_np.sum(),
           color='red')
ax.set_xlim([27,28])
ax.set_xlabel('Emissions reduction (%)')

plt.show()

#%%

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

dir_num = 15
data_path = main_path+'data/'
results_path = 'results/'

numeraire_type = 'wage'
numeraire_country = 'USA'

carb_cost_list = [None]
eta_path = ["cp_estimate_allyears.csv"]
sigma_path = ["cp_estimate_allyears.csv"]
# taxed_countries_list = [None]
# taxing_countries_list = [None]
taxed_countries_list = [None]
taxing_countries_list = [None]
taxed_sectors_list = [None]
spec_tax = pd.DataFrame(index = pd.MultiIndex.from_product([d.get_country_list(),
                                                            d.get_sector_list(),
                                                            d.get_country_list()],
                                                            names = ['row_country',
                                                                    'row_sector',
                                                                    'col_country']),
                        columns = ['value'])
spec_tax['value'] = 0
# # spec_tax.loc[spec_tax.query("row_country != col_country").index, 'value'] = 1
specific_taxing_list = [spec_tax]
# specific_taxing_list = [None]
fair_tax_list = [False]
pol_pay_tax_list = [False]
      
y=2018         
year=str(y)

baseline = d.baseline(year, data_path)

baseline.num_scale(numeraire_type, numeraire_country, inplace = True)

baseline.make_np_arrays(inplace = True)

baseline.compute_shares_and_gammas(inplace = True)

countries_to_test = baseline.country_list.copy()
# countries_to_test = ['ARG', 'AUS',
#  'AUT', 'BEL', 'BGR', 'BRA', 'BRN', 'CAN', 'CHE', 'CHL',
#  'COL', 'CRI', 'CYP', 'CZE', 'DEU', 'DNK', 'ESP', 'EST',
#  'FIN', 'FRA', 'GBR', 'GRC', 'HRV', 'HUN', 'IDN', 'IRL',
#  'ISL', 'ISR', 'ITA', 'JPN', 'KAZ', 'KHM', 'KOR', 'LAO',
#  'LTU', 'LVA', 'MAR', 'MEX', 'MLT', 'MMR', 'NLD', 'NOR',
#  'NZL', 'PER', 'PHL', 'POL', 'PRT', 'ROU', 'RUS', 'SAU',
#  'SGP', 'SVK', 'SVN', 'SWE', 'THA', 'TUN', 'TUR','TWN',
#  'VNM', 'ZAF']
country_emissions = baseline.co2_prod_np.sum()

while len(countries_to_test)>0:
    winning_country = countries_to_test[0]
    country_emissions = baseline.co2_prod_np.sum()
    for country in tqdm(countries_to_test):
        test_spec_tax_df = specific_taxing_list[0].copy()
        test_spec_tax_df.loc[country,:,country] = 1e-4
        test_specific_taxing_list = [test_spec_tax_df]
        
        cases = d.build_cases(eta_path,sigma_path,carb_cost_list,test_taxed_countries_list,
                              taxing_countries_list,
                              taxed_sectors_list,test_specific_taxing_list,
                              fair_tax_list,pol_pay_tax_list,
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
            
            if emissions_sol<country_emissions:
                print('new winning country ',country,', emissions ',emissions_sol)
                country_emissions = emissions_sol
                winning_country = country
    
    print('overall winning country ',winning_country)
    specific_taxing_list[0].loc[winning_country,:,winning_country] = 1e-4
    countries_to_test.remove(winning_country)
    
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

dir_num = 15
results_path = 'results/'
path = results_path+baseline.year+'_'+str(dir_num)
runs_path = path+'/runs.csv'
    
runs = pd.read_csv(runs_path)

import matplotlib.pyplot as plt

# plt.bar(x = runs.index, height = 1-runs['emissions']/baseline.co2_prod_np.sum())
# plt.yscale('log')

#%%

import treatment_funcs as t

sols = [t.sol(run[1],results_path,data_path) for run in runs.iterrows()]

#%%

countries = []
emissions = []

for i,sol in enumerate(sols):
    if i == 0:
        countries.append(sol.params.carb_cost_df[sol.params.carb_cost_df.value != 0
                                               ].index.get_level_values(0).drop_duplicates(
                                                   )[0])
    else:
        for c in sol.params.carb_cost_df[
                sol.params.carb_cost_df.value != 0
                ].index.get_level_values(0).drop_duplicates():
            if c not in countries:
                countries.append(c)
    emissions.append(sol.run.emissions)
           
emissions = np.array(emissions)

#%%

fig,ax = plt.subplots(figsize = (15,20))

bars = ax.barh(countries,(baseline.co2_prod_np.sum() - emissions)*100/baseline.co2_prod_np.sum())

ax.bar_label(bars,
              labels=countries,
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

fig,ax = plt.subplots(figsize = (15,20))

bars = ax.barh(countries,(baseline.co2_prod_np.sum() - emissions)*100/baseline.co2_prod_np.sum())

ax.bar_label(bars,
              labels=countries,
              # rotation=90,
              label_type = 'edge',
              padding=5,
                # color='red',
                fontsize=13,
              zorder=99)

ax.axvline(x=(baseline.co2_prod_np.sum() - emissions[-1])*100/baseline.co2_prod_np.sum(),
           color='red')
ax.set_xlim([23,24])
ax.set_xlabel('Emissions reduction (%)')

plt.show()

#%% 

# dir_num = 34
dir_num = 75
results_path = 'results/'
path = results_path+'2018_'+str(dir_num)
runs_path = path+'/runs.csv'
    
runs = pd.read_csv(runs_path)

import treatment_funcs as t

sols = [t.sol(run[1],results_path,data_path) for run in tqdm(runs.iterrows())]
baseline = d.baseline(2018, data_path)

#%%

sols_by_taxing_countries = {}

for sol in tqdm(sols):
    # sol.compute_solution(baseline,inplace=True)
    taxing_countries = str(sol.params.taxing_countries)
    if taxing_countries in sols_by_taxing_countries.keys():
        sols_by_taxing_countries[taxing_countries].append(sol)
    else:
        sols_by_taxing_countries[taxing_countries] = [sol]


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

#%%

decompositions_by_taxing_countries = {}

for k,sols_by_carb_cost in tqdm(sols_by_taxing_countries.items()):    
    
    decompositions_by_taxing_countries[k] = compute_decomposition_at_100_for_list_of_sols(sols_by_carb_cost,baseline)
    
#%%

from ast import literal_eval

countries = ['EU']
# emissions = []

countries = []

for i,count_list in enumerate(pd.Series(decompositions_by_taxing_countries.keys()).apply(literal_eval)):
    if i == 0:
        countries.append('EU')
    else:
        for c in count_list:
            if c not in countries and c not in EU:
                countries.append(c)

t1 = np.zeros(len(countries))
t2 = np.zeros(len(countries))
t3 = np.zeros(len(countries))

idx_cc = -1
baseline_em = baseline.co2_prod.value.sum()
l_em_reduc= -(np.array([sols_by_taxing_countries[k][idx_cc].run.emissions for k in sols_by_taxing_countries]) - baseline_em)*100/baseline_em
# l_em_reduc = np.ones(len(list(sols_by_taxing_countries.keys())))*100

for j,k in enumerate(decompositions_by_taxing_countries.keys()):
    sum_terms = decompositions_by_taxing_countries[k]['term_1'][idx_cc]+decompositions_by_taxing_countries[k]['term_2'][idx_cc]+\
        decompositions_by_taxing_countries[k]['term_3'][idx_cc]
    t1[j] = decompositions_by_taxing_countries[k]['term_1'][idx_cc]*l_em_reduc[j]/sum_terms
    t2[j] = decompositions_by_taxing_countries[k]['term_2'][idx_cc]*l_em_reduc[j]/sum_terms
    t3[j] = decompositions_by_taxing_countries[k]['term_3'][idx_cc]*l_em_reduc[j]/sum_terms

sum_t = t1 + t2 + t3
    
fig,ax = plt.subplots(figsize = (24,12),dpi = 288)


# sum_terms = np.array(d_term_summed['term_1'])+np.array(d_term_summed['term_2'])+np.array(d_term_summed['term_3'])

# t1 = np.array(d_term_summed['term_1'])*(baseline_em - emissions)*100/baseline_em/(sum_terms)
# t2 = np.array(d_term_summed['term_2'])*(baseline_em - emissions)*100/baseline_em/(sum_terms)
# t3 = np.array(d_term_summed['term_3'])*(baseline_em - emissions)*100/baseline_em/(sum_terms)

# bars = ax.barh(sectors,(baseline.co2_prod_np.sum() - emissions)*100/baseline.co2_prod_np.sum())
bars1 = ax.barh(countries,t1,label = 'Scale effect')
bars2 = ax.barh(countries,t2,left=t1, label = 'Sector effect')
bars3 = ax.barh(countries,t3,left=t1+t2, label = 'Countries effect')

ax.bar_label(bars3,
              labels=countries,
              # rotation=90,
              label_type = 'edge',
              padding=5,
                # color='red',
                fontsize=12,
              zorder=99)

ax.bar_label(bars1,
              labels = [round(t*100/sum_t[i],1) for i,t in enumerate(t1)],
              # rotation=90,
              label_type = 'center',
              padding = 5,
              # color = sns.color_palette()[0],
              color = 'k',
              fontsize = 12,
              zorder = 99)
ax.bar_label(bars2,
              labels = [round(t*100/sum_t[i],1) for i,t in enumerate(t2)],
              # rotation=90,
              label_type = 'center',
              padding = 5,
              # color = sns.color_palette()[0],
              color = 'k',
              fontsize = 12,
              zorder = 99)
ax.bar_label(bars3,
              labels = [round(t*100/sum_t[i],1) for i,t in enumerate(t3)],
              # rotation=90,
              label_type = 'center',
              padding = 5,
              # color = sns.color_palette()[0],
              color = 'k',
              fontsize = 12,
              zorder = 99)

# ax.axvline(x=(baseline.co2_prod.value.sum() - 
#               np.array([sols_by_taxing_countries[k][idx_cc].run.emissions for k in sols_by_taxing_countries])[-1]
#               )*100/baseline.co2_prod.value.sum(),
#             color='red')
ax.axvline(x=l_em_reduc[-1],
            color='red')
ax.set_xlabel('Emissions reduction (%)')
# ax2 = ax.twiny()

# # ax2.scatter(x=runs.utility,y=range(runs.shape[0]),
# #             color=sns.color_palette()[5],label='Real income')
# ax2.grid(None)
# ax2.legend(loc='upper center')
# ax.legend(loc=(0.4,-0.2))

ax.legend(loc=(0,-0.2),title='Decomposition')

plt.tight_layout()

# for save_format in ['eps','png','pdf']:
#     plt.savefig('presentation_material/cp_estimate_allyears_world_va_prod_tax_with_cba/incremental_tax_countries.'+save_format,
#                 format=save_format)

plt.show()

#%%

# t3_consumer = t3.copy()
# t3_consumer_contrib = [t*100/sum_t[i] for i,t in enumerate(t3)]
# sum_terms_consumer = sum_t.copy()
# t3_producer = t3.copy()
# t3_producer_contrib = [t*100/sum_t[i] for i,t in enumerate(t3)]
# sum_terms_producer = sum_t.copy()
# t3_producer_cba = t3.copy()
# t3_producer_cba_contrib = [t*100/sum_t[i] for i,t in enumerate(t3)]
# sum_terms_producer_cba = sum_t.copy()

#%%

fig,ax = plt.subplots(figsize = (20,12),dpi = 288)
lw=3
# ax.plot(countries,t3_consumer,label='Consumer tax',lw=lw)
# ax.plot(countries,t3_producer,label='Producer tax',lw=lw)
# ax.plot(countries,t3_producer_cba,label='Producer tax with carbon border adjustement',lw=lw)
# ax.plot(countries,t3_consumer,label='Consumer tax',lw=lw)
ax.plot(countries,np.array(t3_producer_contrib)-np.array(t3_consumer_contrib),label='Producer tax',lw=lw)
ax.plot(countries,np.array(t3_producer_cba_contrib)-np.array(t3_consumer_contrib),label='Producer tax with carbon border adjustement',lw=lw)
plt.legend(loc='lower right',fontsize = 20)
ax.set_ylabel('Composition country contribution',fontsize = 20)
plt.title('Deviation from consumer tax',fontsize = 20)
plt.show()

#%%

fig,ax = plt.subplots(figsize = (20,12),dpi = 288)
lw=3
ax.plot(countries,sum_terms_consumer,label='Consumer tax',lw=lw)
ax.plot(countries,sum_terms_producer,label='Producer tax',lw=lw)
ax.plot(countries,sum_terms_producer_cba,label='Producer tax with carbon border adjustement',lw=lw)
plt.legend(loc='lower right',fontsize = 20)
ax.set_ylabel('Emissions reduction (Gt)',fontsize = 20)
plt.show()
