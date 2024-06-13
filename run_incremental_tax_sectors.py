#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:18:09 2023

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

data_path = main_path+'data/'
results_path = 'results/'

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

y=2018         
year=str(y)

baseline = d.baseline(year, data_path)

baseline.num_scale(numeraire_type, numeraire_country, inplace = True)

baseline.make_np_arrays(inplace = True)

baseline.compute_shares_and_gammas(inplace = True)

dir_num = 13
results_path = 'results/'
path = results_path+baseline.year+'_'+str(dir_num)
runs_path = path+'/runs.csv'
    
runs = pd.read_csv(runs_path)

import treatment_funcs as t

sols = [t.sol(run[1],results_path,data_path) for run in runs.iterrows()]

taxed_sectors_list = []

for sol in sols:
    for sector in sol.params.taxed_sectors:
        if len(taxed_sectors_list) == 0:
            taxed_sectors_list = [[sector]]
        if sector not in taxed_sectors_list[-1]:
            taxed_sectors_list.append(taxed_sectors_list[-1]+[sector])
#%%
dir_num = 76
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

# carb_cost_list = np.linspace(0,1e-4,101)
carb_cost_list = np.linspace(0,1e-4,11)
eta_path = ["cp_estimate_allyears.csv"]
sigma_path = ["cp_estimate_allyears.csv"]
taxed_countries_list = [None]
taxing_countries_list = [None]
specific_taxing_list = [None]
fair_tax_list = [False]
pol_pay_tax_list = [False]
tax_scheme_list = ['eu_style']

cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                      taxed_sectors_list,specific_taxing_list,fair_tax_list,pol_pay_tax_list,tax_scheme_list,
                      same_elasticities=True)
exclude_direct_emissions = False

print(dir_num,tax_scheme_list)

years = [2018]
      
for y in years:
    # y=2018         
    year=str(y)
    
    baseline = d.baseline(year, data_path, exclude_direct_emissions=exclude_direct_emissions)
    
    baseline.num_scale(numeraire_type, numeraire_country, inplace = True)
    
    baseline.make_np_arrays(inplace = True)
    
    baseline.compute_shares_and_gammas(inplace = True)
    
    vec_init = None
    
    for simulation_case in tqdm(cases):
        if simulation_case['eta_path'] == simulation_case['sigma_path']:

            params = d.params(data_path, **simulation_case)
            params.num_scale_carb_cost(baseline.num, inplace = True)
            
            if not params.fair_tax and not params.pol_pay_tax:
                results = s.solve_one_loop(params, baseline, vec_init = vec_init)
                vec_init = np.concatenate([results['E_hat'].ravel(),
                                            results['p_hat'].ravel(),
                                            results['I_hat'].ravel()] )
            
            if params.fair_tax:
                results = s.solve_fair_tax(params, baseline)
                
            if params.pol_pay_tax:
                results = s.solve_pol_pay_tax(params, baseline)
            
            #compute some aggregated solution quantities to write directly in runs report
            emissions_sol, utility, utility_countries = s.compute_emissions_utility(results, params, baseline)
            
            d.write_solution_csv(results,results_path,dir_num,emissions_sol,utility,params,baseline)