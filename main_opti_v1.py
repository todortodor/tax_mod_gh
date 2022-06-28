#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 14:17:32 2022

@author: simonl
"""

main_path = './'
import sys
sys.path.append(main_path+'lib/')
import solver_funcs as s
import data_funcs as d
from tqdm import tqdm
import numpy as np

dir_num = 1
data_path = main_path+'data/'
results_path = 'results/'
eta = 4
sigma = 4
numeraire_type = 'wage'
numeraire_country = 'USA'

carb_cost_list = [1e-2]
taxed_countries_list = [None, ['CHN']]
taxed_sectors_list = [None]
specific_taxing_list = [None]
fair_tax_list = [False]

cases = d.build_cases(carb_cost_list,taxed_countries_list,taxed_sectors_list,
                specific_taxing_list,fair_tax_list)

years = [y for y in range(2018,2019)]

for y in years:
    
    year=str(y)
    
    baseline = d.baseline(year, data_path)
    
    baseline.num_scale(numeraire_type, numeraire_country, inplace = True)
    
    baseline.make_np_arrays(inplace = True)
    
    baseline.compute_shares_and_gammas(inplace = True)
    
    for simulation_case in tqdm(cases):
        
        params = d.params(eta,sigma,**simulation_case)
        params.num_scale_carb_cost(baseline.num, inplace = True)
        
        if not params.fair_tax:
            results = s.solve_E_p(params, baseline)
        
        if params.fair_tax:
            results = s.solve_fair_tax(params, baseline)
        
        #compute some aggregated solution quantities to write directly in runs report
        emissions_sol, utility, utility_countries = s.compute_emissions_utility(results, params, baseline)
        
        d.write_solution_csv(results,results_path,dir_num,emissions_sol,utility,params,baseline)
