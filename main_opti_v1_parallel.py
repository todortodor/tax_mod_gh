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
# from deco import *
import time
from time import perf_counter
import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing import Manager
import concurrent.futures

dir_num = 6
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

# carb_cost_list = np.append(np.linspace(0,2.5e-4,251),np.linspace(2.5e-4,1e-3,76)[1:])[46:]
carb_cost_list = np.linspace(0,1e-4,21)
# eta_path = ['elasticities_agg1.csv','elasticities_agg2.csv','uniform_elasticities_4.csv']
# sigma_path = ['elasticities_agg1.csv','elasticities_agg2.csv','uniform_elasticities_4.csv']
eta_path = ['elasticities_agg1.csv']
sigma_path = ['uniform_elasticities_4.csv']
# carb_cost_list = [4.6e-4]
taxed_countries_list = [None]
# taxing_countries_list = [None,EU,NAFTA,ASEAN,AANZFTA,APTA,EEA,MERCOSUR,
#                           ['USA'],['CHN'],
#                           EEA+NAFTA,EEA+ASEAN,EEA+APTA,EEA+AANZFTA,EEA+['USA'],EEA+['CHN'],
#                           NAFTA+APTA,NAFTA+MERCOSUR,
#                           APTA+AANZFTA,EU+NAFTA+['CHN'],EU+NAFTA+APTA]
taxing_countries_list = [None]
taxed_sectors_list = [None]
specific_taxing_list = [None]
fair_tax_list = [False]

cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                      taxed_sectors_list,specific_taxing_list,fair_tax_list)

years = [2018]
    
def one_run(data_tuple):
    baseline, data_path, results_path, dir_num, simulation_case = data_tuple
    print(simulation_case['carb_cost'], flush=True)
    params = d.params(data_path, **simulation_case)
    params.num_scale_carb_cost(baseline.num, inplace = True)
    
    if not params.fair_tax:
        results = s.solve_E_p(params, baseline)
    
    if params.fair_tax:
        results = s.solve_fair_tax(params, baseline)
    
    #compute some aggregated solution quantities to write directly in runs report
    emissions_sol, utility, utility_countries = s.compute_emissions_utility(results, params, baseline)
    
    d.write_solution_csv(results,results_path,dir_num,emissions_sol,utility,params,baseline)
    # return simulation_case['carb_cost']
     

# for y in years:
y=2018         
year=str(y)

baseline = d.baseline(year, data_path)

baseline.num_scale(numeraire_type, numeraire_country, inplace = True)

baseline.make_np_arrays(inplace = True)

baseline.compute_shares_and_gammas(inplace = True)
    
#%%
t1 = perf_counter()

# if __name__ == '__main__':   
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         executor.map(one_run, [(baseline, data_path, results_path, dir_num, simulation_case) for simulation_case in cases])

for simulation_case in tqdm(cases):
    
    
    params = d.params(data_path, **simulation_case)
    params.num_scale_carb_cost(baseline.num, inplace = True)
    
    if not params.fair_tax:
        results = s.solve_E_p(params, baseline)
    
    if params.fair_tax:
        results = s.solve_fair_tax(params, baseline)
    one_run((baseline, data_path, results_path, dir_num, simulation_case))
    
t2 = perf_counter()
print(t2-t1)