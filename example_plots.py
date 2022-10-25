#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 21:05:45 2022

@author: simonl
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


results_path = main_path+'results/'
data_path = main_path+'data/'
# dir_num = [1,2,3,4] can be a list to look in multiple directories
dir_num = 200
year = 2018

# test = pd.read_csv(t.dir_path(results_path,year,dir_num)+'/runs.csv', 
#                     index_col = 0)
# run = test.iloc[-1]
# baseline = d.baseline(year, data_path)
# sol = t.sol(run, results_path).compute_solution(baseline)#.compute_hat(baseline)

# tax_test_1 = pd.DataFrame(index = pd.MultiIndex.from_product([d.get_country_list(), d.get_sector_list()],names = ['country','sector']),
#                           columns = ['value'],
#                           data = np.random.rand(len(d.get_country_list())*len(d.get_sector_list()))/1e3
#                           )

# tax_test_1 = pd.read_csv(results_path+'test1.csv',index_col = [0,1])

carb_cost_list = np.linspace(0,1e-4,11)
# carb_cost_list = [None]
eta_path = ['elasticities_agg1.csv','elasticities_agg2.csv','uniform_elasticities_4.csv']
sigma_path = ['elasticities_agg1.csv','elasticities_agg2.csv','uniform_elasticities_4.csv']
# eta_path = ['elasticities_agg1.csv']
# sigma_path = ['uniform_elasticities_4.csv']
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
# spec_tax = pd.DataFrame(index = pd.MultiIndex.from_product([d.get_country_list(),
#                                                             d.get_sector_list(),
#                                                             d.get_country_list()],
#                                                             names = ['row_country',
#                                                                     'row_sector',
#                                                                     'col_country']),
#                         columns = ['value'])
# spec_tax['value'] = 1e-4
# spec_tax.loc[:,'94T98',:] = 0.5e-4
# specific_taxing_list = [spec_tax]
fair_tax_list = [False]

cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                      taxed_sectors_list,specific_taxing_list,fair_tax_list)

years = [2018]

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