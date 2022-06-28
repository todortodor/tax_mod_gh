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
dir_num = 1
year = 2018

# test = pd.read_csv(t.dir_path(results_path,year,dir_num)+'/runs.csv', 
#                     index_col = 0, 
#                     keep_default_na=False,
#                     ).replace('',None)
# run = test.iloc[0]
# baseline = d.baseline(year, data_path)
# sol = t.sol(run, results_path).compute_solution(baseline).compute_hat(baseline)

carb_cost_list = [1e-4]
taxed_countries_list = [None, ['CHN'],['USA'], ['DEU', 'FRA', 'ITA', 'ESP', 'PRT', 'GRC'], ['CHN', 'USA', 'DEU', 'FRA', 'ITA', 'ESP', 'PRT', 'GRC']]
taxed_sectors_list = [None]
specific_taxing_list = [None]
fair_tax_list = [False]
cases = d.build_cases(carb_cost_list,taxed_countries_list,taxed_sectors_list,
                specific_taxing_list,fair_tax_list)

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
                                                      drop_duplicate_runs=True)

#%%

fig, ax = plt.subplots(figsize = (12,8))

x = [so.run.taxed_countries for so in sols]
y = [so.co2_prod.value.sum() for so in sols]

plt.bar(x = np.arange(0,len(y)),tick_label = x, height = y)

plt.show()