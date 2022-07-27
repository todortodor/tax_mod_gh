#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 14:31:59 2022

@author: simonl
"""

main_path = './'
import sys
sys.path.append(main_path+"lib/")
import solver_funcs as s
import data_funcs as d
import treatment_funcs as t
import pandas as pd
# from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt


results_path = main_path+'results/'
data_path = main_path+'data/'

aid = pd.read_csv(data_path+'oecd_international_aid.csv')
# aid = aid[aid['SUBJECT'] == 'ODAFLOWS']['LOCATION',]

dir_num = 5
eta_path = ['uniform_elasticities_4.csv']
sigma_path = ['uniform_elasticities_4.csv']
carb_cost_list = [1e-4]
taxed_countries_list = [None]
taxing_countries_list = [None]
taxed_sectors_list = [None]
specific_taxing_list = [None]
fair_tax_list = [False,True]

cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                      taxed_sectors_list,specific_taxing_list,fair_tax_list)

years = [y for y in range(1995,2019)]

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