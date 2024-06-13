#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:04:22 2024

@author: slepot
"""

main_path = './'
import sys
sys.path.append(main_path+'lib/')
import solver_funcs as s
import data_funcs as d

# spec_tax = pd.DataFrame(index = pd.MultiIndex.from_product([d.get_country_list(),
#                                                             d.get_sector_list(),
#                                                             d.get_country_list()],
#                                                             names = ['row_country',
#                                                                     'row_sector',
#                                                                     'col_country']),
#                         columns = ['value'])
# spec_tax['value'] = 0

par = d.params(data_path='./data/',
             eta_path="cp_estimate_allyears.csv", 
             sigma_path="cp_estimate_allyears.csv",
             carb_cost = 1e-4, #in Mio$ per ton of CO2, so this is 100$  
             taxed_countries = None, 
             taxing_countries = None, 
             taxed_sectors = None,
             specific_taxing = None,
             fair_tax = False,
             pol_pay_tax = False,
             tax_scheme = 'consumer',
             tau_factor=1
             )

baseline = d.baseline(year=2018,
                      data_path='./data/',
                      exclude_direct_emissions=False)

baseline.num_scale(numeraire_type='wage', 
                   numeraire_country='WLD', 
                   inplace = True)

par.num_scale_carb_cost(baseline.num, 
                        inplace = True)

baseline.make_np_arrays(inplace = True)

baseline.compute_shares_and_gammas(inplace = True)

#%%

import solver_funcs as s

results = s.solve_one_loop(
                    params=par, 
                    baseline=baseline, 
                    vec_init = None, 
                    tol=1e-9, 
                    damping=5)

#%%

print(results)
emissions_sol, utility, utility_countries = s.compute_emissions_utility(results, par, baseline, autarky=False)

d.write_solution_csv(results,
                     './results/',
                     1000,
                     emissions_sol,
                     utility,
                     par,
                     baseline,
                     autarky=False)

#%%
import pandas as pd
import treatment_funcs as t

runs_path = 'results/2018_1000/runs.csv'
    
runs = pd.read_csv(runs_path,index_col=0)
sols = t.sol(runs.iloc[0],results_path='./results/',data_path='./data/')


#%%

# function to modify to make a "non-carbon" tax :

# In solver_funcs.py :

#     cons_eq_unit
#     iot_eq_unit
#     solve_one_loop
    
# In data_funcs.py :
    


