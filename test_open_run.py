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

#%%

fig, ax = plt.subplots(figsize = (12,8))

x = [so.run.taxed_countries for so in sols]
y = [so.co2_prod.value.sum() for so in sols]

plt.bar(x = np.arange(0,len(y)),tick_label = x, height = y)

plt.show()

#%%


va_old = pd.read_csv('/Users/simonl/Dropbox/Mac/Documents/taff/tax_mod/old_sol_8_2018/va_old.csv')
iot_old = pd.read_csv('/Users/simonl/Dropbox/Mac/Documents/taff/tax_mod/old_sol_8_2018/iot_old.csv')
cons_old = pd.read_csv('/Users/simonl/Dropbox/Mac/Documents/taff/tax_mod/old_sol_8_2018/cons_old.csv')
output_old = pd.read_csv('/Users/simonl/Dropbox/Mac/Documents/taff/tax_mod/old_sol_8_2018/output_old.csv')

so = sols[0]
ba = baselines[2018]

col_sum_old = va_old.new.values + iot_old.groupby(['col_country','col_sector']).new.sum().values*(1+0.0001*ba.co2_intensity.value.values)
row_sum_old = iot_old.groupby(['row_country','row_sector']).new.sum().values + cons_old.groupby(['row_country','row_sector']).new.sum().values

# np.allclose(col_sum_old,output_old.new.values)
max_diff_col = np.abs((col_sum_old-output_old.new.values)/output_old.new.values).mean()
max_diff_row = np.abs((row_sum_old-output_old.new.values)/output_old.new.values).mean()

col_sum_old_b = va_old.value.values + iot_old.groupby(['col_country','col_sector']).value.sum().values
row_sum_old_b = iot_old.groupby(['row_country','row_sector']).value.sum().values + cons_old.groupby(['row_country','row_sector']).value.sum().values

# np.allclose(col_sum_old,output_old.new.values)
max_diff_col_b = np.abs((col_sum_old_b-output_old.value.values)/output_old.value.values).mean()
max_diff_row_b = np.abs((row_sum_old_b-output_old.value.values)/output_old.value.values).mean()

col_sum_old_so = so.va.value.values + so.iot.groupby(['col_country','col_sector']).value.sum().values*(1+0.0001*ba.co2_intensity.value.values)
col_sum_old_so_untaxed = so.va.value.values + so.iot.groupby(['col_country','col_sector']).value.sum().values
row_sum_old_so = so.iot.groupby(['row_country','row_sector']).value.sum().values + so.cons.groupby(['row_country','row_sector']).value.sum().values

# np.allclose(col_sum_old,output_old.new.values)
max_diff_col_so = np.abs((col_sum_old_so-so.output.value.values)/so.output.value.values).mean()
max_diff_row_so = np.abs((row_sum_old_so-so.output.value.values)/so.output.value.values).mean()


col_sum_old_ba = ba.va.value.values + ba.iot.groupby(['col_country','col_sector']).value.sum().values
row_sum_old_ba = ba.iot.groupby(['row_country','row_sector']).value.sum().values + ba.cons.groupby(['row_country','row_sector']).value.sum().values

# np.allclose(col_sum_old,output_old.new.values)
max_diff_col_ba = np.abs((col_sum_old_ba-ba.output.value.values)/ba.output.value.values).max()
max_diff_row_ba = np.abs((row_sum_old_ba-ba.output.value.values)/ba.output.value.values).max()

#%%

fig, ax = plt.subplots(figsize=(12,8))


ax.scatter(x=so.output.value.values, y=(col_sum_old_so-so.output.value.values)*100/so.output.value.values,s=3,label='col solution')
# ax.scatter(x=so.output.value.values, y=(col_sum_old_so_untaxed-so.output.value.values)*100/so.output.value.values,s=3,label='col untaxed solution')
ax.scatter(x=ba.output.value.values, y=(col_sum_old_ba-ba.output.value.values)*100/ba.output.value.values,s=3,label='col baseline')
ax.scatter(x=so.output.value.values, y=(row_sum_old_so-so.output.value.values)*100/so.output.value.values,s=3,label='row solution')
ax.scatter(x=ba.output.value.values, y=(row_sum_old_ba-ba.output.value.values)*100/ba.output.value.values,s=3,label='row baseline')


# plt.yscale('log')
plt.xscale('log')
ax.legend()
ax.set_ylim([-10,50])

plt.show()

#%%

fig, ax = plt.subplots(figsize=(12,8))


ax.scatter(x=so.output.value.values, y=(row_sum_old_so-so.output.value.values)*100/so.output.value.values,s=3,label='solution')
ax.scatter(x=ba.output.value.values, y=(row_sum_old_ba-ba.output.value.values)*100/ba.output.value.values,s=3,label='baseline')

# plt.yscale('log')
plt.xscale('log')
ax.legend()
ax.set_ylim([-25,100])

plt.show()

#%%

c_balance_left = so.cons.groupby(['col_country']).value.sum().values + so.iot.groupby(['col_country']).value.sum().values
c_balance_right = ba.deficit.value.values + so.cons.groupby(['row_country']).value.sum().values + so.iot.groupby(['row_country']).value.sum().values

fig, ax = plt.subplots(figsize=(12,8))


ax.scatter(x=c_balance_left, y=(c_balance_left - c_balance_right)/c_balance_left,s=3,label='solution')
# ax.plot((c_balance_left - c_balance_right)/c_balance_left)
# ax.scatter(x=so.output.value.values, y=(row_sum_old_so-so.output.value.values)*100/so.output.value.values,s=3,label='solution')
# ax.scatter(x=ba.output.value.values, y=(row_sum_old_ba-ba.output.value.values)*100/ba.output.value.values,s=3,label='baseline')

# plt.yscale('log')
plt.xscale('log')
ax.legend()
# ax.set_ylim([-25,100])

plt.show()