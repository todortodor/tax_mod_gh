#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 21:05:45 2022

@author: simonl
"""

main_path = './'
import sys
sys.path.append(main_path+"lib/")

from cons_funs import*


# notation
# - i = row_country
# - s = row_sector
# - k = col_sector
# - n = income quintile
# - C = Counterfactual
# - B = Baseline

# - p = price
# - q = quantity
# - w = value added
# -

# TODO: notes:: ICIO data has 42 sectors, CEX data 24

# load US consumption data by income quintile
CE_data = pd.read_csv('consumption_analysis\\US_data.csv')

# get relevant datasets from solutions
sectors, countries, pq_is_C, pq_is_B, w_s_C, w_s_B, iot_isk_C, iot_isk_B, \
emint_is_B, carb_cost_is_C, p_tilde_is, P_tilde_s, sigma_s = \
    get_model_solutions(results_path = main_path+'results/',
    data_path = main_path+'data/',
    dir_num = 1,
    year =2018,
    carb_cost_list = [1e-4],
    eta_path = ['uniform_elasticities_4.csv'],
    sigma_path = ['uniform_elasticities_4.csv'],
    taxed_countries_list = [None],
    taxing_countries_list = [None],
    taxed_sectors_list = [None],
    specific_taxing_list = [None],
    fair_tax_list = [False],
    years = [2018],
    model_id = -1)


# get dimensions
groups = CE_data['group'].unique()
labels = [sectors, countries, groups]
dims = [len(sectors), len(countries), len(groups)]

# scale and disaggregate CEX data across origins
pq_isn_B, pq_sn_CEX = prep_CEX(labels, dims, CE_data, pq_is_B, year=2018)

# quintile-spicific income change
I_tilde_n, I_n_B, I_n_C = get_i_tilde_n(labels, dims, pq_sn_CEX, w_s_B, w_s_C, iot_isk_C, pq_is_C, emint_is_B, carb_cost_is_C)

# solve consumer problem
t_is, q_tilde_isn, pq_isn_C, T_sn_C, tau_sn, tau_sn_2, tax_n = \
    get_consumer_problem_solution(p_tilde_is, emint_is_B, carb_cost_is_C, sigma_s,
                                  P_tilde_s, I_tilde_n, pq_isn_B, pq_sn_CEX, I_n_C)

# plots
make_plots(labels, dims, pq_isn_B, I_n_B, I_tilde_n, pq_isn_C, I_n_C, t_is,
               T_sn_C, tau_sn, tau_sn_2, tax_n)
