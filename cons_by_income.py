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


# subscripts:
# - i = row_country
# - s = row_sector
# - k = col_sector
# - n = income quintile
# - C = Counterfactual
# - B = Baseline


# load US consumption data by income quintile
CE_data = pd.read_csv('data/US_data.csv')

# load sector names
sector_names = pd.read_csv('consumption_analysis/sector_labels_agg.txt', sep='\t')

# get relevant datasets from solutions
sectors, countries, pq_is_C, pq_is_B, w_C, w_B, iot_isk_C, iot_isk_B, \
emint_is_B, carb_cost_is_C, p_tilde_is, P_tilde_s, sigma_s, so, ba = \
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
I_tilde_n, I_n_B, I_n_C, tax_rev_is = get_i_tilde_n(labels, dims, pq_sn_CEX, w_B, w_C, iot_isk_C, pq_is_C, emint_is_B, carb_cost_is_C)

# solve consumer problem
t_is, q_tilde_isn, pq_isn_C, T_sn_C, tau_sn, tau_sn_2, tax_n, beta_sn, U_tilde_sn, U_tilde_n = \
    get_consumer_problem_solution(p_tilde_is, emint_is_B, carb_cost_is_C, sigma_s,
                                  P_tilde_s, I_tilde_n, pq_isn_B, pq_sn_CEX, I_n_C)

# make plots for analysis
relevant_plots(labels, dims, sector_names, I_n_C, pq_isn_C, t_is, T_sn_C, emint_is_B, tax_rev_is,
                   I_tilde_n, tau_sn, tax_n, U_tilde_n, save_plots=True)


# - - - - - - - - - - - - - - - - - -
# other plots (not relevant anymore)
# plots
make_data_plots(labels, dims, sector_names, pq_isn_B, I_n_B, I_tilde_n, pq_isn_C, I_n_C, t_is, T_sn_C, tax_rev_is, tau_sn, tau_sn_2, tax_n, U_tilde_sn, U_tilde_n)
make_analysis_plots(labels, dims, sector_names, tau_sn, tau_sn_2, tax_n, tax_rev_is, I_n_C, pq_isn_C)
make_correlation_plots(labels, dims, sector_names, emint_is_B, pq_isn_C, tau_sn)




















