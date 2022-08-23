import pandas as pd
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

import itertools
import seaborn as sns
import matplotlib.patches as mpatches


import solver_funcs as s
import data_funcs as d
import treatment_funcs as t


# 1) get solutions from model
def get_model_solutions(results_path, data_path, dir_num, year, carb_cost_list, eta_path,
              sigma_path, taxed_countries_list, taxing_countries_list,
              taxed_sectors_list, specific_taxing_list, fair_tax_list, years,
              sample_countries):

    cases = d.build_cases(eta_path, sigma_path, carb_cost_list,
                          taxed_countries_list, taxing_countries_list,
                          taxed_sectors_list, specific_taxing_list,
                          fair_tax_list)

    sols, baselines, relevant_runs, found_cases, not_found_cases = t.sol.load_sols(
        cases,
        years,
        dir_num,
        results_path,
        data_path,
        baselines=None,
        compute_sols=True,
        # compute_hats= True,
        return_not_found_cases=True,
        drop_duplicate_runs=True,
        keep='last')

    so = sols[-1]
    ba = baselines[int(years[-1])]

    # get country and sector labels
    sectors = so.cons.reset_index()['row_sector'].unique()
    countries = so.cons.reset_index()['row_country'].unique()

    # get country and sector dimensions
    i = len(countries)
    j = len(sample_countries)
    s = len(sectors)
    k = len(sectors)

    idx = pd.IndexSlice

    # counterfactual consumption
    pq_isj_C = np.array(so.cons.loc[idx[:, :, sample_countries], :]).reshape((i, s, j))

    # baseline consumption
    pq_isj_B = np.array(ba.cons.loc[idx[:, :, sample_countries], :]).reshape((i, s, j))

    # counterfactual value added
    w_j_C = np.sum(np.array(so.va.loc[idx[sample_countries, :], :]).reshape(j,s), axis=1)
    # adjust value added by deficit
    w_j_C = w_j_C + np.array(ba.deficit.loc[idx[sample_countries], :]).reshape(j)

    # baseline value added
    w_j_B = np.sum(np.array(ba.va.loc[idx[sample_countries, :], :]).reshape(j,s), axis=1)
    # adjust value added by deficit
    w_j_B = w_j_B + np.array(ba.deficit.loc[idx[sample_countries], :]).reshape(j)

    # counterfactual iot
    iot_isjk_C = np.array(so.iot.loc[idx[:, :, sample_countries, :], :]).reshape((i, s, j, k))

    # baseline iot
    iot_isjk_B = np.array(ba.iot.loc[idx[:, :, sample_countries, :], :]).reshape((i, s, j, k))

    # get baseline CO2 intensity
    emint_is_B = np.array(ba.co2_intensity['value']).reshape((i, s))

    # get carbon cost
    carb_cost_isj_C = np.array(so.params.carb_cost_df.loc[idx[:, :, sample_countries], :]).reshape((i, s, j))

    # get price change
    p_tilde_is = np.array(so.price['hat']).reshape((i, s))

    # get consumer price index change
    P_tilde_sj = np.array(so.consumer_price_agg.loc[idx[:, sample_countries], :]).reshape((s, j))

    # get sigmas
    sigmas = pd.read_csv('data/elasticities/' + so.run.sigma_path)
    sigma_s = np.array(sigmas['epsilon']).reshape((s))

    return sectors, countries, pq_isj_C, pq_isj_B, w_j_C, w_j_B, iot_isjk_C, \
           iot_isjk_B, emint_is_B, carb_cost_isj_C, p_tilde_is, P_tilde_sj, sigma_s, so, ba



# define helper function for dimensions
def getdims(dims):

    sectors = dims[0]
    countries = dims[1]
    groups = dims[2]
    sample = dims[3]
    sector_names = dims[4]

    s = len(sectors)
    k = len(sectors)
    i = len(countries)
    n = len(groups)
    j = len(sample)

    # prepare dataframe with all sector/group combinations
    sector_group_sample_pairs = list(itertools.product(sectors, groups, sample))
    sector_group_sample_df = pd.DataFrame({'icio': [i[0] for i in sector_group_sample_pairs],
                                           'group': [i[1] for i in sector_group_sample_pairs],
                                           'country': [i[2] for i in sector_group_sample_pairs]})

    # add sector names
    sector_group_sample_df = sector_group_sample_df.merge(sector_names, on='icio', how='left')

    return s, k, i, j, n, sector_group_sample_df



def disaggregate_data(dims, pq_isj_B, cons_eu):

    # dimensions
    s, k, i, j, n, sector_group_sample_df = getdims(dims)

    # prepare eurostat consumption data
    full_cons_eu = sector_group_sample_df.copy().merge(cons_eu, on=['icio','group','country'], how='left').fillna(0)
    pq_snj_eu = np.array(full_cons_eu.sort_values(['icio','group','country'], ignore_index=True)['consumption_mio_eur']).reshape(s,n,j)

    # compute scaling factor
    scaling = pq_isj_B.sum() / pq_snj_eu.sum()
    print(f'scaling factor = {np.round(scaling, decimals=2)}')

    # scale eurostat data
    pq_snj_eu = pq_snj_eu * scaling

    # disaggregate scaled data across origins
    one_over_pq_sj_B = 1 / np.einsum('isj->sj', pq_isj_B)
    pq_isnj_B = np.einsum('snj,isj,sj->isnj', pq_snj_eu, pq_isj_B, one_over_pq_sj_B)

    return pq_isnj_B, pq_snj_eu




def get_income_change(dims, pq_snj_eu, w_j_B, w_j_C, iot_isjk_C, pq_isj_C, emint_is_B, carb_cost_isj_C):

    # dimensions
    s, k, i, j, n, sector_group_sample_df = getdims(dims)

    # get quintile specific income share
    alpha_nj = np.sum(pq_snj_eu, axis=0) / np.sum(np.sum(pq_snj_eu, axis=0), axis=0, keepdims=True)

    # baseline quintile-specific income
    I_nj_B = np.einsum('nj,j->nj', alpha_nj, w_j_B)

    # counterfactual quintile-specific income
    tax_rev_isj = np.einsum('is,isj,isj->isj', emint_is_B, carb_cost_isj_C, (pq_isj_C + np.sum(iot_isjk_C, axis=3)))
    I_nj_C = np.einsum('nj,j->nj', alpha_nj, w_j_C) + np.einsum('isj->j', tax_rev_isj)

    # quintile specific income change
    I_tilde_nj = I_nj_C / I_nj_B

    return I_tilde_nj, I_nj_B, I_nj_C, tax_rev_isj





def get_consumer_problem_solution(dims, p_tilde_is, emint_is_B, carb_cost_isj_C, sigma_s, P_tilde_sj, I_tilde_nj, pq_isnj_B, pq_snj_eu, I_nj_C):

    # dimensions
    s, k, i, j, n, sector_group_sample_df = getdims(dims)

    # effective tax rate
    t_isj = np.einsum('is,isj->isj', emint_is_B, carb_cost_isj_C)

    sigma_sj = sigma_s.reshape(s, 1)
    sigma_isj = sigma_s.reshape(1, s, 1)

    # solve problem
    q_tilde_isnj = np.einsum('isj,sj,nj->isnj', (np.repeat(p_tilde_is[:, :, np.newaxis], j, axis=2)*(1+t_isj))**(-1*sigma_isj), P_tilde_sj**(sigma_sj-1), I_tilde_nj)

    # counterfactual consumption
    pq_isnj_C = pq_isnj_B * np.einsum('is,isnj->isnj', p_tilde_is, q_tilde_isnj)

    # total counterfactual spending
    T_snj_C = np.einsum('isnj,isj->snj', pq_isnj_C, 1+t_isj)

    # tax paid (approach 1)
    tau_snj = np.einsum('isj,isnj->snj', t_isj, pq_isnj_C)

    # expenditure share
    beta_snj = pq_snj_eu / np.sum(pq_snj_eu, axis=0, keepdims=True)
    beta_snj[beta_snj<1e-15] = 0

    # tax paid (approach 2)
    tau_snj_2 = np.einsum('snj,nj->snj', beta_snj, I_nj_C) - np.sum(pq_isnj_C, axis=0)

    # verify if approach 1 and approach 2 yield same results
    print(f'Overlap in tax paid (tau) between approach 1 and 2: {np.round(np.mean(np.round(tau_snj_2, decimals=2) == np.round(tau_snj, decimals=2))*100, 2)}%')

    # total tax paid by consumers
    tax_nj = np.sum(tau_snj, axis=0)

    # utility change
    sig1 = ((sigma_s - 1) / sigma_s).reshape((1, s, 1, 1))
    sig2 = (sigma_s / (sigma_s - 1)).reshape(s, 1, 1)
    cons_share = pq_isnj_C / np.sum(pq_isnj_B, axis=0, keepdims=True)
    cons_share = np.nan_to_num(cons_share)

    U_tilde_snj = np.sum(cons_share * q_tilde_isnj ** sig1, axis=0) ** sig2
    U_tilde_nj = np.prod(np.power(U_tilde_snj, beta_snj), axis=0)

    return t_isj, q_tilde_isnj, pq_isnj_C, T_snj_C, tau_snj, tax_nj, U_tilde_snj, U_tilde_nj




