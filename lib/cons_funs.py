import solver_funcs as s
import data_funcs as d
import treatment_funcs as t
import pandas as pd
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

import itertools
import seaborn as sns
import matplotlib.patches as mpatches

# 1. get the solutions from the model
def get_model_solutions(results_path, data_path, dir_num, year, carb_cost_list, eta_path,
              sigma_path, taxed_countries_list, taxing_countries_list,
              taxed_sectors_list, specific_taxing_list, fair_tax_list, years,
              model_id = -1):

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

    so = sols[model_id]
    ba = baselines[int(years[-1])]

    # get country and sector labels
    sectors = so.cons.reset_index()['row_sector'].unique()
    countries = so.cons.reset_index()['row_country'].unique()

    # get country and sector dimensions
    i = len(countries)
    s = len(sectors)
    k = len(sectors)

    # counterfactual consumption
    pq_is_C = np.array(so.cons.reset_index().query("col_country == 'USA'")['value']).reshape((i,s))

    # baseline consumption
    pq_is_B = np.array(ba.cons.reset_index().query("col_country == 'USA'")['value']).reshape((i,s))

    # counterfactual value added
    w_s_C = np.array(so.va.reset_index().query("col_country == 'USA'")['value']).reshape((s))

    # baseline value added
    w_s_B = np.array(ba.va.reset_index().query("col_country == 'USA'")['value']).reshape((s))

    # counterfactual iot
    iot_isk_C = np.array(so.iot.reset_index().query("col_country == 'USA'")['value']).reshape((i,s,k))

    # baseline iot
    iot_isk_B = np.array(ba.iot.reset_index().query("col_country == 'USA'")['value']).reshape((i,s,k))

    # get baseline CO2 intensity
    emint_is_B = np.array(ba.co2_intensity['value']).reshape((i,s))

    # get carbon cost
    carb_cost_is_C = np.array(so.params.carb_cost_df.reset_index().query("col_country == 'USA'")['value']).reshape((i,s))

    # get price change
    p_tilde_is = np.array(so.price['hat']).reshape((i,s))

    # get consumer price index change
    P_tilde_s = np.array(so.consumer_price_agg.reset_index().query("col_country == 'USA'")['hat']).reshape((s))

    # get sigmas
    sigmas = pd.read_csv('data\\elasticities\\' + so.run.sigma_path)
    sigma_s = np.array(sigmas['epsilon']).reshape((s))

    return sectors, countries, pq_is_C, pq_is_B, w_s_C, w_s_B, iot_isk_C, \
           iot_isk_B, emint_is_B, carb_cost_is_C, p_tilde_is, P_tilde_s, sigma_s






# scale CE data and disaggregate across origins
def prep_CEX(labels, dims, CE_data, pq_is_B, year):

    # get labels
    sectors = labels[0]
    countries = labels[1]
    groups = labels[2]

    # get dims
    s = int(dims[0])
    i = int(dims[1])
    n = int(dims[2])

    # prepare CEX data
    CE_data = CE_data[CE_data['year'] == year].filter(['icio','group','total_value'])
    CE_data['icio'] = CE_data['icio'].str[1:]
    sector_group_pairs = list(itertools.product(sectors,groups))
    cex = pd.DataFrame({'icio':[i[0] for i in sector_group_pairs],'group':[i[1] for i in sector_group_pairs]}).merge(CE_data, on=['icio','group'], how='left')
    cex['total_value'] = cex['total_value'].fillna(1e-20)

    pq_sn_CEX = np.array(cex['total_value']).reshape((s,n))

    # compute scaling factor
    scaling = pq_is_B.sum() / pq_sn_CEX.sum()
    print('The scaling factor ICIO/CEX = ' + str(scaling))

    # scale CE data
    pq_sn_CEX = pq_sn_CEX * scaling

    # disaggregate scaled CEX across origins
    one_over_pq_s_B = 1 / np.einsum('is->s',pq_is_B)
    pq_isn_B = np.einsum('sn,is,s -> isn', pq_sn_CEX, pq_is_B, one_over_pq_s_B)

    return pq_isn_B, pq_sn_CEX






# compute quintile-specific income change
def get_i_tilde_n(labels, dims, pq_sn_CEX, w_s_B, w_s_C, iot_isk_C, pq_is_C, emint_is_B, carb_cost_is_C):

    # get labels
    sectors = labels[0]
    countries = labels[1]
    groups = labels[2]

    # get dims
    s = int(dims[0])
    i = int(dims[1])
    n = int(dims[2])

    # quintile-specific income share
    alpha_n = np.sum(pq_sn_CEX, axis=0) / np.stack([pq_sn_CEX.sum()] * n)

    # baseline quintile-specific income
    I_n_B = alpha_n * w_s_B.sum()

    # counterfactual quintile-specific income
    tax_rev_is = emint_is_B * carb_cost_is_C * (pq_is_C + np.sum(iot_isk_C, axis=2))
    I_n_C = alpha_n * w_s_C.sum() + np.sum(tax_rev_is)/5

    # quintile-spicific income change
    I_tilde_n = I_n_C / I_n_B

    return I_tilde_n, I_n_B, I_n_C




# solve consumer problem at quintile level
def get_consumer_problem_solution(p_tilde_is, emint_is_B, carb_cost_is_C, sigma_s, P_tilde_s, I_tilde_n, pq_isn_B, pq_sn_CEX, I_n_C):

    # effective tax rate
    t_is = emint_is_B * carb_cost_is_C

    # solve problem
    q_tilde_isn = np.einsum('is,s,n->isn', (p_tilde_is*(1+t_is))**(-1*sigma_s), P_tilde_s**(sigma_s-1), I_tilde_n)

    # counterfactual consumption
    pq_isn_C = pq_isn_B * np.einsum('is,isn->isn', p_tilde_is, q_tilde_isn)

    # total counterfactual spending
    T_sn_C = np.einsum('isn,is->sn', pq_isn_C, 1+t_is)

    # tax paid (approach 1)
    tau_sn = np.einsum('is,isn->sn', t_is, pq_isn_C)

    # expenditure share
    one_over_cex_sum = 1 / np.sum(pq_sn_CEX, axis=0)
    beta_sn = np.einsum('sn,n->sn',pq_sn_CEX, one_over_cex_sum)
    beta_sn[beta_sn<1e-10] = 0

    # tax_paid (approach 2)
    tau_sn_2 = np.einsum('sn,sn->sn', beta_sn, I_n_C - np.sum(pq_isn_C,axis=0))

    # compute total tax paid
    tax_n = np.sum(tau_sn, axis=0)

    return t_is, q_tilde_isn, pq_isn_C, T_sn_C, tau_sn, tau_sn_2, tax_n



def make_plots(labels, dims, pq_isn_B, I_n_B, I_tilde_n, pq_isn_C, I_n_C, t_is,
               T_sn_C, tau_sn, tau_sn_2, tax_n):

    # get labels
    sectors = labels[0]
    countries = labels[1]
    groups = labels[2]

    # get dims
    s = int(dims[0])
    i = int(dims[1])
    n = int(dims[2])

    # prepare dataframe with all sector/group combinations
    sector_group_pairs = list(itertools.product(sectors, groups))
    sector_group_df = pd.DataFrame({'icio': [i[0] for i in sector_group_pairs], 'group': [i[1] for i in sector_group_pairs]})

    # consumption share by income quintile across sectors (Baseline)
    # -> sum_i pq_isn_B / I_n_B
    s_share_by_n_B = sector_group_df.copy()
    one_over_InB = 1 / I_n_B
    s_share_by_n_B['sum_i pq_isn_B / I_n_B'] = np.einsum('isn,n->sn', pq_isn_B, one_over_InB).reshape(s*n)

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="icio", y="sum_i pq_isn_B / I_n_B", hue="group", hue_order=hue_order,
                data=s_share_by_n_B, ci=None).set(title='Consumption share by income quintile across sectors (Baseline)')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.show()

    # quintile-specific income change
    income_change = pd.DataFrame({'group': groups, 'I_tilde_n': I_tilde_n * 100 - 100})
    plt.figure(figsize=(16, 8))
    x_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
               'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x='group', y='I_tilde_n', order=x_order,
                data=income_change).set(title="Income Change")
    plt.ylabel('change in %\n')
    plt.xlabel('')
    plt.show()

    # consumption share by income quintile across sectors (Counterfactal)
    # -> sum_i pq_isn_C (1+t_is) / I_n_C
    s_share_by_n_C = sector_group_df.copy()
    one_over_InC = 1 / I_n_C
    s_share_by_n_C['sum_i pq_isn_C*(1+t_is) / I_n_C'] = np.einsum('isn,is,n->sn', pq_isn_C, 1+t_is, one_over_InC).reshape(s * n)

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent',
                 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="icio", y="sum_i pq_isn_C*(1+t_is) / I_n_C", hue="group", hue_order=hue_order,
                data=s_share_by_n_C, ci=None).set(title='Consumption share by income quintile across sectors (Counterfactual)')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.show()

    # compare pq_sn_C and pq_sn_B consumption shares
    s_shares = s_share_by_n_B.merge(s_share_by_n_C, on=['icio','group'],how='left')
    s_shares['ratio'] = s_shares['sum_i pq_isn_C*(1+t_is) / I_n_C'] / s_shares['sum_i pq_isn_B / I_n_B']
    s_shares['ratio'][(s_shares['sum_i pq_isn_C*(1+t_is) / I_n_C'] < 1e-5) & (s_shares['sum_i pq_isn_B / I_n_B'] < 1e-5)] = 0.0
    s_shares['ratio-1'] = s_shares['ratio'] - 1
    s_shares['ratio-1'][(s_shares['sum_i pq_isn_C*(1+t_is) / I_n_C'] < 1e-5) & (s_shares['sum_i pq_isn_B / I_n_B'] < 1e-5)] = 0.0


    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="icio", y='ratio', hue="group", hue_order=hue_order,
                data=s_shares, ci=None).set(title='Consumption share counterfactual / Consumption share baseline')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.legend(loc='lower right')
    plt.show()

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="icio", y='ratio-1', hue="group", hue_order=hue_order,
                data=s_shares, ci=None).set(
        title='Consumption share counterfactual / Consumption share baseline - 1')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.show()

    # total spending (Baseline)
    T_sn_B = np.sum(pq_isn_B, axis=0)
    spending_B = sector_group_df.copy()
    spending_B['T_sn_B'] = T_sn_B.reshape(s*n)

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent',
                 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="icio", y="T_sn_B", hue="group", hue_order=hue_order,
                data=spending_B, ci=None).set(
        title='Total spending (Baseline)')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.show()

    # total spending (Counterfactual)
    spending_C =  sector_group_df.copy()
    spending_C['T_sn_C'] = T_sn_C.reshape(s*n)

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent',
                 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="icio", y="T_sn_C", hue="group", hue_order=hue_order,
                data=spending_C, ci=None).set(
        title='Total spending (Counterfactual)')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.show()

    # total spending ratio
    spending = spending_C.merge(spending_B, on=['icio','group'], how='left')
    spending['T_sn_C / T_sn_B - 1'] = spending['T_sn_C'] / spending['T_sn_B'] - 1

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent',
                 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="icio", y="T_sn_C / T_sn_B - 1", hue="group", hue_order=hue_order,
                data=spending, ci=None).set(
        title='Total spending counterfactual / total spending baseline - 1')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.show()

    # tax paid (approach 1)
    tax = sector_group_df.copy()
    tax['tau_sn'] = tau_sn.reshape(s*n)

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent',
                 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="icio", y="tau_sn", hue="group", hue_order=hue_order,
                data=tax, ci=None).set(
        title='Tax paid (approach 1)')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.show()

    # tax paid (approach 2)
    tax2 = sector_group_df.copy()
    tax2['tau_sn_2'] = tau_sn_2.reshape(s * n)

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent',
                 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="icio", y="tau_sn_2", hue="group", hue_order=hue_order,
                data=tax2, ci=None).set(
        title='Tax paid (approach 2)')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.show()

    # comparison approach 1 & 2
    taus = tax.merge(tax2, on=['icio','group'],how='left')
    taus['tau_sn / tau_sn_2'] = taus['tau_sn'] / taus['tau_sn_2']
    taus['tau_sn_2 / tau_sn'] = taus['tau_sn_2'] / taus['tau_sn']

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent',
                 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="icio", y="tau_sn / tau_sn_2", hue="group", hue_order=hue_order,
                data=taus, ci=None).set(title='Tax paid approach 1 / tax paid approach 2')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.show()

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent',
                 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="icio", y="tau_sn_2 / tau_sn", hue="group",
                hue_order=hue_order,
                data=taus, ci=None).set(
        title='Tax paid approach 2 / tax paid approach 1')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.show()

    # total tax paid (approach 1) / revenue by quintile  # remember: revenue = spending
    tax_rev = pd.DataFrame({'group':groups})
    tax_rev['tax_n / I_n_C'] = tax_n / I_n_C

    plt.figure(figsize=(16, 8))
    order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
            'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="group", y="tax_n / I_n_C", order=order, data=tax_rev, ci=None).set(
        title='Total tax paid approach 1 / total revenue')
    plt.show()

    tax_rev2 = pd.DataFrame({'group': groups})
    tax_rev2['tax_n_2 / I_n_C'] = np.sum(tau_sn_2, axis=0) / I_n_C

    plt.figure(figsize=(16, 8))
    order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
             'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="group", y="tax_n_2 / I_n_C", order=order, data=tax_rev2,
                ci=None).set(title='Total tax paid approach 2 / total revenue')
    plt.show()
