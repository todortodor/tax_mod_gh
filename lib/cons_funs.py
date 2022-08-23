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

sns.set(font_scale=1.5)
sns.set_style("whitegrid")

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
    pq_is_C = np.array(so.cons.xs('USA', level='col_country')).reshape((i,s))

    # baseline consumption
    pq_is_B = np.array(ba.cons.xs('USA', level='col_country')).reshape((i,s))

    # counterfactual value added
    w_C = np.array(so.va.xs('USA', level='col_country')).sum()
    # adjust value added by deficit
    w_C = w_C + np.array(ba.deficit.xs('USA'))

    # baseline value added
    w_B = np.array(ba.va.xs('USA', level='col_country')).sum()
    # adjust value added by deficit
    w_B = w_B + np.array(ba.deficit.xs('USA'))

    # counterfactual iot
    iot_isk_C = np.array(so.iot.xs('USA', level='col_country')).reshape((i,s,k))

    # baseline iot
    iot_isk_B = np.array(ba.iot.xs('USA', level='col_country')).reshape((i,s,k))

    # get baseline CO2 intensity
    emint_is_B = np.array(ba.co2_intensity['value']).reshape((i,s))

    # get carbon cost
    carb_cost_is_C = np.array(so.params.carb_cost_df.xs('USA', level='col_country')).reshape((i,s))

    # get price change
    p_tilde_is = np.array(so.price['hat']).reshape((i,s))

    # get consumer price index change
    P_tilde_s = np.array(so.consumer_price_agg.xs('USA', level='col_country')).reshape((s))

    # get sigmas
    sigmas = pd.read_csv('data/elasticities/' + so.run.sigma_path)
    sigma_s = np.array(sigmas['epsilon']).reshape((s))

    return sectors, countries, pq_is_C, pq_is_B, w_C, w_B, iot_isk_C, \
           iot_isk_B, emint_is_B, carb_cost_is_C, p_tilde_is, P_tilde_s, sigma_s, so, ba






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
def get_i_tilde_n(labels, dims, pq_sn_CEX, w_B, w_C, iot_isk_C, pq_is_C, emint_is_B, carb_cost_is_C):

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
    I_n_B = alpha_n * w_B

    # counterfactual quintile-specific income
    tax_rev_is = emint_is_B * carb_cost_is_C * (pq_is_C + np.sum(iot_isk_C, axis=2))
    I_n_C = alpha_n * w_C + np.sum(tax_rev_is)/5

    # quintile-spicific income change
    I_tilde_n = I_n_C / I_n_B

    return I_tilde_n, I_n_B, I_n_C, tax_rev_is




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
    tau_sn_2 = np.einsum('sn,n->sn', beta_sn, I_n_C) - np.sum(pq_isn_C,axis=0)

    # compute total tax paid
    tax_n = np.sum(tau_sn, axis=0)

    # compute utility change
    sig1 = ((sigma_s - 1) / sigma_s).reshape((1,len(sigma_s),1))
    sig2 = (sigma_s / (sigma_s - 1)).reshape(len(sigma_s), 1)
    one_over_pq_sn = 1 / np.einsum('isn->sn', pq_isn_B)

    U_tilde_sn = np.einsum('isn,isn->sn', np.einsum('isn,sn->isn', pq_isn_B, one_over_pq_sn), q_tilde_isn ** sig1) ** sig2
    U_tilde_n = np.prod(np.power(U_tilde_sn, beta_sn), axis=0)

    return t_is, q_tilde_isn, pq_isn_C, T_sn_C, tau_sn, tau_sn_2, tax_n, beta_sn, U_tilde_sn, U_tilde_n




def relevant_plots(labels, dims, sector_names, I_n_C, pq_isn_C, t_is, T_sn_C, emint_is_B, tax_rev_is,
                   I_tilde_n, tau_sn, tax_n, U_tilde_n, save_plots=False):

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
    sector_group_df = pd.DataFrame({'icio': [i[0] for i in sector_group_pairs],
                                    'group': [i[1] for i in
                                              sector_group_pairs]})
    # add sector names
    sector_group_df = sector_group_df.merge(sector_names, on='icio', how='left')

    # define y label for value units
    value_label = 'value in thousand USD\n'


    # plot1:
    # Consumption shape by income quintile across sectors (Counterfactual)
    s_share_by_n_C = sector_group_df.copy()
    one_over_InC = 1 / I_n_C
    s_share_by_n_C['sum_i pq_isn_C*(1+t_is) / I_n_C'] = np.einsum('isn,is,n->sn', pq_isn_C, 1+t_is, one_over_InC).reshape(s * n)*100

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent',
                 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="industry", y="sum_i pq_isn_C*(1+t_is) / I_n_C", hue="group", hue_order=hue_order,
                data=s_share_by_n_C, ci=None).set(title='Consumption share by income quintile across sectors\n')
    plt.xlabel('')
    plt.ylabel('Share in % of total consumption expenditure\n by income quintile\n', fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.legend(title='', fontsize=12)
    if save_plots:
        plt.savefig('fair_tax_results/consumption_patterns.png', dpi=300)
    plt.show()


    # plot 2:
    # total spending (Counterfactual)
    spending_C = sector_group_df.copy()
    spending_C['T_sn_C'] = T_sn_C.reshape(s * n)/1e6
    spending_C.loc[spending_C['T_sn_C'] < 1e-15, 'T_sn_C'] = 0

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent',
                 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="industry", y="T_sn_C", hue="group", hue_order=hue_order,
                data=spending_C, ci=None).set(
        title='Total spending (after tax)\n')
    plt.xlabel('')
    plt.ylabel('Value in billion USD\n')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.legend(title='', fontsize=12)
    if save_plots:
        plt.savefig('fair_tax_results/total_spending.png', dpi=300)
    plt.show()


    # plot 3:
    # total emissions
    emissions_isn = np.einsum('isn,is->isn', pq_isn_C/1000, emint_is_B)
    emissions = sector_group_df.copy()
    emissions['emissions_isn'] = np.einsum('isn,is->sn', pq_isn_C/1000, emint_is_B).reshape(s*n) / 1e6 # emissions in tons -> megatons

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent',
                 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="industry", y="emissions_isn", hue="group", hue_order=hue_order,
                data=emissions, ci=None).set(title='Total emissions\n')
    plt.xlabel('')
    plt.ylabel('Value in Megatons of CO2eq\n')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.legend(title='', fontsize=12)
    if save_plots:
        plt.savefig('fair_tax_results/emissions.png', dpi=300)
    plt.show()


    # plot 4:
    # total emissions
    emissions_n = pd.DataFrame({'group': groups, 'emissions_n': np.einsum('isn->n', emissions_isn)/1e6})
    plt.figure(figsize=(16, 8))
    x_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
               'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x='group', y='emissions_n', order=x_order, data=emissions_n).set(title='Total emissions\n')
    plt.xlabel('')
    plt.ylabel('Value in Megatons of CO2eq\n')
    if save_plots:
        plt.savefig('fair_tax_results/total_emissions.png', dpi=300)
    plt.show()


    # plot 5:
    # compute weighted average of sector emission intensity
    weight = np.sum(pq_isn_C, axis=-1) / np.einsum('isn->s', pq_isn_C)
    emint_s = np.sum(emint_is_B * weight, axis=0)

    emint = sector_names.copy()
    emint['emint_s'] = emint_s

    plt.figure(figsize=(16, 8))
    ax = sns.barplot(x="industry", y="emint_s", data=emint, color=sns.color_palette("tab10")[5], ci=None)
    ax.text(x=0.5, y=1.2, s='Weighted average of emission intensity by sector', fontsize=18,
            ha='center', va='bottom', transform=ax.transAxes)
    ax.text(x=0.5, y=1.125,
            s='weighted by origin-specific share in sectoral spending', fontsize=12,
            alpha=.75, ha='center', va='bottom', transform=ax.transAxes)

    plt.xlabel('')
    plt.ylabel('Emission intensity (tons of CO2eq / mio. USD)\n')
    plt.xticks(rotation=90)
    plt.tight_layout()
    if save_plots:
        plt.savefig('fair_tax_results/emint.png', dpi=300)
    plt.show()


    # plot 6:
    # total emission intensity
    emint_n = pd.DataFrame({'group': groups, 'emint_n': np.einsum('isn->n', emissions_isn) / np.einsum('isn->n', pq_isn_C/1000)})
    plt.figure(figsize=(16, 8))
    x_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
               'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x='group', y='emint_n', order=x_order,
                data=emint_n).set(title='Total emission intensity\n')
    plt.xlabel('')
    plt.ylabel('Emission intensity (tons of CO2eq / mio. USD)\n')
    if save_plots:
        plt.savefig('fair_tax_results/total_emint.png', dpi=300)
    plt.show()


    # plot 7:
    # total tax revenue by sector
    tax_revenue = sector_names.copy()
    tax_revenue['tax_rev_s'] = np.einsum('is->s', tax_rev_is)/1e3

    plt.figure(figsize=(16, 8))
    ax = sns.barplot(x="industry", y="tax_rev_s", color=sns.color_palette("tab10")[5], data=tax_revenue, ci=None)
    ax.text(x=0.5, y=1.2, s='Tax revenue',
            fontsize=18,
            ha='center', va='bottom', transform=ax.transAxes)
    ax.text(x=0.5, y=1.125,
            s='(incl. consumer and producer tax)',
            fontsize=12,
            alpha=.75, ha='center', va='bottom', transform=ax.transAxes)
    plt.xlabel('')
    plt.ylabel('Value in million USD\n')
    plt.xticks(rotation=90)
    plt.tight_layout()
    if save_plots:
        plt.savefig('fair_tax_results/tax_rev.png', dpi=300)
    plt.show()


    # plot 8:
    # quintile-specific income change
    income_change = pd.DataFrame({'group': groups, 'I_tilde_n': I_tilde_n * 100 - 100})
    plt.figure(figsize=(16, 8))
    x_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
               'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x='group', y='I_tilde_n', order=x_order, data=income_change).set(title='Income change\n')
    plt.ylabel('% of initial income\n')
    plt.xlabel('')
    if save_plots:
        plt.savefig('fair_tax_results/income_change.png', dpi=300)
    plt.show()


    # plot 9:
    # compute effective average tax rate
    pq_sn_C = np.einsum('isn->sn', pq_isn_C)
    tax_rate = sector_group_df.copy()
    tax_rate['tau_sn / pq_sn_C'] = (tau_sn / pq_sn_C).reshape(s*n)*100

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="industry", y="tau_sn / pq_sn_C", hue="group", hue_order=hue_order,
                data=tax_rate, ci=None).set(
        title='Effective tax rate\n')
    plt.xlabel('')
    plt.ylabel('Tax rate in % of consumption expenditure\n')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.legend(title='', fontsize=12)
    if save_plots:
        plt.savefig('fair_tax_results/tax_rate.png', dpi=300)
    plt.show()


    # plot 10:
    # compute effective average tax rate
    tax_rate_n = pd.DataFrame({'group': groups, 'tau_n / pq_n_C': np.einsum('sn->n', tau_sn) / np.einsum('isn->n', pq_isn_C)*100})

    plt.figure(figsize=(16, 8))
    x_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
               'Fourth 20 percent', 'Highest 20 percent']
    ax = sns.barplot(x='group', y='tau_n / pq_n_C', order=x_order, data=tax_rate_n)
    ax.set(title='Effective aggregate tax rate\n')
    plt.ylabel('Tax rate in % of consumption expenditure\n')
    plt.xlabel('')
    plt.tight_layout()
    if save_plots:
        plt.savefig('fair_tax_results/total_tax_rate.png', dpi=300)
    plt.show()


    # plot 11:
    # tax paid (approach 1)
    tax = sector_group_df.copy()
    tax['tau_sn'] = tau_sn.reshape(s*n) /1e3

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent',
                 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="industry", y="tau_sn", hue="group", hue_order=hue_order,
                data=tax, ci=None).set(
        title='Tax burden\n')
    plt.xlabel('')
    plt.ylabel('Value in million USD\n')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.legend(title='', fontsize=12)
    if save_plots:
        plt.savefig('fair_tax_results/tax_burden.png', dpi=300)
    plt.show()


    # plot 12:
    # tax paid aggregate(approach 1)
    tax_agg = pd.DataFrame({'group': groups, 'tau_n': np.einsum('sn->n', tau_sn)/1e3})

    plt.figure(figsize=(16, 8))
    x_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
               'Fourth 20 percent', 'Highest 20 percent']
    ax = sns.barplot(x='group', y='tau_n', order=x_order, data=tax_agg)
    ax.set(title='Total tax burden\n')
    ax.axhline((np.sum(tax_rev_is) / 5)/1e3, ls='--', color='red')
    ax.text(-.4,(np.sum(tax_rev_is) / 5)/1e3 * 1.02, 'Redistributed tax revenue',
            color='red')
    plt.xlabel('')
    plt.ylabel('Value in million USD\n')
    if save_plots:
        plt.savefig('fair_tax_results/total_tax_burden.png', dpi=300)
    plt.show()


    # plot 13:
    # get tax burden relative to income by sector
    tax_rev_sect = sector_group_df.copy()
    tax_rev_sect['tau_sn / I_n_C'] = (tau_sn / I_n_C).reshape(s*n)*100

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="industry", y="tau_sn / I_n_C", hue="group", hue_order=hue_order,
                data=tax_rev_sect, ci=None).set(title='Tax burden\n')
    plt.xlabel('')
    plt.ylabel('In % of quintile\'s total income\n')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.legend(title='', fontsize=12)
    if save_plots:
        plt.savefig('fair_tax_results/tax_burden_income.png', dpi=300)
    plt.show()


    # plot 14:
    # total tax paid (approach 1) / revenue by quintile
    tax_rev = pd.DataFrame({'group': groups})
    tax_rev['tax_n / I_n_C'] = tax_n / I_n_C * 100

    plt.figure(figsize=(16, 8))
    order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
             'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="group", y="tax_n / I_n_C", order=order, data=tax_rev,
                ci=None).set(
        title='Tax burden\n')
    plt.ylabel('In % of quintile\'s total income\n')
    plt.xlabel('')
    if save_plots:
        plt.savefig('fair_tax_results/total_tax_burden_income.png', dpi=300)
    plt.show()


    # plot 15:
    # utility change
    # plot change in utility by income group
    U_n = pd.DataFrame({'group': groups, 'U_tilde_n - 1': (U_tilde_n.reshape(n) - 1)*100})

    plt.figure(figsize=(16, 8))
    x_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
               'Fourth 20 percent', 'Highest 20 percent']
    ax = sns.barplot(x='group', y='U_tilde_n - 1', order=x_order, data=U_n)
    ax.set(title='Utility change\n')
    plt.xlabel('')
    plt.ylabel('Change in % of initial utility\n')
    if save_plots:
        plt.savefig('fair_tax_results/utility_change.png', dpi=300)
    plt.show()
















### not needed anymore




def make_data_plots(labels, dims, sector_names, pq_isn_B, I_n_B, I_tilde_n, pq_isn_C,
                    I_n_C, t_is, T_sn_C, tax_rev_is, tau_sn, tau_sn_2, tax_n,
                    U_tilde_sn, U_tilde_n):

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

    # add sector names
    sector_group_df = sector_group_df.merge(sector_names, on='icio', how='left')

    # consumption share by income quintile across sectors (Baseline)
    # -> sum_i pq_isn_B / I_n_B
    s_share_by_n_B = sector_group_df.copy()
    one_over_InB = 1 / I_n_B
    s_share_by_n_B['sum_i pq_isn_B / I_n_B'] = np.einsum('isn,n->sn', pq_isn_B, one_over_InB).reshape(s*n)

    plt.figure(figsize=(16,8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="industry", y="sum_i pq_isn_B / I_n_B", hue="group", hue_order=hue_order,
                data=s_share_by_n_B, ci=None).set(title='Consumption share by income quintile across sectors (Baseline)')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # quintile-specific income change
    income_change = pd.DataFrame({'group': groups, 'I_tilde_n': I_tilde_n * 100 - 100})
    plt.figure(figsize=(16, 8))
    x_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
               'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x='group', y='I_tilde_n', order=x_order, data=income_change).set(title='Income change')
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
    sns.barplot(x="industry", y="sum_i pq_isn_C*(1+t_is) / I_n_C", hue="group", hue_order=hue_order,
                data=s_share_by_n_C, ci=None).set(title='Consumption share by income quintile across sectors (Counterfactual)')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # compare pq_sn_C and pq_sn_B consumption shares
    s_shares = s_share_by_n_B.merge(s_share_by_n_C, on=['icio','industry','group'],how='left')
    s_shares['ratio'] = s_shares['sum_i pq_isn_C*(1+t_is) / I_n_C'] / s_shares['sum_i pq_isn_B / I_n_B']
    s_shares['ratio'][(s_shares['sum_i pq_isn_C*(1+t_is) / I_n_C'] < 1e-5) & (s_shares['sum_i pq_isn_B / I_n_B'] < 1e-5)] = 0.0
    s_shares['ratio-1'] = s_shares['ratio'] - 1
    s_shares['ratio-1'][(s_shares['sum_i pq_isn_C*(1+t_is) / I_n_C'] < 1e-5) & (s_shares['sum_i pq_isn_B / I_n_B'] < 1e-5)] = 0.0


    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="industry", y='ratio', hue="group", hue_order=hue_order,
                data=s_shares, ci=None).set(title='Consumption share counterfactual / Consumption share baseline')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="industry", y='ratio-1', hue="group", hue_order=hue_order,
                data=s_shares, ci=None).set(
        title='Consumption share counterfactual / Consumption share baseline - 1')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # total spending (Baseline)
    T_sn_B = np.sum(pq_isn_B, axis=0)
    spending_B = sector_group_df.copy()
    spending_B['T_sn_B'] = T_sn_B.reshape(s*n)
    spending_B['T_sn_B'][spending_B['T_sn_B'] < 1e-10] = 0

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent',
                 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="industry", y="T_sn_B", hue="group", hue_order=hue_order,
                data=spending_B, ci=None).set(
        title='Total spending (Baseline)')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # total spending (Counterfactual)
    spending_C =  sector_group_df.copy()
    spending_C['T_sn_C'] = T_sn_C.reshape(s*n)
    spending_C['T_sn_C'][spending_C['T_sn_C'] < 1e-10] = 0

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent',
                 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="industry", y="T_sn_C", hue="group", hue_order=hue_order,
                data=spending_C, ci=None).set(
        title='Total spending (Counterfactual)')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # total spending ratio
    spending = spending_C.merge(spending_B, on=['icio','industry','group'], how='left')
    spending['T_sn_C / T_sn_B - 1'] = spending['T_sn_C'] / spending['T_sn_B'] - 1

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent',
                 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="industry", y="T_sn_C / T_sn_B - 1", hue="group", hue_order=hue_order,
                data=spending, ci=None).set(
        title='Total spending counterfactual / total spending baseline - 1')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # total tax revenue by sector
    tax_revenue = sector_names.copy()
    tax_revenue['tax_rev_s'] = np.einsum('is->s', tax_rev_is)

    plt.figure(figsize=(16, 8))
    sns.barplot(x="industry", y="tax_rev_s", data=tax_revenue, ci=None).set(
        title='Tax revenue (producer + consumer)')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


    # tax paid (approach 1)
    tax = sector_group_df.copy()
    tax['tau_sn'] = tau_sn.reshape(s*n)

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent',
                 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="industry", y="tau_sn", hue="group", hue_order=hue_order,
                data=tax, ci=None).set(
        title='Tax paid (approach 1)')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # tax paid (approach 2)
    tax2 = sector_group_df.copy()
    tax2['tau_sn_2'] = tau_sn_2.reshape(s * n)

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent',
                 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="industry", y="tau_sn_2", hue="group", hue_order=hue_order,
                data=tax2, ci=None).set(
        title='Tax paid (approach 2)')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # comparison approach 1 & 2
    taus = tax.merge(tax2, on=['icio','industry','group'],how='left')
    taus['tau_sn / tau_sn_2'] = taus['tau_sn'] / taus['tau_sn_2']
    taus['tau_sn_2 / tau_sn'] = taus['tau_sn_2'] / taus['tau_sn']

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent',
                 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="industry", y="tau_sn / tau_sn_2", hue="group", hue_order=hue_order,
                data=taus, ci=None).set(title='Tax paid approach 1 / tax paid approach 2')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent',
                 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="industry", y="tau_sn_2 / tau_sn", hue="group",
                hue_order=hue_order,
                data=taus, ci=None).set(
        title='Tax paid approach 2 / tax paid approach 1')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # plot change in utility by sector and income group
    U_sn = sector_group_df.copy()
    U_sn['U_tilde_sn - 1'] = U_tilde_sn.reshape(s * n) - 1

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="industry", y="U_tilde_sn - 1", hue="group",
                hue_order=hue_order,
                data=U_sn, ci=None).set(
        title='Utility')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # plot change in utility by income group
    U_n = pd.DataFrame(
        {'group': groups, 'U_tilde_n - 1': U_tilde_n.reshape(n) - 1})

    plt.figure(figsize=(16, 8))
    x_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
               'Fourth 20 percent', 'Highest 20 percent']
    ax = sns.barplot(x='group', y='U_tilde_n - 1', order=x_order, data=U_n)
    ax.set(title='Utility')
    plt.xlabel('')
    plt.show()







# analysis plots
def make_analysis_plots(labels, dims, sector_names, tau_sn, tau_sn_2, tax_n, tax_rev_is, I_n_C, pq_isn_C):
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
    sector_group_df = pd.DataFrame({'icio': [i[0] for i in sector_group_pairs],
                                    'group': [i[1] for i in
                                              sector_group_pairs]})

    # add sector names
    sector_group_df = sector_group_df.merge(sector_names, on='icio', how='left')



    # total tax paid by income quintile
    # tax paid aggregate(approach 1)
    tax_agg = pd.DataFrame({'group': groups, 'tau_n': np.einsum('sn->n', tau_sn)})

    plt.figure(figsize=(16, 8))
    x_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
               'Fourth 20 percent', 'Highest 20 percent']
    ax = sns.barplot(x='group', y='tau_n', order=x_order, data=tax_agg)
    ax.set(title='Total tax burden (approach 1)')
    ax.axhline(np.sum(tax_rev_is)/5, ls='--', color='red')
    ax.text(-.4, np.sum(tax_rev_is)/5*1.02, 'Redistributed tax revenue', color='red')
    plt.xlabel('')
    plt.show()

    # tax paid (approach 2)
    tax_agg_2 = pd.DataFrame({'group': groups, 'tau_n': np.einsum('sn->n', tau_sn_2)})

    plt.figure(figsize=(16, 8))
    x_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
               'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x='group', y='tau_n', order=x_order, data=tax_agg_2).set(
        title='Total tax burden (approach 2)')
    plt.xlabel('')
    plt.show()

    # total tax paid (approach 1) / revenue by quintile  # remember: revenue = spending
    tax_rev = pd.DataFrame({'group': groups})
    tax_rev['tax_n / I_n_C'] = tax_n / I_n_C

    plt.figure(figsize=(16, 8))
    order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
             'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="group", y="tax_n / I_n_C", order=order, data=tax_rev,
                ci=None).set(
        title='Total tax paid approach 1 / total revenue')
    plt.show()

    # total tax paid (approach 12) / revenue by quintile
    tax_rev2 = pd.DataFrame({'group': groups})
    tax_rev2['tax_n_2 / I_n_C'] = np.sum(tau_sn_2, axis=0) / I_n_C

    plt.figure(figsize=(16, 8))
    order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
             'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="group", y="tax_n_2 / I_n_C", order=order, data=tax_rev2,
                ci=None).set(title='Total tax paid approach 2 / total revenue')
    plt.show()

    # get tax burden relative to income by sector
    tax_rev_sect = sector_group_df.copy()
    tax_rev_sect['tau_sn / I_n_C'] = (tau_sn / I_n_C).reshape(s*n)

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="industry", y="tau_sn / I_n_C", hue="group", hue_order=hue_order,
                data=tax_rev_sect, ci=None).set(title='Tax paid / total group Income')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


    # compute effective average tax rate
    pq_sn_C = np.einsum('isn->sn', pq_isn_C)
    tax_rate = sector_group_df.copy()
    tax_rate['tau_sn / pq_sn_C'] = (tau_sn / pq_sn_C).reshape(s*n)

    plt.figure(figsize=(16, 8))
    hue_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
                 'Fourth 20 percent', 'Highest 20 percent']
    sns.barplot(x="industry", y="tau_sn / pq_sn_C", hue="group", hue_order=hue_order,
                data=tax_rate, ci=None).set(
        title='Effective average tax rate')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # compute effective average tax rate
    tax_rate_n = pd.DataFrame({'group': groups, 'tau_n / pq_n_C': np.einsum('sn->n', tau_sn) / np.einsum('isn->n', pq_isn_C)*100})

    plt.figure(figsize=(16, 8))
    x_order = ['Lowest 20 percent', 'Second 20 percent', 'Third 20 percent',
               'Fourth 20 percent', 'Highest 20 percent']
    ax = sns.barplot(x='group', y='tau_n / pq_n_C', order=x_order, data=tax_rate_n)
    ax.set(title='Effective tax rate')
    plt.ylabel('Tax rate in %\n')
    plt.xlabel('')
    plt.show()




def make_correlation_plots(labels, dims, sector_names, emint_is_B, pq_isn_C, tau_sn):

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
    sector_group_df = pd.DataFrame({'icio': [i[0] for i in sector_group_pairs],
                                    'group': [i[1] for i in
                                              sector_group_pairs]})

    # add sector names
    sector_group_df = sector_group_df.merge(sector_names, on='icio', how='left')


    # compute weighted average of sector emission intensity
    weight = np.sum(pq_isn_C, axis=-1) / np.einsum('isn->s', pq_isn_C)
    emint_s = np.sum(emint_is_B * weight, axis=0)

    emint = sector_names.copy()
    emint['emint_s'] = emint_s

    plt.figure(figsize=(16, 8))
    sns.barplot(x="industry", y="emint_s", data=emint, ci=None).set(
        title='Weighted average of emission intensity by sector')
    plt.xlabel('')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


    # add effective taxrate
    emint['tax_rate'] = np.sum(tau_sn, axis=1) / np.einsum('isn->s', pq_isn_C) * 100
    emint = emint.sort_values('tax_rate')

    plt.figure(figsize=(16,8))
    sns.scatterplot(x='tax_rate', y='emint_s', hue='industry', data=emint)
    plt.tight_layout()
    plt.xlabel('tax rate in %')
    plt.ylabel('emission intensity (tons / mio. USD)')
    plt.legend(ncol=4, fontsize=10)
    plt.show


    # add consumption
    emint = emint.sort_values('icio')
    emint['consumption'] = np.einsum('isn->s', pq_isn_C)
    emint_nonzero = emint[emint['consumption'] > 1e-10]
    emint_nonzero = emint_nonzero.sort_values('consumption')

    plt.figure(figsize=(16,8))
    sns.scatterplot(x='consumption', y='emint_s', hue='industry', data=emint_nonzero)
    plt.tight_layout()
    plt.xlabel('consumption in thousand USD')
    plt.ylabel('emission intensity (tons / mio. USD)')
    plt.legend(ncol=6, fontsize=10)
    plt.show












