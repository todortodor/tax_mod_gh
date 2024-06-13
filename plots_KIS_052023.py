#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:52:57 2023

@author: slepot
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
import matplotlib.pylab as pylab
import seaborn as sns
from adjustText import adjust_text
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (25, 18),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')

# save_path = '/Users/slepot/Library/CloudStorage/Dropbox/Green Logistics/KIS/KIS_052023/KIS_0523_plots/'
save_path = 'presentation_material/cp_estimate_allyears_world_va_prod_tax/'
save_formats = ['eps','png','pdf']

#%% load solution

results_path = main_path+'results/'
data_path = main_path+'data/'

dir_num = [60,61]
year = 2018

carb_cost_list = [1e-4]
# eta_path = ['uniform_elasticities_4.csv']
# sigma_path = ['uniform_elasticities_4.csv']
eta_path = ['cp_estimate_allyears.csv']
sigma_path = ['cp_estimate_allyears.csv']

taxed_countries_list = [None]

taxing_countries_list = [None]
taxed_sectors_list = [None]
specific_taxing_list = [None]
fair_tax_list = [False,True]
pol_pay_tax_list = [False,True]
tax_scheme_list=['producer']
tau_factor_list=[1]

cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                      taxed_sectors_list,specific_taxing_list,fair_tax_list,pol_pay_tax_list,
                      tax_scheme_list,tau_factor_list)

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

sol = sols[0]
b = baselines[2018]
b = b.make_np_arrays().compute_shares_and_gammas()
sol_pol_pay = sols[1]
sol_fair = sols[2]

palette = [sns.color_palette()[i] for i in [0,2,3]]
income_colors = {
    'Low-income' : sns.color_palette()[3],
    'Middle-income' : sns.color_palette()[0],
    'High-income' : sns.color_palette()[2],
                    }
#%% calculate eff tax rate

tax_paid = sol.run.carb_cost*(
    b.co2_intensity.rename_axis(['row_country','row_sector'])
    *(sol.iot.groupby(['row_country','row_sector','col_country']).sum()+sol.cons)
                              ).groupby('col_country').sum()

expenditure = (
    (1+sol.run.carb_cost*b.co2_intensity.rename_axis(['row_country','row_sector']))
    *(sol.iot.groupby(['row_country','row_sector','col_country']).sum()+sol.cons)
                              ).groupby('col_country').sum()

eff_tax_rate = tax_paid/expenditure

eff_tax_rate = eff_tax_rate.join(pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0)).sort_values('value')

#%% plot eff tax rate

colors = [income_colors[eff_tax_rate.loc[country,'income_label']] for country in eff_tax_rate.index]

fig, ax = plt.subplots()

ax.bar(eff_tax_rate.index,
       eff_tax_rate['value']*100,
       color=colors
       )
ax.set_xticklabels([''])
ax.bar_label(ax.containers[0],
             labels=eff_tax_rate.index.get_level_values(0), 
             rotation=90,
              label_type = 'edge',
              padding=5,zorder=10)
handles = [mpatches.Patch(color=income_colors[ind], label=ind) for ind in eff_tax_rate['income_label'].drop_duplicates()]
ax.legend(handles=handles,fontsize=25)

ax.set_ylabel('Effective tax rate (%)')
plt.grid(axis='x')

for save_format in save_formats:
    plt.savefig(save_path+'effective_tax_rate_by_country_by_income_group.'+save_format,format=save_format)
    
plt.show()

#%% calculate output reallocation

from scipy.spatial.distance import pdist

country_map = pd.read_csv('data/countries_after_agg.csv',sep=';').set_index('country')
country_list = sol.iot.index.get_level_values(0).drop_duplicates().to_list()
# output = b.output
country_dist = []
country_change = []
country_realloc = []
country_realloc_pos = []
country_realloc_neg = []
for country in country_list:
    country_dist.append(pdist([b.output.xs(country,level=0).value,sol.output.xs(country,level=0).value], metric = 'correlation')[0])
    temp = (sol.output['value']/sol.price['hat']).xs(country,level=0)-b.output.xs(country,level=0).value
    country_change.append(temp.sum())
    country_realloc_pos.append(temp[temp>0].sum())
    country_realloc_neg.append(temp[temp<0].sum())

country_dist_df = pd.DataFrame(index=pd.Index(country_list,name='country'))
country_dist_df['output'] = b.output.groupby(level=0).sum().value.values
country_dist_df['output_new'] = (sol.output['value']/sol.price['hat']).groupby(level=0).sum().values
country_dist_df['distance'] = country_dist
country_dist_df['realloc_pos'] = country_realloc_pos
country_dist_df['realloc_neg'] = country_realloc_neg
country_dist_df['change'] = country_change
country_dist_df['share_percent'] = (country_dist_df['output']/country_dist_df['output'].sum())*100
country_dist_df['share_new_percent'] = (country_dist_df['output_new']/country_dist_df['output_new'].sum())*100

country_dist_df['realloc_pos'] = np.abs(country_dist_df['realloc_pos'])
country_dist_df['realloc_neg'] = np.abs(country_dist_df['realloc_neg'])
country_dist_df['realloc'] = country_dist_df[['realloc_neg','realloc_pos']].min(axis=1)
country_dist_df['realloc'] = country_dist_df['realloc'] * np.sign(country_dist_df['change'])
country_dist_df['change_tot_nom'] = (country_dist_df['change']+country_dist_df['realloc'])

country_dist_df['realloc_percent'] = (country_dist_df['realloc']/country_dist_df['output'])*100
country_dist_df['change_percent'] = (country_dist_df['change']/country_dist_df['output'])*100
country_dist_df['total_change'] = country_dist_df['realloc_percent'] + country_dist_df['change_percent']

country_dist_df.sort_values('change_percent',inplace = True)
country_dist_df = country_dist_df.join(
    pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0)
    )
#%% plot output reallocation

print('Plotting production reallocation in percentages')

colors = [income_colors[country_dist_df.loc[country,'income_label']] for country in country_dist_df.index]

fig, ax = plt.subplots()
ax2 = ax.twinx()
ax2.grid(False)
ax2.set_yticks([])
ax.bar(country_dist_df.index.get_level_values(0)
            ,country_dist_df.change_percent
            ,label='Net change of output (%)',
            color=colors
            )

ax.bar(country_dist_df.index.get_level_values(0)
            ,country_dist_df.realloc_percent
            ,bottom = country_dist_df.change_percent
            ,label='Reallocated output (%)',
            color=colors,
            hatch="////")

ax.set_xticklabels([''])

ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)
ax.set_ylabel('Percent change and reallocation (%)',
              fontsize = 25)


leg = ax.legend(fontsize=25,
                loc='lower right')


handles = [mpatches.Patch(color=income_colors[ind], label=ind) for ind in country_dist_df['income_label'].drop_duplicates()]
ax2.legend(handles=handles,
           fontsize=25,
            # loc = 'center right',
            loc = (0.81,0.1)
           )

ax.grid(axis='x')

ax.bar_label(ax.containers[1],
             labels=country_dist_df.index.get_level_values(0),
             rotation=90,
              label_type = 'edge',
              padding=2,zorder=10)

for save_format in save_formats:
    plt.savefig(save_path+'reallocation_with_income_group.'+save_format,format=save_format)
    

plt.show()

#%% labor reallocation

labor = b.labor.set_index('country').rename_axis('col_country')['2018'].to_frame()
labor.columns = ['value']

wage_new = sol.va.groupby('col_country').sum()/labor
wage_old = b.va.groupby('col_country').sum()/labor

labor_realloc = np.abs(sol.va/wage_new - b.va/wage_old).groupby('col_country').sum()/2

labor_realloc['percent'] = labor_realloc['value']*100/labor['value']

labor_realloc = labor_realloc.sort_values('percent')

labor_realloc = labor_realloc.join(pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0))


colors = [income_colors[labor_realloc.loc[country,'income_label']] for country in labor_realloc.index]

fig, ax = plt.subplots()

ax.bar(labor_realloc.index,
       labor_realloc['percent'],
       color=colors
       )
ax.set_xticklabels([''])
ax.bar_label(ax.containers[0],
             labels=labor_realloc.index.get_level_values(0), 
             rotation=90,
              label_type = 'edge',
              padding=5,zorder=10)
handles = [mpatches.Patch(color=income_colors[ind], label=ind) for ind in labor_realloc['income_label'].drop_duplicates()]
ax.legend(handles=handles,fontsize=25)
ax.grid(axis='x')
ax.set_ylabel('Labor reallocation (% of workforce)',fontsize = 25)

for save_format in save_formats:
    plt.savefig(save_path+'labor_reallocation_by_country_by_income_group.'+save_format,format=save_format)
    
plt.show()

#%% cumul emissions bar plot

cumul_em = b.cumul_emissions_share*100

cumul_em = cumul_em.join(pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0))
cumul_em  = cumul_em.sort_values('value')

colors = [income_colors[cumul_em.loc[country,'income_label']] for country in cumul_em.index]

fig, ax = plt.subplots()

ax.bar(cumul_em.index,
       cumul_em['value'],
       color=colors
       )
ax.set_xticklabels([''])
ax.bar_label(ax.containers[0],
             labels=cumul_em.index.get_level_values(0), 
             rotation=90,
              label_type = 'edge',
              padding=5,zorder=10)
handles = [mpatches.Patch(color=income_colors[ind], label=ind) for ind in cumul_em['income_label'].drop_duplicates()]
ax.legend(handles=handles,fontsize=25)
ax.grid(axis='x')
ax.set_ylabel('Share of historical cumulative emissions (%)',fontsize = 25)

for save_format in save_formats:
    plt.savefig(save_path+'cumul_emission_share_bar_plot.'+save_format,format=save_format)
    
plt.show()

#%% cumul emissions wrt GDP

cumul_em = b.cumul_emissions_share.copy()
cumul_em['value'] = cumul_em['value']*labor['value'].sum()/labor['value']
cumul_em['value'] = cumul_em['value']*100/cumul_em['value'].sum()
cumul_em['gdp'] = b.va.groupby('col_country').sum().rename_axis('country')['value']/labor['value']

cumul_em = cumul_em.join(pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0))
# cumul_em  = cumul_em.sort_values('value')

colors = [income_colors[cumul_em.loc[country,'income_label']] for country in cumul_em.index]

fig, ax = plt.subplots()

ax.scatter(cumul_em['gdp'],
           cumul_em['value'],
           color = colors,
           lw=5
           )
ax.set_xscale('log')
# ax.set_yscale('log')

texts = [plt.text(cumul_em['gdp'].loc[country], 
                  cumul_em['value'].loc[country], 
                  country,
                  size=20, 
                  c = colors[i]) for i,country in enumerate(country_list)]

adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        )
            )
handles = [mpatches.Patch(color=income_colors[ind], label=ind) for ind in cumul_em['income_label'].drop_duplicates()]
ax.legend(handles=handles,
           fontsize=25,
           loc = 'lower right')
ax.set_xlabel('GDP per capita (millions US$)',fontsize = 25)
ax.set_ylabel('Share of historical emissions per capita (%)',fontsize = 25)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())


for save_format in save_formats:
    plt.savefig(save_path+'per_capita_historical_em_contribution_by_country_by_income_group.'+save_format,format=save_format)
    
plt.show()

#%% welfare changes wrt gdp per capita

welfare_change = sol.utility.copy()

welfare_change['gdp_p_c'] = (b.va.groupby('col_country').sum()/labor).rename_axis('country')
welfare_change = welfare_change.join(
    pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0)
    )
welfare_change['hat'] = (welfare_change['hat']-1)*100

fig, ax = plt.subplots()

colors = [income_colors[welfare_change.loc[country,'income_label']] for country in welfare_change.index]

ax.scatter(welfare_change['gdp_p_c'],
           welfare_change['hat'],
           color = colors,
           lw=5
           )

texts = [plt.text(welfare_change['gdp_p_c'].loc[country], 
                  welfare_change['hat'].loc[country], 
                  country,
                  size=20, 
                  c = colors[i]) for i,country in enumerate(country_list)]

adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        )
            )
handles = [mpatches.Patch(color=income_colors[ind], label=ind) for ind in welfare_change['income_label'].drop_duplicates()]
ax.legend(handles=handles,
           fontsize=25,
           loc = 'lower right')
ax.set_xlabel('GDP per capita (millions US$)',fontsize = 25)
ax.set_ylabel('Real income change (%)',fontsize = 25)

ax.axhline(0, color='k')

for save_format in save_formats:
    plt.savefig(save_path+'welfare_change_by_gdp_by_country_by_income_group.'+save_format,format=save_format)
    
plt.show()


#%% welfare changes wrt historical emissions

welfare_change = sol.utility.copy()

welfare_change['share_cumul_emissions'] = pd.read_csv('data/share_of_cumulative_co2_treated.csv',index_col=[0,1]).loc[2018]*100
welfare_change = welfare_change.join(pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0))
welfare_change['hat'] = (welfare_change['hat']-1)*100

fig, ax = plt.subplots()

colors = [income_colors[welfare_change.loc[country,'income_label']] for country in welfare_change.index]

ax.scatter(welfare_change['share_cumul_emissions'],
           welfare_change['hat'],
           color = colors,
           lw=5
           )
ax.set_xscale('log')
texts = [plt.text(welfare_change['share_cumul_emissions'].loc[country], 
                  welfare_change['hat'].loc[country], 
                  country,
                  size=20, 
                  c = colors[i]) for i,country in enumerate(country_list)]

adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        )
            )
handles = [mpatches.Patch(color=income_colors[ind], label=ind) for ind in eff_tax_rate['income_label'].drop_duplicates()]
ax.legend(handles=handles,
           fontsize=25,
           loc = 'lower right')
ax.set_xlabel('Share of cumulative historical emissions (%)',fontsize = 25)
ax.set_ylabel('Real income change (%)',fontsize = 25)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.axhline(0, color='k')

for save_format in save_formats:
    plt.savefig(save_path+'welfare_change_by_share_cumul_emissions_by_country_by_income_group.'+save_format,format=save_format)
    
plt.show()

#%% fair tax transfer per capita

contrib = sol_fair.contrib.copy()
contrib['per_capita'] = contrib['value']*1e6/labor['value']
contrib = contrib.join(
    pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0)
    ).sort_values('per_capita')

fig, ax = plt.subplots()

colors = [income_colors[contrib.loc[country,'income_label']] for country in contrib.index]

ax.bar(contrib.index,
       contrib['per_capita'],
       color=colors
       )
ax.set_xticklabels([''])
ax.bar_label(ax.containers[0],
             labels=contrib.index.get_level_values(0), 
             rotation=90,
              label_type = 'edge',
              padding=5,zorder=10)
handles = [mpatches.Patch(color=income_colors[ind], label=ind) for ind in contrib['income_label'].drop_duplicates()]
ax.legend(handles=handles,fontsize=25)
ax.grid(axis='x')
ax.set_ylabel('Monetary transfer per capita (US$)',fontsize=25)

for save_format in save_formats:
    plt.savefig(save_path+'fair_tax_transfers_by_country_by_income_group.'+save_format,format=save_format)
    
plt.show()

#%% fair tax welfare changes wrt historical emissions

welfare_change = sol_fair.utility.copy()

welfare_change['share_cumul_emissions'] = pd.read_csv('data/share_of_cumulative_co2_treated.csv',index_col=[0,1]).loc[2018]
welfare_change = welfare_change.join(pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0))

fig, ax = plt.subplots()

colors = [income_colors[welfare_change.loc[country,'income_label']] for country in welfare_change.index]

ax.scatter(welfare_change['share_cumul_emissions'],
            welfare_change['hat'],
            color = colors,
            lw=5
            )
ax.set_xscale('log')
texts = [plt.text(welfare_change['share_cumul_emissions'].loc[country], 
                  welfare_change['hat'].loc[country], 
                  country,
                  size=20, 
                  c = colors[i]) for i,country in enumerate(country_list)]

adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        )
            )
handles = [mpatches.Patch(color=income_colors[ind], label=ind) for ind in welfare_change['income_label'].drop_duplicates()]
ax.legend(handles=handles,
            fontsize=25,
            loc = 'lower right')
ax.set_xlabel('Share of cumulative historical emissions',fontsize = 25)
ax.set_ylabel('Real income change',fontsize = 25)
ax.set_ylim([0.95,1])

for save_format in save_formats:
    plt.savefig(save_path+'fair_tax_welfare_change_by_share_cumul_emissions_by_country_by_income_group.'+save_format,format=save_format)
    
plt.show()


#%% pol pay tax transfer per capita

contrib = sol_pol_pay.contrib.copy()
contrib['per_capita'] = contrib['value']*1e6/labor['value']
contrib = contrib.join(
    pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0)
    ).sort_values('per_capita')

fig, ax = plt.subplots()

colors = [income_colors[contrib.loc[country,'income_label']] for country in contrib.index]

ax.bar(contrib.index,
       contrib['per_capita'],
       color=colors
       )
ax.set_xticklabels([''])
ax.bar_label(ax.containers[0],
             labels=contrib.index.get_level_values(0), 
             rotation=90,
              label_type = 'edge',
              padding=5,zorder=10)
handles = [mpatches.Patch(color=income_colors[ind], label=ind) for ind in contrib['income_label'].drop_duplicates()]
ax.legend(handles=handles,fontsize=25)
ax.grid(axis='x')
ax.set_ylabel('Monetary transfer per capita (US$)',fontsize=25)

for save_format in save_formats:
    plt.savefig(save_path+'pol_pay_tax_transfers_by_country_by_income_group.'+save_format,format=save_format)
    
plt.show()

#%% pol pay tax welfare changes wrt historical emissions

welfare_change = sol_pol_pay.utility.copy()

welfare_change['share_cumul_emissions'] = pd.read_csv('data/share_of_cumulative_co2_treated.csv',index_col=[0,1]).loc[2018]*100
welfare_change = welfare_change.join(pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0))

welfare_change['hat'] = (welfare_change['hat']-1)*100

fig, ax = plt.subplots()

colors = [income_colors[welfare_change.loc[country,'income_label']] for country in welfare_change.index]

ax.scatter(welfare_change['share_cumul_emissions'],
           welfare_change['hat'],
           color = colors,
           lw=5
           )
ax.set_xscale('log')
texts = [plt.text(welfare_change['share_cumul_emissions'].loc[country], 
                  welfare_change['hat'].loc[country], 
                  country,
                  size=20, 
                  c = colors[i]) for i,country in enumerate(country_list)]

# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         )
#             )
handles = [mpatches.Patch(color=income_colors[ind], label=ind) for ind in eff_tax_rate['income_label'].drop_duplicates()]
ax.legend(handles=handles,
           fontsize=25,
           loc = 'lower right')
ax.set_xlabel('Share of cumulative historical emissions (%)',fontsize = 25)
ax.set_ylabel('Real income change (%)',fontsize = 25)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
# ax.set_ylim([0.95,1])

# for save_format in save_formats:
#     plt.savefig(save_path+'pol_pay_tax_welfare_change_by_share_cumul_emissions_by_country_by_income_group.'+save_format,format=save_format)
    
plt.show()

#%% pol pay tax welfare changes wrt gdp per capita

welfare_change = sol_pol_pay.utility.copy()

welfare_change['share_cumul_emissions'] = pd.read_csv('data/share_of_cumulative_co2_treated.csv',index_col=[0,1]).loc[2018]*100
welfare_change = welfare_change.join(pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0))
welfare_change['gdp_p_c'] = (b.va.groupby('col_country').sum()/labor).rename_axis('country')

welfare_change['hat'] = (welfare_change['hat']-1)*100

fig, ax = plt.subplots()

colors = [income_colors[welfare_change.loc[country,'income_label']] for country in welfare_change.index]

ax.scatter(welfare_change['gdp_p_c'],
           welfare_change['hat'],
           color = colors,
           lw=5
           )
# ax.set_xscale('log')
texts = [plt.text(welfare_change['gdp_p_c'].loc[country], 
                  welfare_change['hat'].loc[country], 
                  country,
                  size=20, 
                  c = colors[i]) for i,country in enumerate(country_list)]

adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        )
            )
handles = [mpatches.Patch(color=income_colors[ind], label=ind) for ind in eff_tax_rate['income_label'].drop_duplicates()]
ax.legend(handles=handles,
           fontsize=25,
           loc = 'lower right')
ax.set_xlabel('GDP per capita (millions US$)',fontsize = 25)
ax.set_ylabel('Real income change (%)',fontsize = 25)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
# ax.set_ylim([0.95,1])

for save_format in save_formats:
    plt.savefig(save_path+'pol_pay_tax_welfare_change_by_gdp_per_cap_by_country_by_income_group.'+save_format,format=save_format)
    
plt.show()

#%% pol pay tax welfare changes wrt historical emissions

welfare_change = -(sol_pol_pay.utility.copy()-1)*b.cons.groupby('col_country').sum(
    ).rename_axis(['country']).rename(columns={'value':'hat'})

welfare_change['share_cumul_emissions'] = pd.read_csv(
    'data/share_of_cumulative_co2_treated.csv',index_col=[0,1]
    ).loc[2018]
welfare_change = welfare_change.join(pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0))

fig, ax = plt.subplots()

colors = [income_colors[welfare_change.loc[country,'income_label']] for country in welfare_change.index]

ax.scatter(welfare_change['share_cumul_emissions'],
           welfare_change['hat'],
           color = colors,
           lw=5
           )
# ax.set_xscale('symlog')
# ax.set_yscale('symlog')
texts = [plt.text(welfare_change['share_cumul_emissions'].loc[country], 
                  welfare_change['hat'].loc[country], 
                  country,
                  size=20, 
                  c = colors[i]) for i,country in enumerate(country_list)]

# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         )
#             )
handles = [mpatches.Patch(color=income_colors[ind], label=ind) for ind in eff_tax_rate['income_label'].drop_duplicates()]
ax.legend(handles=handles,
           fontsize=25,
           # loc = 'lower right'
           )
ax.set_xlabel('Share of cumulative historical emissions',fontsize = 25)
ax.set_ylabel('Real income level decrease (economic cost in M$)',fontsize = 25)
# ax.set_ylim([0.95,1])

# for save_format in save_formats:
#     plt.savefig(save_path+'pol_pay_tax_delta_welfare_by_share_cumul_emissions_by_country_by_income_group.'+save_format,format=save_format)
    
plt.show()

#%% pol_pay vs fair tax transfers

# transfers = sol_fair.contrib.copy().rename(columns={'value':'fair'})
# transfers['pol_pay'] = sol_pol_pay.contrib['value']
# transfers['fair'] = transfers['fair']/labor['value']
# transfers['pol_pay'] = transfers['pol_pay']/labor['value']
# transfers['diff'] = transfers['pol_pay']-transfers['fair']

# transfers = transfers.join(
#     pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0)
#     )

# transfers = transfers.sort_values('diff')

# fig, ax = plt.subplots()

# colors = [income_colors[transfers.loc[country,'income_label']] for country in transfers.index]

# ax.scatter(transfers['fair'],
#            transfers['pol_pay'],
#            color = colors,
#            lw=5
#            )
# ax.plot(transfers['fair'].sort_values(),
#         transfers['fair'].sort_values(),
#         ls='--',
#         color='grey')
# # ax.set_xscale('symlog')
# # ax.set_yscale('symlog')
# texts = [plt.text(transfers['fair'].loc[country], 
#                   transfers['pol_pay'].loc[country], 
#                   country,
#                   size=20, 
#                   c = colors[i]) for i,country in enumerate(country_list)]

# # adjust_text(texts, precision=0.001,
# #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
# #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
# #         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
# #                         )
# #             )
# handles = [mpatches.Patch(color=income_colors[ind], label=ind) for ind in transfers['income_label'].drop_duplicates()]
# ax.legend(handles=handles,
#            fontsize=25,
#            loc = 'lower right')
# ax.set_xlabel('Fair tax monetary transfer',fontsize = 25)
# ax.set_ylabel('Real income change (%)',fontsize = 25)

# # ax.axhline(0, color='k')

# # for save_format in save_formats:
# #     plt.savefig(save_path+'welfare_change_by_gdp_by_country_by_income_group.'+save_format,format=save_format)
    
# plt.show()

#%%

labor = b.labor.set_index('country').rename_axis('col_country')['2018'].to_frame()
labor.columns = ['value']

transfers = sol_fair.contrib.copy().rename(columns={'value':'fair'})
transfers['pol_pay'] = sol_pol_pay.contrib['value']
transfers['fair'] = transfers['fair']#/labor['value']
transfers['pol_pay'] = transfers['pol_pay']#/labor['value']
transfers['diff'] = transfers['pol_pay']-transfers['fair']
transfers = transfers.sort_values('diff')
transfers = transfers.join(
    pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0)
    )
colors = [income_colors[transfers.loc[country,'income_label']] for country in transfers.index]
fig, ax = plt.subplots()

ax.bar(transfers.index,
        transfers['diff'],
        color=colors
        )
ax.set_xticklabels([''])
ax.bar_label(ax.containers[0],
              labels=transfers.index.get_level_values(0), 
              rotation=90,
              label_type = 'edge',
              padding=5,zorder=10)
handles = [mpatches.Patch(color=income_colors[ind], label=ind) for ind in transfers['income_label'].drop_duplicates()]
ax.legend(handles=handles,fontsize=25)
ax.grid(axis='x')
ax.set_ylabel('Pol pay transfer - fair tax transfer',fontsize = 25)

# for save_format in save_formats:
#     plt.savefig(save_path+'diff_pol_pay_fair_p.'+save_format,format=save_format)
    
plt.show()

transfers['labor'] = labor
transfers['fair_p_c'] = transfers['fair']*1e6/transfers['labor']
transfers['pol_pay_p_c'] = transfers['pol_pay']*1e6/transfers['labor']
transfers['diff_p_c'] = transfers['pol_pay_p_c']-transfers['fair_p_c']

transfers = transfers.sort_values('fair_p_c')

# transfers.to_csv(save_path+'transfers.csv')
