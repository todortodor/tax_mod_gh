#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:06:09 2022

@author: slepot
"""

main_path = './'
import sys
# sys.path.append(main_path+"lib/")
# import solver_funcs as s
import lib.data_funcs as d
import lib.treatment_funcs as t
import pandas as pd
# from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize as opt
import seaborn as sns
sns.set_style("whitegrid")

#%% load solution
main_path = './'
results_path = main_path+'results/'
data_path = main_path+'data/'
dir_num = 1
year = 2018
carb_cost_list = [1e-4]
# carb_cost_list = np.linspace(0,1e-3,1001)
# eta_path = ['elasticities_agg1.csv']
# sigma_path = ['elasticities_agg1.csv']
eta_path = ['uniform_elasticities_4.csv']
sigma_path = ['uniform_elasticities_4.csv']
taxed_countries_list = [None]
taxing_countries_list = [None]
taxed_sectors_list = [None]
specific_taxing_list = [None]
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

# #%% load trade emissions

# df = pd.read_csv('data/trade_emissions.csv')
# df = df.loc[df.ICIO != 'D09']
# corresp_sect = pd.read_csv('data/corresp_sect_trade_em_tax_mod.csv').set_index('trade_em_sectors')
# corresp_sect.index.name = 'ICIO'
# df = df.loc[df.PERIOD == 2018]
# s = df.ICIO.replace(corresp_sect['tax_mod_sectors'])
# df['sector'] = s
# df1 = df.loc[df.FLOW == 1]
# df1.set_index(['PARTNER_ISO3','sector','DECLARANT_ISO3','TRANSPORT_MODE'],inplace=True)
# df1.rename_axis(['row_country','row_sector','col_country','TRANSPORT_MODE'],inplace=True)
# df2 = df.loc[df.FLOW == 2]
# df2.set_index(['DECLARANT_ISO3','sector','PARTNER_ISO3','TRANSPORT_MODE'],inplace=True)
# df2.rename_axis(['row_country','row_sector','col_country','TRANSPORT_MODE'],inplace=True)
# df = pd.concat([df1,df2])

# #%%

# trade = baselines[year].iot.groupby(level=[0,1,2]).sum()
# trade['cons'] = baselines[year].cons.value
# trade['baseline'] = trade.value + trade.cons
# trade['cf'] = sols[0].iot.groupby(level=[0,1,2]).sum().value + sols[0].cons.value
# trade = trade[['baseline','cf']]
# trade['hat'] = trade['cf']/trade['baseline']

# prod_change = trade.groupby(['row_sector','col_country']).sum()
# prod_change['hat'] = prod_change['cf']/prod_change['baseline']
# prod_change = prod_change[['hat']]
# prod_change.columns = ['sector_hat']

# co2_int_base = baselines[year].co2_intensity
# co2_int = co2_int_base.copy()
# co2_int.columns = ['co2_int']
# co2_int.rename_axis(['row_country','row_sector'],inplace=True)
# co2_int_dest = co2_int_base.copy()
# co2_int_dest.columns = ['co2_int_dest']
# co2_int_dest.rename_axis(['col_country','row_sector'],inplace=True)
# df = df.join(co2_int)
# df = df.join(co2_int_dest)
# df['co2_int'] = df['co2_int']/1e3
# df['co2_int_dest'] = df['co2_int_dest']/1e3

# df_temp = df[['TRADE_VALUE','co2_int','co2_int_dest','TRNSP_EMINT_LOW']]
# df_temp['mode_spec_trade_em'] = df_temp['TRADE_VALUE']*(df_temp['co2_int']+df_temp['TRNSP_EMINT_LOW'])
# df_temp['mode_spec_autarky_em'] = df_temp['TRADE_VALUE']*(df_temp['co2_int_dest'])

# df_temp = df_temp[['TRADE_VALUE','mode_spec_trade_em','mode_spec_autarky_em']]

# df_temp = df_temp.groupby(['row_country','row_sector','col_country']).sum()

# df_temp['mode_spec_trade_em_int'] = df_temp['mode_spec_trade_em']/df_temp['TRADE_VALUE']
# df_temp['mode_spec_autarky_em_int'] = df_temp['mode_spec_autarky_em']/df_temp['TRADE_VALUE']
# df_temp['ratio_autarky_trade'] = df_temp['mode_spec_autarky_em_int']/df_temp['mode_spec_trade_em_int']

# df_temp['hat'] = trade['hat']

# df_temp = df_temp.join(df['route']).groupby(['row_country','row_sector','col_country']).first()
# df_temp = df_temp.dropna()
# df_temp = df_temp.join(prod_change)

# #%%
# # mpl.rc_file_defaults()
# def func(x, a, b):
#       return np.exp(-a * np.log(x) + b)
# # def func(x, a, b):
# #       return a  -b * x
# # def func(x, b):
# #      return np.exp(-b * x)

# fig,ax = plt.subplots(figsize=(16,12))

# for route in df_temp['route'].drop_duplicates()[:10]:
#     ax.scatter(df_temp.loc[df_temp['route'] == route, 'ratio_autarky_trade'], 
#                df_temp.loc[df_temp['route'] == route, 'hat'],
#                label = route, s=10)

#     # optimizedParameters, pcov = opt.curve_fit(func, 
#     #                                           df_temp.loc[df_temp['route'] == route, 'ratio_autarky_trade'], 
#     #                                           df_temp.loc[df_temp['route'] == route, 'hat']);
    
#     # # Use the optimized parameters to plot the best fit
#     # ax.plot(df_temp.loc[df_temp['route'] == route, 'ratio_autarky_trade'].sort_values(), 
#     #          func(df_temp.loc[df_temp['route'] == route, 'ratio_autarky_trade'].sort_values(), *optimizedParameters), 
#     #           # label="fit_"+route,
#     #          # c = 'red',
#     #          ls = '--',
#     #          lw=3);



# lgnd = plt.legend(fontsize = 10)
# for handle in lgnd.legendHandles:
#     handle.set_sizes([20])
# ax.set_xlabel('Emission intensity at destination / Emission intensity trade',fontsize = 20)
# ax.set_ylabel('Change in trade flow',fontsize = 20)
# plt.xscale('log')
# plt.yscale('log')
# plt.ylim(1e-2,10)
# plt.xlim(1e-4,1e4)
# ax.tick_params(axis='both', which='major', labelsize=15)
# ax.tick_params(axis='both', which='minor', labelsize=15)

# plt.show()

# #%%
# # mpl.rc_file_defaults()
# # def func(x, a, b):
# #       return np.exp(-a * np.log(x) + b)
# # def func(x, a, b):
# #       return a  -b * x
# # def func(x, b):
# #      return np.exp(-b * x)

# fig,ax = plt.subplots(figsize=(16,12))

# for route in df_temp['route'].drop_duplicates()[:10]:
#     ax.scatter(df_temp.loc[df_temp['route'] == route, 'ratio_autarky_trade'], 
#                df_temp.loc[df_temp['route'] == route, 'hat']/df_temp.loc[df_temp['route'] == route, 'sector_hat'],
#                label = route, s=10)

#     # optimizedParameters, pcov = opt.curve_fit(func, 
#     #                                           df_temp.loc[df_temp['route'] == route, 'ratio_autarky_trade'], 
#     #                                           df_temp.loc[df_temp['route'] == route, 'hat']);
    
#     # # Use the optimized parameters to plot the best fit
#     # ax.plot(df_temp.loc[df_temp['route'] == route, 'ratio_autarky_trade'].sort_values(), 
#     #          func(df_temp.loc[df_temp['route'] == route, 'ratio_autarky_trade'].sort_values(), *optimizedParameters), 
#     #           # label="fit_"+route,
#     #          # c = 'red',
#     #          ls = '--',
#     #          lw=3);



# lgnd = plt.legend(fontsize = 10)
# for handle in lgnd.legendHandles:
#     handle.set_sizes([20])
# ax.set_xlabel('Emission intensity at destination / Emission intensity trade',fontsize = 20)
# ax.set_ylabel('Change in trade flow/ change in imports in sectorxdest',fontsize = 20)
# plt.xscale('log')
# plt.yscale('log')
# plt.ylim(1e-2,10)
# plt.xlim(1e-4,1e4)
# ax.tick_params(axis='both', which='major', labelsize=15)
# ax.tick_params(axis='both', which='minor', labelsize=15)

# plt.show()

# #%%

# fig,ax = plt.subplots(figsize=(16,12))

# for route in df_temp['route'].drop_duplicates()[:10]:
#     x = df_temp.loc[df_temp['route'] == route, 'mode_spec_autarky_em_int']
#     y = df_temp.loc[df_temp['route'] == route, 'mode_spec_trade_em_int']
#     ax.scatter(x, 
#                y,
#                label = route, s=10)

#     # optimizedParameters, pcov = opt.curve_fit(func, 
#     #                                           x, 
#     #                                           y);
    
#     # # Use the optimized parameters to plot the best fit
#     # ax.plot(x.sort_values(), 
#     #           func(x.sort_values(), *optimizedParameters), 
#     #           # label="fit_"+route,
#     #           # c = 'red',
#     #           ls = '--',
#     #           lw=3);

# ax.plot([0,df_temp['mode_spec_autarky_em_int'].max()],
#         [0,df_temp['mode_spec_autarky_em_int'].max()],
#         color='black',lw=2,ls='--')

# lgnd = plt.legend(fontsize = 10)
# for handle in lgnd.legendHandles:
#     handle.set_sizes([20])
# ax.set_xlabel('Emission intensity at destination',fontsize = 20)
# ax.set_ylabel('Emission intensity trade (at origin + transport)',fontsize = 20)
# plt.xscale('log')
# plt.yscale('log')
# # plt.ylim(1e-2,10)
# # plt.xlim(1e-4,1e4)
# ax.tick_params(axis='both', which='major', labelsize=15)
# ax.tick_params(axis='both', which='minor', labelsize=15)

# plt.show()


# #%%

# l_sectors = df.ICIO.drop_duplicates().to_list()

# for s in baselines[2018].sector_list: 
#     if 'D'+s in l_sectors:
#         print(s,'in')
#     else:
#         print(s,'out')
        
# for s in l_sectors:
#     if s[1:] in baselines[2018].sector_list:
#         print(s,'in')
#     else:
#         print(s,'out')
        
# #%%

# corresp_sect = pd.DataFrame()
# corresp_sect['trade_em_sectors'] = l_sectors
# corresp_sect['tax_mod_sectors'] = [s[1:] for s in corresp_sect['trade_em_sectors']]
# corresp_sect['tax_mod_sectors'].replace('94T96','94T98',inplace=True)

# #%% technique
# b = baselines[2018]
# cf = sols[0]
# N = b.country_number
# S = b.sector_number

# trade_baseline = trade.baseline.values.reshape((N,S,N))
# trade_cf = trade.cf.values.reshape((N,S,N)) 
# e = b.co2_intensity.value.values.reshape((N,S))

# technique = np.einsum('s,->',
#                       trade_baseline.sum(axis=2).sum(axis=0)*\
#                           np.einsum('isj,is->s',
#                                     trade_cf,
#                                     e)/\
#                           1/trade_cf.sum(axis=2).sum(axis=0),
#                         1/np.einsum('isj,is->',
#                                   trade_cf,
#                                   e)
#                       )

# importer_all = np.einsum('isj,is->j',
#           trade_cf,
#           e)/np.einsum('isj,is->j',
#                     trade_baseline,
#                     e)

# importer_scale = np.einsum('isj->j',trade_cf)/\
#     np.einsum('isj->j',trade_baseline)
    
# importer_TC = importer_all/importer_scale

# importer_technique = np.einsum('sj,j->j',
#                       trade_baseline.sum(axis=0)*\
#                           np.einsum('isj,is->sj',
#                                     trade_cf,
#                                     e)/\
#                           1/trade_cf.sum(axis=0),
#                         1/np.einsum('isj,is->j',
#                                   trade_cf,
#                                   e)
#                       )

# #%% 

# fig,ax = plt.subplots(figsize = (8,12))

# ax.scatter(importer_all,b.country_list,label = 'All')
# ax.scatter(importer_scale,b.country_list,label = 'Scale',marker='v')
# ax.scatter(importer_TC,b.country_list,label = 'Techinque and comp',marker = 's')
# ax.scatter(importer_technique,b.country_list,label = 'Technique',marker = '*')
# ax.plot(np.ones(b.country_number),b.country_list,color='black',ls='--')

# plt.legend()
# plt.grid(axis='y')

# plt.savefig('graph_images/decompositions.png')

# plt.show()

# #%% Ralph version 

b = baselines[2018]
# cf = sols[0]
N = b.country_number
S = b.sector_number

# trade_baseline = trade.baseline.values.reshape((N,S,N))
# trade_cf = trade.cf.values.reshape((N,S,N)) 
# e = b.co2_intensity.value.values.reshape((N,S))

#%%

# trade_baseline = trade.baseline.values.reshape((N,S,N))
# trade_cf = trade[10].values.reshape((N,S,N)) 
# e = b.co2_intensity.value.values.reshape((N,S))

def X(trade):
    res = np.einsum('isj->',
                  trade)
    return res

def alpha_s(trade):
    num = np.einsum('isj->s',trade)
    denom = trade.sum()
    return num/denom

def alpha_sj(trade):
    num = np.einsum('isj->sj',trade)
    denom = np.einsum('isj->s',trade)
    return np.einsum('sj,s->sj',num,1/denom)

def alpha_isj(trade):
    num = np.einsum('isj->isj',trade)
    denom = np.einsum('isj->sj',trade)
    return np.einsum('isj,sj->isj',num,1/denom)

def e_sj(trade,e):
    return np.einsum('isj,is->sj',alpha_isj(trade),e)
    
def e_s(trade,e):
    return np.einsum('sj,sj->s',alpha_sj(trade),e_sj(trade,e))

def e_scal(trade,e):
    return np.einsum('s,s->',alpha_s(trade),e_s(trade,e))

def epsilon_s(trade,e):
    return np.einsum('s,s,,->s',
                     np.einsum('isj->s',trade),
                     e_s(trade,e),
                     1/np.einsum('isj->',trade),
                     1/e_scal(trade,e))

def epsilon_sj(trade,e):
    return np.einsum('sj,sj,,->sj',
                     np.einsum('isj->sj',trade),
                     e_sj(trade,e),
                     1/np.einsum('isj->',trade),
                     1/e_scal(trade,e))

def epsilon_isj(trade,e):
    return np.einsum('isj,is,,->isj',
                     np.einsum('isj->isj',trade),
                     e,
                     1/np.einsum('isj->',trade),
                     1/e_scal(trade,e))



# term_1 = (X(trade_cf) - X(trade_baseline))/X(trade_baseline)

# term_2 = np.einsum('s,s,s->',
#                    epsilon_s(trade_baseline,e),
#                    alpha_s(trade_cf)-alpha_s(trade_baseline),
#                    1/alpha_s(trade_baseline))

# term_3 = np.einsum('sj,sj,sj->',
#            epsilon_sj(trade_baseline,e),
#            alpha_sj(trade_cf)-alpha_sj(trade_baseline),
#            1/alpha_sj(trade_baseline))

# term_4 = np.einsum('isj,isj,isj->',
#            epsilon_isj(trade_baseline,e),
#            alpha_isj(trade_cf)-alpha_isj(trade_baseline),
#            np.divide(1, 
#                      alpha_isj(trade_baseline), 
#                      out = np.zeros_like(alpha_isj(trade_baseline)), 
#                      where = alpha_isj(trade_baseline)!=0 )
#            )

# print(term_1,term_2,term_3,term_4)

# term_3_decomp = np.einsum('sj,sj,sj->j',
#            epsilon_sj(trade_baseline,e),
#            alpha_sj(trade_cf)-alpha_sj(trade_baseline),
#            1/alpha_sj(trade_baseline))

# term_4_decomp = np.einsum('isj,isj,isj->i',
#            epsilon_isj(trade_baseline,e),
#            alpha_isj(trade_cf)-alpha_isj(trade_baseline),
#            np.divide(1, 
#                      alpha_isj(trade_baseline), 
#                      out = np.zeros_like(alpha_isj(trade_baseline)), 
#                      where = alpha_isj(trade_baseline)!=0 )
#            )

# term_2_decomp_alpha = np.einsum('s,s->s',
#                    alpha_s(trade_cf)-alpha_s(trade_baseline),
#                    1/alpha_s(trade_baseline))

# term_2_decomp_epsilon = np.einsum('s->s',
#                    epsilon_s(trade_baseline,e))

# #%%
# country_descr = pd.read_csv('data/countries_after_agg.csv',sep=';').set_index('country')

# fig,ax = plt.subplots(figsize = (10,14))

# ax.scatter(term_3_decomp ,np.arange(b.country_number),marker='s',color=sns.color_palette()[3])

# # plt.legend()
# plt.grid(axis='y')
# ax.plot(np.zeros(b.country_number),b.country_list,color='grey',ls='--')
# plt.yticks(np.arange(b.country_number),labels = [])
# ax.tick_params(axis='both', which='major', labelsize=15)
# for i,c in enumerate(b.country_list):
#     plt.text(x=term_3_decomp[i]+0.0003, y=i-0.1,s=country_descr.loc[c,'country_name'],
#              horizontalalignment='left',
#      verticalalignment='center',)
# plt.margins(y=0.02)
# plt.xlim([-0.025,0.005])
# # plt.title('Term 3 decomposition', fontsize = 20)
# # plt.savefig('graph_images/decomposition_of_term_3.pdf')

# plt.show()
# #%%

# country_descr = pd.read_csv('data/countries_after_agg.csv',sep=';').set_index('country')

# fig,ax = plt.subplots(figsize = (10,14))

# ax.scatter(term_4_decomp ,np.arange(b.country_number),marker='s',color=sns.color_palette()[3])

# # plt.legend()
# plt.grid(axis='y')
# ax.plot(np.zeros(b.country_number),b.country_list,color='grey',ls='--')
# plt.yticks(np.arange(b.country_number),labels = [])
# ax.tick_params(axis='both', which='major', labelsize=15)
# for i,c in enumerate(b.country_list):
#     plt.text(x=term_4_decomp[i]+0.0001, y=i-0.1,s=country_descr.loc[c,'country_name'],
#              horizontalalignment='left',
#      verticalalignment='center',)
# plt.margins(y=0.02)
# plt.xlim([-0.008,0.002])
# # plt.title('Term 3 decomposition', fontsize = 20)
# # plt.savefig('graph_images/decomposition_of_term_4.pdf')

# plt.show()

#%%

trade = baselines[year].iot.groupby(level=[0,1,2]).sum()
trade['cons'] = baselines[year].cons.value
trade['baseline'] = trade.value + trade.cons
trade = trade[['baseline']]

for i,sol in enumerate(sols):
    trade[i] = sol.iot.groupby(level=[0,1,2]).sum().value + sol.cons.value
    
#%%
l_term_1 = []
l_term_2 = []
l_term_3 = []
l_term_4 = []
l_em_reduc = []
for i in range(100):
    print(i)
    trade_baseline = trade[i].values.reshape((N,S,N))
    # trade_baseline = trade['baseline'].values.reshape((N,S,N))
    trade_cf = trade[i+1].values.reshape((N,S,N)) 
    e = b.co2_intensity.value.values.reshape((N,S))
    term_1 = (X(trade_cf) - X(trade_baseline))/X(trade_baseline)

    term_2 = np.einsum('s,s,s->',
                       epsilon_s(trade_baseline,e),
                       alpha_s(trade_cf)-alpha_s(trade_baseline),
                       1/alpha_s(trade_baseline))

    term_3 = np.einsum('sj,sj,sj->',
               epsilon_sj(trade_baseline,e),
               alpha_sj(trade_cf)-alpha_sj(trade_baseline),
               1/alpha_sj(trade_baseline))

    term_4 = np.einsum('isj,isj,isj->',
               epsilon_isj(trade_baseline,e),
               alpha_isj(trade_cf)-alpha_isj(trade_baseline),
               np.divide(1, 
                         alpha_isj(trade_baseline), 
                         out = np.zeros_like(alpha_isj(trade_baseline)), 
                         where = alpha_isj(trade_baseline)!=0 )
               )
    em_reduc = (np.einsum('isj,is->',
                         trade_cf,
                         e)-\
                np.einsum('isj,is->',
                          trade_baseline,
                          e))/np.einsum('isj,is->',
                                    trade_baseline,
                                    e)
    l_term_1.append(term_1)
    l_term_2.append(term_2)
    l_term_3.append(term_3)
    l_term_4.append(term_4)
    l_em_reduc.append(em_reduc)


#%%

fig,ax = plt.subplots(figsize = (14,10))
ax2 = ax.twinx()
ax.plot(np.linspace(0,1e-3,101)[:-1]*1e6,l_term_1, label='term 1',lw=3)
ax.plot(np.linspace(0,1e-3,101)[:-1]*1e6,l_term_2, label='term 2',lw=3)
ax.plot(np.linspace(0,1e-3,101)[:-1]*1e6,l_term_3, label='term 3',lw=3)
ax.plot(np.linspace(0,1e-3,101)[:-1]*1e6,l_term_4, label='term 4',lw=3)
ax2.plot(np.linspace(0,1e-3,101)[:-1]*1e6,l_em_reduc, 
         label='Emissions reduction',color='black'
         ,lw=3)

ax.legend(loc='lower left',fontsize = 20)
ax2.legend(loc='lower center',fontsize = 20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax2.tick_params(axis='both', which='major', labelsize=15)
ax2.grid([])
ax.set_xlabel('Carbon tax',fontsize = 20)
ax.set_ylabel('Decomposition effects',fontsize = 20)
ax2.set_ylabel('Emissions reduction',fontsize = 20)
# plt.savefig('graph_images/decomposition_different_effects_increasing_tax.pdf')
plt.show()
#%%

fig,ax = plt.subplots(figsize = (14,10))
# ax2 = ax.twinx()
ax.plot(np.linspace(0,1e-3,101)[:-1]*1e6,l_term_1, label='term 1',lw=3)
ax.plot(np.linspace(0,1e-3,101)[:-1]*1e6,l_term_2, label='term 2',lw=3)
ax.plot(np.linspace(0,1e-3,101)[:-1]*1e6,l_term_3, label='term 3',lw=3)
ax.plot(np.linspace(0,1e-3,101)[:-1]*1e6,l_term_4, label='term 4',lw=3)
ax.plot(np.linspace(0,1e-3,101)[:-1]*1e6,l_em_reduc, 
         label='Marginal emissions reduction dE/E',color='black'
         ,lw=3)

ax.legend(loc='lower right',fontsize = 20)
# ax2.legend(loc='lower center',fontsize = 20)
ax.tick_params(axis='both', which='major', labelsize=15)
# ax2.tick_params(axis='both', which='major', labelsize=15)
# ax2.grid([])
ax.set_xlabel('Carbon tax',fontsize = 20)
ax.set_ylabel('Decomposition effects',fontsize = 20)
# ax2.set_ylabel('Marginal emissions reduction dE/E',fontsize = 20)
# plt.savefig('graph_images/decomposition_different_effects_increasing_tax.pdf')
plt.show()

#%%
fig,ax = plt.subplots(figsize = (14,10))

ax.stackplot(np.linspace(0,1e-3,101)[:-1]*1e6,
              l_term_1, l_term_2, l_term_3,l_term_4, 
              labels=['term 1','term 2','term 3','term 4'])
ax.plot(np.linspace(0,1e-3,101)[:-1]*1e6,l_em_reduc, 
         label='Marginal emissions reduction dE/E',color='black'
         ,lw=3)
ax.legend(loc='lower right',fontsize = 20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_xlabel('Carbon tax',fontsize = 20)
# plt.savefig('graph_images/stacked_decomposition_different_effects_increasing_tax.pdf')
plt.show()
#%%
fig,ax = plt.subplots(figsize = (14,10))

ax.stackplot(np.linspace(0,1e-3,101)[:-1]*1e6,
              [term/(l_term_1[i]+l_term_2[i]+l_term_3[i]+l_term_4[i]) for i,term in enumerate(l_term_1)], 
              [term/(l_term_1[i]+l_term_2[i]+l_term_3[i]+l_term_4[i]) for i,term in enumerate(l_term_2)], 
              [term/(l_term_1[i]+l_term_2[i]+l_term_3[i]+l_term_4[i]) for i,term in enumerate(l_term_3)], 
              [term/(l_term_1[i]+l_term_2[i]+l_term_3[i]+l_term_4[i]) for i,term in enumerate(l_term_4)], 
              labels=['term 1','term 2','term 3','term 4'])
# ax.plot(np.linspace(0,1e-3,101)[:-1]*1e6,l_em_reduc, 
#          label='Marginal emissions reduction',color='black'
#          ,lw=3)
ax.legend(fontsize = 20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_xlabel('Carbon tax',fontsize = 20)
# plt.savefig('graph_images/fontagne_normalized_stacked_decomposition_different_effects_increasing_tax.pdf')
plt.show()
