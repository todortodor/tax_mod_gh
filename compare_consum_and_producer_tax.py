#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:05:32 2023

@author: slepot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from adjustText import adjust_text
import matplotlib as mpl
from tqdm import tqdm
# from labellines import labelLines
# import treatment_funcs as t
import lib.data_funcs as d
import lib.treatment_funcs as t
import os
import matplotlib.patches as patches
import scipy.optimize as opt

sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 25})
plt.rcParams['text.usetex'] = False

data_path = 'data/'
results_path = 'results/'

save_all = False
save_format = 'eps'
save_format = 'png'

# carb_cost_list = np.linspace(0,1e-3,1001)
carb_cost_list = [1e-4]
elast_path = 'cp_estimate_allyears.csv'

save_path = 'presentation_material/'+elast_path[:-4]+'_world_va/'

try:
    os.mkdir(save_path)
except:
    pass

eta_path = elast_path
sigma_path = elast_path

EU = d.countries_from_fta('EU')

taxed_countries_list = [None]
taxing_countries_list = [None]
taxed_sectors_list = [None]
specific_taxing_list = [None]
fair_tax_list = [False]
pol_pay_tax_list=[False]
tax_scheme_list = ['consumer','producer']
y  = 2018
year = str(y)
years = [y]
dir_num = [50,60]

cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                      taxed_sectors_list,specific_taxing_list,fair_tax_list,
                      pol_pay_tax_list,tax_scheme_list)

cases = [cas for cas in cases if cas['eta_path'] == cas['sigma_path']]

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

b = baselines[y]
carb_taxes = np.array([sol.params.carb_cost*1e6  for sol in sols])
t_index = np.argmin(np.abs(carb_taxes-100))
sol = sols[t_index]
N = b.country_number
S = b.sector_number

sector_map = pd.read_csv(data_path+'industry_labels_after_agg_expl_wgroup.csv')
sector_map['sector'] = sector_map['ind_code'].str.replace('D','')
sector_map.set_index('sector',inplace=True)

#%%

sols_cons_tax = []
sols_prod_tax = []

for sol in sols:
    print(sol.params.tax_scheme)
    if sol.params.tax_scheme == 'consumer':
        sols_cons_tax.append(sol)
    if sol.params.tax_scheme == 'producer':
        sols_prod_tax.append(sol)

#%%

welfare_cons_tax = sols_cons_tax[0].utility
welfare_prod_tax = sols_prod_tax[0].utility

x = b.va.groupby(['col_country']).sum()

fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(b.va.groupby(['col_country']).sum().values.squeeze(),welfare_cons_tax,label='consumer tax')
plt.scatter(b.va.groupby(['col_country']).sum().values.squeeze(),welfare_prod_tax,label='producer tax')
# plt.scatter(b.va.groupby(['col_country']).sum().values.squeeze(),transfers_eu_style,label='eu_style')

texts = [plt.text(
        x.loc[i,'value'], 
        welfare_cons_tax.loc[i,'hat'], 
        i,
        size=12,
        # color=colors[g]
        ) 
        for i in welfare_cons_tax.index]
# plt.yscale('symlog',linthresh=100)
plt.xscale('log')
adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))
ax.set_ylabel('Welfare change')
ax.set_xlabel('GDP (Mio.$)')
plt.legend()
plt.show()

#%%

transfers_cons = pd.read_csv('results/2018_51/3_results_contrib.csv',index_col=0)
transfers_prod = pd.read_csv('results/2018_61/3_results_contrib.csv',index_col=0)
transfers_eu_style = pd.read_csv('results/2018_71/3_results_contrib.csv',index_col=0)

x = b.va.groupby(['col_country']).sum()

fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(b.va.groupby(['col_country']).sum().values.squeeze(),transfers_cons,label='consumer tax')
ax.scatter(b.va.groupby(['col_country']).sum().values.squeeze(),transfers_prod,label='producer tax')
# plt.scatter(b.va.groupby(['col_country']).sum().values.squeeze(),transfers_eu_style,label='eu_style')

texts = [plt.text(
        x.loc[i,'value'], 
        transfers_cons.loc[i,'value'], 
        i,
        size=12,
        # color=colors[g]
        ) 
        for i in transfers_cons.index]
# plt.yscale('symlog',linthresh=100)
plt.xscale('log')
adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))
plt.legend()
plt.show()


#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from adjustText import adjust_text
import matplotlib as mpl
from tqdm import tqdm
# from labellines import labelLines
# import treatment_funcs as t
import lib.data_funcs as d
import lib.treatment_funcs as t
import os
import matplotlib.patches as patches
import scipy.optimize as opt

sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 25})
plt.rcParams['text.usetex'] = False

data_path = 'data/'
results_path = 'results/'

save_all = False
save_format = 'eps'
save_format = 'png'

# carb_cost_list = np.linspace(0,1e-3,1001)
carb_cost_list = [1e-4]
elast_path = 'cp_estimate_allyears.csv'

save_path = 'presentation_material/'+elast_path[:-4]+'_world_va/'

try:
    os.mkdir(save_path)
except:
    pass

eta_path = elast_path
sigma_path = elast_path

EU = d.countries_from_fta('EU')

taxed_countries_list = [None]
taxing_countries_list = [EU]
taxed_sectors_list = [None]
specific_taxing_list = [None]
fair_tax_list = [False]
pol_pay_tax_list=[False]
tax_scheme_list = ['consumer','producer','eu_style']
y  = 2018
year = str(y)
years = [y]
dir_num = [55,56,65,66,75,76]

cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                      taxed_sectors_list,specific_taxing_list,fair_tax_list,
                      pol_pay_tax_list,tax_scheme_list)

cases = [cas for cas in cases if cas['eta_path'] == cas['sigma_path']]

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

b = baselines[y]
carb_taxes = np.array([sol.params.carb_cost*1e6  for sol in sols])
t_index = np.argmin(np.abs(carb_taxes-100))
sol = sols[t_index]
N = b.country_number
S = b.sector_number

sector_map = pd.read_csv(data_path+'industry_labels_after_agg_expl_wgroup.csv')
sector_map['sector'] = sector_map['ind_code'].str.replace('D','')
sector_map.set_index('sector',inplace=True)


#%%

welfare_prod_tax = sols[1].utility
welfare_prod_tax_cba = sols[2].utility

x = b.va.groupby(['col_country']).sum()

fig,ax = plt.subplots(figsize=(16,8))
# ax.scatter(b.va.groupby(['col_country']).sum().values.squeeze(),welfare_prod_tax,label='Without BCA')
# ax.scatter(b.va.groupby(['col_country']).sum().values.squeeze(),welfare_prod_tax_cba,label='With BCA')
# # plt.scatter(b.va.groupby(['col_country']).sum().values.squeeze(),transfers_eu_style,label='eu_style')
bars = ax.bar(EU+countries[1:],welfare_prod_tax.loc[EU+countries[1:]].hat*100-100,label='Without BCA')
ax.scatter(EU+countries[1:],welfare_prod_tax_cba.loc[EU+countries[1:]].hat*100-100,label='With BCA')
# plt.scatter(b.va.groupby(['col_country']).sum().values.squeeze(),transfers_eu_style,label='eu_style')

# texts = [plt.text(
#         x.loc[i,'value'], 
#         welfare_prod_tax.loc[i,'hat'], 
#         i,
#         size=10,
#         color=sns.color_palette()[0]
#         ) 
#         for i in welfare_prod_tax.index]

# texts = [plt.text(
#         x.loc[i,'value'], 
#         welfare_prod_tax_cba.loc[i,'hat'], 
#         i,
#         size=10,
#         color=sns.color_palette()[1]
#         ) 
#         for i in welfare_prod_tax.index]
# plt.yscale('symlog',linthresh=100)
# plt.xscale('log')
# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))
ax.bar_label(bars,
              labels=EU+countries[1:],
              rotation=90,
              label_type = 'edge',
              padding=5,
              fontsize=12,
              # color=colors[g],
              zorder=99)
ax.set_xticklabels(['']
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)
ax.vlines(x=len(EU)-0.5,
            ymin=-1,
            ymax=1,
            lw=3,
            ls = '--',
            color = 'r',label='EU')
ax.set_ylabel('Welfare change')
# ax.set_xlabel('GDP (Mio.$)')
plt.title('EU taxing only')
plt.legend()
plt.show()

#%%

df = welfare_prod_tax_cba.loc[EU+countries[1:]]-welfare_prod_tax.loc[EU+countries[1:]]
df = df.sort_values('hat')
df.columns = ['Gains from implementing CBA']

fig,ax = plt.subplots(figsize=(16,8))
bars = ax.bar(df.index.tolist(),df['Gains from implementing CBA'],label='Gains from implemeting CBA in the EU')

ax.bar_label(bars,
              labels=df.index.tolist(),
              rotation=90,
              label_type = 'edge',
              padding=5,
              fontsize=12,
              # color=colors[g],
              zorder=99)
ax.set_xticklabels(['']
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)
ax[0,0].vlines(x=100,
            ymin=0,
            ymax=y_100,
            lw=3,
            ls = '--',
            color = color)
ax.set_ylabel('Welfare change')
# ax.set_xlabel('GDP (Mio.$)')
plt.title('EU taxing only')
plt.legend()
plt.show()

#%% compute trade
trade_by_tax_scheme = {}
share_traded_by_tax_scheme = {}

for tax_scheme in ['consumer','producer']:
    if tax_scheme == 'consumer':
        sols = sols_cons_tax
    if tax_scheme == 'producer':
        sols = sols_prod_tax
    trade = baselines[y].iot.groupby(level=[0,1,2]).sum()
    trade['cons'] = baselines[y].cons.value
    trade['baseline'] = trade.value + trade.cons
    trade = trade[['baseline']]
    print("Computing trade")
    l_trade = []
    for i,sol in tqdm(enumerate(sols)):
        l_trade.append( sol.iot.value.values.reshape((N,S,N,S)).sum(axis=-1).ravel()
                       + sol.cons.value.values )
    
    trades = pd.concat([pd.Series(l_t) for l_t in l_trade],axis=1)
    trades.index = trade.index
    trade = pd.concat([trade,trades],axis=1)
    
    print('Computing share of output traded')
    
    iot_traded_unit = b.iot.copy()
    iot_traded_unit['value'] = 1
    iot_traded_unit.loc[iot_traded_unit.query("row_country == col_country").index, 'value'] = 0
    cons_traded_unit = b.cons.copy()
    cons_traded_unit['value'] = 1
    cons_traded_unit.loc[cons_traded_unit.query("row_country == col_country").index, 'value'] = 0
    
    share_traded = np.array([((sol.cons.value.to_numpy() * cons_traded_unit.value.to_numpy()).sum() + (sol.iot.value.to_numpy()  * iot_traded_unit.value.to_numpy()).sum())*100 /\
                             (sol.cons.value.to_numpy().sum()+sol.iot.value.to_numpy().sum()) for sol in sols])
    
    total_output = np.array([sol.output.sum() for sol in sols])

    trade_by_tax_scheme[tax_scheme] = trade
    share_traded_by_tax_scheme[tax_scheme] = share_traded
    
#%%

fig, ax = plt.subplots(2,2,figsize=(16,12))

for tax_scheme in ['consumer','producer']:
    if tax_scheme == 'consumer':
        sols = sols_cons_tax
        ls = ':'
    if tax_scheme == 'producer':
        sols = sols_prod_tax
        ls = '-'
        
    share_traded = share_traded_by_tax_scheme[tax_scheme]

    carb_taxes = np.array([sol.params.carb_cost*1e6  for sol in sols_cons_tax])
    norm_emissions = np.array([sol.co2_prod.value.sum()
                                /baselines[y].co2_prod.value.sum() 
                               for sol in sols])
    norm_emissions_real = np.array([sol.run.emissions/baselines[y].co2_prod.value.sum() for sol in sols])
    norm_gdp = np.array([sol.va.value.sum()/baselines[y].va.value.sum() for sol in sols])
    norm_real_income = np.array([sol.run.utility for sol in sols])
    norm_total_output = np.array([sol.output.value.sum()/baselines[y].output.value.sum() for sol in sols])
    
    color = 'g'
    
    # Upper left - Emissions
    ax[0,0].plot(carb_taxes,norm_emissions,lw=4,color=color,label='Global emissions',ls=ls)
    ax[0,0].legend()
    ax[0,0].set_xlabel('')
    ax[0,0].tick_params(axis='x', which='both',
        bottom=False,
        top=False,
        labelbottom=False)
    
    
    y_100 = np.array(norm_emissions)[np.argmin(np.abs(carb_taxes-100))]
    
    ax[0,0].vlines(x=100,
                ymin=0,
                ymax=y_100,
                lw=3,
                ls = '--',
                color = color)
    
    ax[0,0].hlines(y=y_100,
                xmin=0,
                xmax=100,
                lw=3,
                ls = '--',
                color = color)
    
    ax[0,0].margins(y=0)
    
    ax[0,0].annotate(str((100*(y_100-1)).round(0))+'%',
                 xy=(100,y_100),
                 xytext=(0,0),
                 textcoords='offset points',color=color)
    
    ax[0,0].set_ylim(norm_emissions.min(),norm_emissions.max()+0.05)
    ax[0,0].set_ylim(-0.05,1.05)
    
    # Upper right - GDP
    color = 'b'
    
    ax[1,1].plot(carb_taxes,share_traded,lw=4,ls=ls)
    ax[1,1].set_xlabel('')
    ax[0,1].tick_params(axis='x', which='both',
        bottom=False,
        top=False,
        labelbottom=False)
    
    ax[1,1].legend(['Share of output traded (%)'])
    ax[1,1].margins(y=0)
    
    ax[1,1].set_ylim(10,20)
    
    # Bottom left - Welfare
    color = 'r'
    
    ax[1,0].plot(carb_taxes,norm_real_income,lw=4,color='r',ls=ls)
    ax[1,0].legend(['Real income'])
    ax[1,0].set_xlabel('Carbon tax ($/ton of CO2eq.)')
    
    y_100 = norm_real_income[np.argmin(np.abs(carb_taxes-100))]
    
    ax[1,0].vlines(x=100,
                ymin=norm_real_income.min(),
                ymax=y_100,
                lw=3,
                ls = '--',
                color = color)
    
    ax[1,0].hlines(y=y_100,
                xmin=0,
                xmax=100,
                lw=3,
                ls = '--',
                color = color)
    
    ax[1,0].margins(y=0)
    
    ax[1,0].set_ylim(norm_real_income.min(),1.005)
    
    ax[1,0].annotate(str((100*(y_100-1)).round(1))+'%',
                  xy=(100,y_100),
                  xytext=(0,0),
                  textcoords='offset points',color=color)

# Bottom right summary
ax[0, 1].axis("off")
ax[0, 1].plot([],[],color='k',ls=':',label='Consumer taxed scheme')
ax[0, 1].plot([],[],color='k',ls='-',label='Producer taxed scheme')
ax[0, 1].legend(loc='center')
# color = 'k'

# ax[0,1].plot(carb_taxes,norm_total_output,lw=4,color='k')
# ax[0,1].legend(['World gross output'])
# ax[1,1].set_xlabel('Carbon tax ($/ton of CO2eq.)')

# y_100 = norm_total_output[np.argmin(np.abs(carb_taxes-100))]

# ax[0,1].vlines(x=100,
#             ymin=norm_total_output.min(),
#             ymax=y_100,
#             lw=3,
#             ls = '--',
#             color = color)

# ax[0,1].hlines(y=y_100,
#             xmin=0,
#             xmax=100,
#             lw=3,
#             ls = '--',
#             color = color)

# ax[0,1].margins(y=0)

# ax[0,1].set_ylim(norm_total_output.min(),1.005)

# ax[0,1].annotate(str((100*(y_100-1)).round(1))+'%',
#               xy=(105,y_100),
#               xytext=(0,0),
#               textcoords='offset points',color=color)

plt.tight_layout()

# if save or save_all:
#     plt.savefig(save_path+'macro_effects.'+save_format,format=save_format)
plt.show()

#%%
fig, ax = plt.subplots(figsize=(16,12))

for tax_scheme in ['consumer','producer']:
    if tax_scheme == 'consumer':
        sols = sols_cons_tax
        ls = ':'
    if tax_scheme == 'producer':
        sols = sols_prod_tax
        ls = '-'
        
    share_traded = share_traded_by_tax_scheme[tax_scheme]

    carb_taxes = np.array([sol.params.carb_cost*1e6  for sol in sols])
    norm_emissions = np.array([sol.co2_prod.value.sum()
                                /baselines[y].co2_prod.value.sum() 
                               for sol in sols])
    norm_emissions_real = np.array([sol.run.emissions/baselines[y].co2_prod.value.sum() for sol in sols])
    norm_gdp = np.array([sol.va.value.sum()/baselines[y].va.value.sum() for sol in sols])
    norm_real_income = np.array([sol.run.utility for sol in sols])
    norm_total_output = np.array([sol.output.value.sum()/baselines[y].output.value.sum() for sol in sols])
    
    print(norm_emissions[-1])
    
    # color = 'g'
    
    # Upper left - Emissions
    ax.plot(carb_taxes,norm_real_income,
                  color=color,
                 label=tax_scheme,
                 ls=ls)
    ax.legend()
    ax.set_xlabel('')
    ax.tick_params(axis='x', which='both',
        bottom=False,
        top=False,
        labelbottom=False)
    
    
    # y_100 = np.array(norm_emissions)[np.argmin(np.abs(carb_taxes-100))]
    
    # ax.vlines(x=100,
    #             ymin=0,
    #             ymax=y_100,
    #             lw=3,
    #             ls = '--',
    #             color = color)
    
    # ax.hlines(y=y_100,
    #             xmin=0,
    #             xmax=100,
    #             lw=3,
    #             ls = '--',
    #             color = color)
    
    # ax.margins(y=0)
    
    # ax.annotate(str((100*(y_100-1)).round(0))+'%',
    #              xy=(100,y_100),
    #              xytext=(0,0),
    #              textcoords='offset points',color=color)
    
    # ax.set_ylim(norm_emissions.min(),norm_emissions.max()+0.05)
    # ax.set_ylim(-0.05,1.05)

plt.show()
