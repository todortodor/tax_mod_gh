#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 21:41:15 2022

@author: slepot
"""

#%% import libraries

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

sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 25})
plt.rcParams['text.usetex'] = False

#%% load data


data_path = 'data/'
results_path = 'results/'




save_all = True
save_format = 'eps'

carb_cost_list = np.linspace(0,1e-3,101)
# carb_cost_list = [1e-4]
# eta_path = ['elasticities_agg1.csv']
# sigma_path = ['elasticities_agg1.csv']

list_of_elasticities = ['rescaled_to_4elasticities_agg1.csv',
 'rescaled_to_5elasticities_agg1.csv',
 'rescaled_to_4_output_weightedelasticities_agg1.csv',
 'rescaled_to_5_output_weightedelasticities_agg1.csv',
 'rescaled_to_4elasticities_agg2.csv',
 'rescaled_to_5elasticities_agg2.csv',
 'rescaled_to_4_output_weightedelasticities_agg2.csv',
 'rescaled_to_5_output_weightedelasticities_agg2.csv']

for elast_path in list_of_elasticities:

    save_path = 'presentation_material/'+elast_path[:-4]+'/'
    
    try:
        os.mkdir(save_path)
    except:
        pass
    
    eta_path = elast_path
    sigma_path = elast_path
    
    taxed_countries_list = [None]
    taxing_countries_list = [None]
    taxed_sectors_list = [None]
    specific_taxing_list = [None]
    fair_tax_list = [False]
    y  = 2018
    year = str(y)
    years = [y]
    dir_num = 6
    
    cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                          taxed_sectors_list,specific_taxing_list,fair_tax_list)
    
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
    
    #%% compute trade
    
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
    
    #%% compute decomposition
    
    def X(trade):
        res = np.einsum('isj->',
                      trade)
        return res
    
    def alpha_s(trade):
        num = np.einsum('isj->s',trade)
        denom = trade.sum()
        return num/denom
    
    def alpha_is(trade):
        num = np.einsum('isj->is',trade)
        denom = np.einsum('isj->s',trade)
        return np.einsum('is,s->is',num,1/denom)
    
    def e_s(trade,e):
        return np.einsum('is,is->s',alpha_is(trade),e)
    
    def e_scal(trade,e):
        return np.einsum('s,s->',alpha_s(trade),e_s(trade,e))
    
    def epsilon_s(trade,e):
        return np.einsum('s,s,,->s',
                         np.einsum('isj->s',trade),
                         e_s(trade,e),
                         1/np.einsum('isj->',trade),
                         1/e_scal(trade,e))
    
    def epsilon_is(trade,e):
        return np.einsum('is,is,,->is',
                         np.einsum('isj->is',trade),
                         e,
                         1/np.einsum('isj->',trade),
                         1/e_scal(trade,e))
    
    
    
    l_term_1 = []
    l_term_2 = []
    l_term_3 = []
    # l_term_4 = []
    l_em_reduc = []
    e = b.co2_intensity.value.values.reshape((N,S))
    print("Computing decomposition")
    for i in tqdm(range(len(sols)-1)):
        trade_baseline = trade[i].values.reshape((N,S,N))
        # trade_baseline = trade['baseline'].values.reshape((N,S,N))
        trade_cf = trade[i+1].values.reshape((N,S,N)) 
        
        term_1 = (X(trade_cf) - X(trade_baseline))/X(trade_baseline)
    
        term_2 = np.einsum('s,s,s->s',
                           epsilon_s(trade_baseline,e),
                           alpha_s(trade_cf)-alpha_s(trade_baseline),
                           1/alpha_s(trade_baseline))
    
        term_3 = np.einsum('is,is,is->is',
                   epsilon_is(trade_baseline,e),
                   alpha_is(trade_cf)-alpha_is(trade_baseline),
                   1/alpha_is(trade_baseline))
    
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
        l_em_reduc.append(em_reduc)
    
    l_term_1 = np.array(l_term_1)
    l_term_2 = np.array(l_term_2)
    l_term_3 = np.array(l_term_3)
    
    l_em_reduc = np.array(l_em_reduc)
        
    d_term = {
        'term_1':l_term_1,
        'term_2':l_term_2,
        'term_3':l_term_3,
              }
    
    d_term_summed = {key: [v.sum() for v in l_term] for key, l_term in tqdm(d_term.items())}
    
    emiss_reduc_contrib = {}
    for term, l_term in d_term_summed.items():
        emiss_reduc_contrib[term] = np.array([l_term[k]*sols[k].co2_prod.value.sum()/b.co2_prod.value.sum() 
                                     for k in range(len(l_term))])
        
    l_em_incr = np.array([l_em_reduc[k]*sols[k].co2_prod.value.sum()/b.co2_prod.value.sum() 
                 for k in range(len(l_em_reduc))])
    
    
    term_labels = {
        'term_1':'Scale',
        'term_2':'Composition sectors',
        'term_3':'Composition countries'
        # 'em_reduc':l_em_reduc
              }
    
    #%% Plot macro effects
    
    save = False
    
    print('Plotting welfare and GDP cost corresponding to a carbon tax')
    
    carb_taxes = np.array([sol.params.carb_cost*1e6  for sol in sols])
    norm_emissions = np.array([(sol.output.value.values*b.co2_intensity.value.values/1e6).sum()/baselines[y].co2_prod.value.sum() 
                               for sol in sols])
    norm_emissions_real = np.array([sol.run.emissions/baselines[y].co2_prod.value.sum() for sol in sols])
    norm_gdp = np.array([sol.va.value.sum()/baselines[y].va.value.sum() for sol in sols])
    norm_real_income = np.array([sol.run.utility for sol in sols])
    norm_total_output = np.array([sol.output.value.sum()/baselines[y].output.value.sum() for sol in sols])
    
    fig, ax = plt.subplots(2,2,figsize=(16,12))
    
    color = 'g'
    
    # Upper left - Emissions
    ax[0,0].plot(carb_taxes,norm_emissions,lw=4,color=color,label='Global emissions')
    # ax[0,0].plot(carb_taxes,norm_emissions_real,lw=4,ls=':',color=color,label='Global emissions real')
    ax[0,0].legend()
    ax[0,0].set_xlabel('')
    ax[0,0].tick_params(axis='x', which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    ax[0,0].set_xlim(0,1000)
    
    y_100 = np.array(norm_emissions)[np.argmin(np.abs(carb_taxes-100))]
    # y_0 = runs_low_carb_cost.iloc[0].emissions
    
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
    
    # Upper right - GDP
    color = 'b'
    
    ax[1,1].plot(carb_taxes,share_traded,lw=4)
    ax[1,1].set_xlabel('')
    ax[0,1].tick_params(axis='x', which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    ax[1,1].set_xlim(0,1000)
    ax[1,1].legend(['Share of output traded (%)'])
    
    # y_100 = share_traded[np.argmin(np.abs(carb_taxes-100))]
    # 
    # ax[0,1].vlines(x=100,
    #             ymin=norm_gdp.min(),
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
    
    ax[1,1].margins(y=0)
    
    # ax[0,1].annotate(str(100*(1-y_100).round(1)),
    #               xy=(100,y_100),
    #               xytext=(0,0),
    #               textcoords='offset points',color=color)
    
    ax[1,1].set_ylim(10,20)
    
    # Bottom left - Welfare
    color = 'r'
    
    ax[1,0].plot(carb_taxes,norm_real_income,lw=4,color='r')
    ax[1,0].legend(['Real income'])
    ax[1,0].set_xlabel('Carbon tax ($/ton of CO2eq.)')
    ax[1,0].set_xlim(0,1000)
    # ax[1,0].set_ylim(min(utility),1.001)
    
    y_100 = norm_real_income[np.argmin(np.abs(carb_taxes-100))]
    # y_0 = runs_low_carb_cost.iloc[0].emissions
    
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
    color = 'k'
    
    ax[0,1].plot(carb_taxes,norm_total_output,lw=4,color='k')
    ax[0,1].legend(['World gross output'])
    ax[1,1].set_xlabel('Carbon tax ($/ton of CO2eq.)')
    ax[0,1].set_xlim(0,1000)
    # ax[1,1].set_ylim(10,15)
    
    y_100 = norm_total_output[np.argmin(np.abs(carb_taxes-100))]
    # y_0 = runs_low_carb_cost.iloc[0].emissions
    
    ax[0,1].vlines(x=100,
                ymin=norm_total_output.min(),
                ymax=y_100,
                lw=3,
                ls = '--',
                color = color)
    
    ax[0,1].hlines(y=y_100,
                xmin=0,
                xmax=100,
                lw=3,
                ls = '--',
                color = color)
    
    ax[0,1].margins(y=0)
    
    ax[0,1].set_ylim(norm_total_output.min(),1.005)
    
    ax[0,1].annotate(str((100*(y_100-1)).round(1))+'%',
                  xy=(105,y_100),
                  xytext=(0,0),
                  textcoords='offset points',color=color)
    
    plt.tight_layout()
    
    if save or save_all:
        plt.savefig(save_path+'macro_effects.'+save_format,format=save_format)
    plt.show()
    
    #%% plot decomposition
    
    cumul_terms = {key: np.array([value[:i].sum()
                          for i in range(len(value))]) for key, value in emiss_reduc_contrib.items()}
    
    fig,ax = plt.subplots(figsize = (16,12))
    
    ax.stackplot(carb_taxes[1:],
                  [term for term in cumul_terms.values()],
                  labels=[term for term in cumul_terms.keys()])
    ax.plot(carb_taxes[1:],[l_em_incr[:i].sum() for i in range(len(l_em_incr))], 
              label='Emissions',color='black'
              ,lw=3)
    # ax.plot(carb_taxes,norm_emissions-1, 
    #           label='Emissions real',color='y'
    #           ,lw=3)
    ax.legend(loc='lower left',fontsize = 20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel('Carbon tax',fontsize = 20)
    
    # if save or save_all:
    #     plt.savefig(save_path+'decomposition_stacked.'+save_format,format=save_format)
    plt.show()
    
    #%% plot decomposition diff labels
    
    cumul_terms = {key: np.array([value[:i].sum()
                          for i in range(len(value))]) for key, value in emiss_reduc_contrib.items()}
    
    fig,ax = plt.subplots(figsize = (16,16))
    
    ax.stackplot(carb_taxes[1:],
                  [term for term in cumul_terms.values()],
                  labels=[term_labels[term] for term in cumul_terms.keys()])
    ax.plot(carb_taxes[1:],[l_em_incr[:i].sum() for i in range(len(l_em_incr))], 
              label='Emissions',color='black'
              ,lw=3)
    # ax.plot(carb_taxes,norm_emissions-1, 
    #           label='Emissions real',color='y'
    #           ,lw=3)
    ax.legend(loc='lower left',fontsize = 30)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_xlabel('Carbon tax',fontsize = 35)
    
    if save or save_all:
        plt.savefig(save_path+'decomposition_stacked_diff_labels.'+save_format,format=save_format)
    plt.show()
    
    #%% plot decomposition normalized
    
    cumul_terms = {key: np.array([value[:i].sum()
                          for i in range(len(value))]) for key, value in emiss_reduc_contrib.items()}
    
    sum_terms = cumul_terms['term_1']+cumul_terms['term_2']+cumul_terms['term_3']
    
    fig,ax = plt.subplots(figsize = (16,16))
    
    ax.stackplot(carb_taxes[1:],
                  [-term/sum_terms for term in cumul_terms.values()],
                  labels=[term_labels[term] for term in cumul_terms.keys()])
    # ax.plot(carb_taxes[1:],[l_em_incr[:i].sum() for i in range(len(l_em_incr))], 
    #           label='Emissions',color='black'
    #           ,lw=3)
    # ax.plot(carb_taxes,norm_emissions-1, 
    #           label='Emissions real',color='y'
    #           ,lw=3)
    # ax.legend(loc='lower left',fontsize = 20)
    offset = 0
    for name,term in cumul_terms.items():
        loc = 50
        ax.text(carb_taxes[1:][loc], -(term[1:]/sum_terms[1:])[loc]/2+offset, term_labels[name]+' : '+str(((term[1:]/sum_terms[1:]).mean()*100).round(1))+'%',
                ha='center', va='center',color='white',fontsize = 35)
        offset = offset-(term[1:]/sum_terms[1:])[loc]
    
    # loc = y2.argmax()
    # ax.text(loc, y1[loc] + y2[loc]*0.33, areaLabels[1])
    
    # loc = y3.argmax()
    # ax.text(loc, y1[loc] + y2[loc] + y3[loc]*0.75, areaLabels[2]) 
    # ax.legend(fontsize = 20)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_xlabel('Carbon tax',fontsize = 35)
    
    if save or save_all:
        plt.savefig(save_path+'decomposition_stacked_norm.'+save_format,format=save_format)
    plt.show()
    
    #%% plot composition term epsilon(alpha)
    # font = {'weight' : 'bold',
    #         'size'   : 150}
    # plt.rcdefaults()
    # mpl.rc('font', **font)
    # plt.rcParams.update({'font.size': 60})
    # trade_baseline = trade[10].values.reshape((N,S,N))
    trade_baseline = trade['baseline'].values.reshape((N,S,N))
    trade_cf = trade[10].values.reshape((N,S,N)) 
    
    fig, ax = plt.subplots(figsize=(25,15))
    # ax2 = ax.twinx()
    colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    texts = []
    data_base = sector_map.copy()
    data_base['y'] = epsilon_s(trade_baseline, e)
    data_base['x'] = alpha_s(trade_baseline)
    for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
        data = data_base.loc[data_base['group_label'] == group_label]
        ax.scatter(data.x,data.y, label = group_label,color=colors[g])
        texts_group = [plt.text(data.x.iloc[i], 
                data.y.iloc[i], 
                industry,
                size=25,
                color=colors[g]) 
                for i,industry in enumerate(data.industry)]     # For kernel density
        texts = texts+texts_group
    # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # ax2.grid([])
    ax.set_xlabel(r'$\alpha_s$',fontsize = 30)
    ax.set_ylabel(r'$\epsilon_s$',fontsize = 30,rotation=0,labelpad=30)
    ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
    # ax.set_xlim([1e-2,3e1])
    plt.legend(fontsize = 25)
    plt.xscale('log')
    plt.yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=25)
    # ax.tick_params(axis='both', which='minor', labelsize=20)
    
    
    adjust_text(texts, precision=0.001,
            expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
            force_text=(0.01, 0.25), force_points=(0.01, 0.25),
            arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                            ))
    # if save or save_all:
    #     plt.savefig(save_path+'term_2_eps_alpha.'+save_format,format=save_format)
    if save or save_all:
        plt.savefig(save_path+'term_2_eps_alpha.pdf',format='pdf')
    
    plt.show()
    
    #%% plot composition term d(alpha)/alpha func of epsilon
    
    # trade_baseline = trade[10].values.reshape((N,S,N))
    trade_baseline = trade['baseline'].values.reshape((N,S,N))
    trade_cf = trade[10].values.reshape((N,S,N)) 
    
    fig, ax = plt.subplots(figsize=(25,15))
    # ax2 = ax.twinx()
    colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    texts = []
    data_base = sector_map.copy()
    data_base['x'] = epsilon_s(trade_baseline, e)
    data_base['y'] = (alpha_s(trade_cf)-alpha_s(trade_baseline))/alpha_s(trade_baseline)
    for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
        data = data_base.loc[data_base['group_label'] == group_label]
        ax.scatter(data.x,data.y, label = group_label,color=colors[g])
        texts_group = [plt.text(data.x.iloc[i], 
                data.y.iloc[i], 
                industry,
                size=20,
                color=colors[g]) 
                for i,industry in enumerate(data.industry)]     # For kernel density
        texts = texts+texts_group
    # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # ax2.grid([])
    ax.set_xlabel(r'$\epsilon_s$',fontsize = 40)
    ax.set_ylabel(r'$\left(\frac{d\alpha_s}{\alpha_s}\right)_{_2}$',fontsize = 40, labelpad=60,rotation=0)
    # ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
    # ax.set_xlim([1e-2,3e1])
    plt.legend(fontsize = 25)
    plt.xscale('log')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    # plt.yscale('log')
    
    plt.axhline(y=0, color='k', linestyle='-',lw=0.5)
    
    adjust_text(texts, precision=0.001,
            expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
            force_text=(0.01, 0.25), force_points=(0.01, 0.25),
            arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                            ))
    if save or save_all:
        plt.savefig(save_path+'term_2_d_alpha_over_alpha_func_of_eps.'+save_format,format=save_format)
    
    plt.show()
    
    #%% plot composition term epsilon/alpha(alpha) schema
    
    # trade_baseline = trade[10].values.reshape((N,S,N))
    trade_baseline = trade['baseline'].values.reshape((N,S,N))
    trade_cf = trade[10].values.reshape((N,S,N)) 
    
    fig, ax = plt.subplots(figsize=(25,15))
    # ax2 = ax.twinx()
    colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    texts = []
    data_base = sector_map.copy()
    data_base['y'] = epsilon_s(trade_baseline, e)/alpha_s(trade_baseline)
    data_base['x'] = alpha_s(trade_baseline)
    data_base = data_base.loc[data_base.industry.isin(['Basic metals','Construction'])]
    # for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    data = data_base
    ax.scatter(data.x,data.y)
    texts_group = [plt.text(data.x.iloc[i], 
            data.y.iloc[i], 
            industry,
            size=20)
            # color=colors[g]) 
            for i,industry in enumerate(data.industry)]     # For kernel density
    texts = texts+texts_group
    # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # ax2.grid([])
    ax.set_xlabel('Alpha',fontsize = 30)
    ax.set_ylabel('Epsilon/Alpha',fontsize = 30)
    
    # plt.legend(fontsize = 25)
    # plt.xscale('log')
    # plt.yscale('log')
    
    rect1 = patches.Rectangle((0, data.loc[data.industry == 'Basic metals'].iloc[0]['y']), 
                              data.loc[data.industry == 'Basic metals'].iloc[0]['x'], 
                              -data[data.industry == 'Basic metals'].iloc[0]['y'], 
                              alpha = 0.5,
                             # linewidth=1, 
                              edgecolor='r', 
                               facecolor=sns.color_palette()[0]
                             )
    
    rect2 = patches.Rectangle((0, data.loc[data.industry == 'Construction'].iloc[0]['y']), 
                              data.loc[data.industry == 'Construction'].iloc[0]['x'], 
                              -data[data.industry == 'Construction'].iloc[0]['y'], 
                              alpha = 0.5,
                             # linewidth=1, 
                              edgecolor='r', 
                               facecolor=sns.color_palette()[1]
                             )
    
    rect3 = patches.Rectangle((data.loc[data.industry == 'Basic metals'].iloc[0]['x'], data.loc[data.industry == 'Basic metals'].iloc[0]['y']), 
                              -0.001, 
                              -data[data.industry == 'Basic metals'].iloc[0]['y'], 
                              alpha = 0.5,
                             # linewidth=1, 
                              edgecolor='r', 
                              hatch='/',
                               facecolor=sns.color_palette()[0],
                               label = 'Emissions reduction from Basic metals'
                             )
    
    rect4 = patches.Rectangle((data.loc[data.industry == 'Construction'].iloc[0]['x'], data.loc[data.industry == 'Construction'].iloc[0]['y']), 
                              0.001, 
                              -data[data.industry == 'Construction'].iloc[0]['y'], 
                              alpha = 0.5,
                             # linewidth=1, 
                              edgecolor='r', 
                              hatch='/',
                               facecolor=sns.color_palette()[1],
                               label = 'Additional emissions from Construction'
                             )
    
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.add_patch(rect4)
    plt.legend(fontsize = 25)
    
    plt.text(data.loc[data.industry == 'Basic metals'].iloc[0]['x']-0.0005,
             -0.1,'d(alpha)',
             horizontalalignment='center')
    plt.text(data.loc[data.industry == 'Construction'].iloc[0]['x']+0.0005,
             -0.1,'d(alpha)',
             horizontalalignment='center')
    
    # adjust_text(texts, precision=0.001,
    #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    #         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
    #                         ))
    if save or save_all:
        plt.savefig(save_path+'term_2_eps_alpha_schema.pdf',format='pdf')
    
    plt.show()
    
    #%% plot composition term function of epsilon / alpha
    
    # trade_baseline = trade[10].values.reshape((N,S,N))
    trade_baseline = trade['baseline'].values.reshape((N,S,N))
    trade_cf = trade[10].values.reshape((N,S,N)) 
    
    fig, ax = plt.subplots(figsize=(25,15))
    # ax2 = ax.twinx()
    colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    texts = []
    data_base = sector_map.copy()
    data_base['x'] = epsilon_s(trade_baseline, e)/alpha_s(trade_baseline)
    data_base['y'] = alpha_s(trade_cf)-alpha_s(trade_baseline)
    for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
        data = data_base.loc[data_base['group_label'] == group_label]
        ax.scatter(data.x,data.y, label = group_label,color=colors[g])
        texts_group = [plt.text(data.x.iloc[i], 
                data.y.iloc[i], 
                industry,
                size=20,
                color=colors[g]) 
                for i,industry in enumerate(data.industry)]     # For kernel density
        texts = texts+texts_group
    # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # ax2.grid([])
    ax.set_xlabel(r'$\frac{\epsilon_s}{\alpha_s}$',fontsize = 40)
    ax.set_ylabel(r'$d\alpha_s$',fontsize = 40, labelpad=20,rotation=0)
    # ax.set_xlim([1e-2,3e1])
    plt.legend(fontsize = 25)
    plt.xscale('log')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    
    plt.axhline(y=0, color='k', linestyle='-',lw=0.5)
    
    adjust_text(texts, precision=0.001,
            expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
            force_text=(0.01, 0.25), force_points=(0.01, 0.25),
            arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                            ))
    if save or save_all:
        plt.savefig(save_path+'term_2_eps_over_alpha.'+save_format,format=save_format)
    
    plt.show()
    
    #%% plot composition term function of alpha
    
    # trade_baseline = trade[10].values.reshape((N,S,N))
    trade_baseline = trade['baseline'].values.reshape((N,S,N))
    trade_cf = trade[10].values.reshape((N,S,N)) 
    
    fig, ax = plt.subplots(figsize=(25,15))
    # ax2 = ax.twinx()
    colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    texts = []
    data_base = sector_map.copy()
    data_base['x'] = alpha_s(trade_baseline)
    data_base['y'] = alpha_s(trade_cf)-alpha_s(trade_baseline)
    for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
        data = data_base.loc[data_base['group_label'] == group_label]
        ax.scatter(data.x,data.y, label = group_label,color=colors[g])
        texts_group = [plt.text(data.x.iloc[i], 
                data.y.iloc[i], 
                industry,
                size=20,
                color=colors[g]) 
                for i,industry in enumerate(data.industry)]     # For kernel density
        texts = texts+texts_group
    # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # ax2.grid([])
    ax.set_xlabel('Alpha',fontsize = 30)
    ax.set_ylabel('d(alpha)',fontsize = 30)
    # ax.set_xlim([1e-2,3e1])
    plt.legend(fontsize = 25)
    plt.xscale('log')
    plt.title('Term 2', fontsize = 30)
    
    adjust_text(texts, precision=0.001,
            expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
            force_text=(0.01, 0.25), force_points=(0.01, 0.25),
            arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                            ))
    if save or save_all:
        plt.savefig(save_path+'term_2_alpha.'+save_format,format=save_format)
    
    plt.show()
    
    #%% plot composition term function of alpha cons
    
    # trade_baseline = trade[10].values.reshape((N,S,N))
    trade_baseline = b.cons.value.values.reshape((N,S,N))
    trade_cf = sol.cons.value.values.reshape((N,S,N)) 
    
    fig, ax = plt.subplots(figsize=(25,15))
    # ax2 = ax.twinx()
    colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    texts = []
    data_base = sector_map.copy()
    data_base['x'] = alpha_s(trade_baseline)
    data_base['y'] = alpha_s(trade_cf)-alpha_s(trade_baseline)
    for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
        data = data_base.loc[data_base['group_label'] == group_label]
        ax.scatter(data.x,data.y, label = group_label,color=colors[g])
        texts_group = [plt.text(data.x.iloc[i], 
                data.y.iloc[i], 
                industry,
                size=20,
                color=colors[g]) 
                for i,industry in enumerate(data.industry)]     # For kernel density
        texts = texts+texts_group
    # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # ax2.grid([])
    ax.set_xlabel('Alpha',fontsize = 30)
    ax.set_ylabel('d(alpha)',fontsize = 30)
    # ax.set_xlim([1e-2,3e1])
    plt.legend(fontsize = 25)
    plt.xscale('log')
    plt.title('Term 2 final consumption only', fontsize = 30)
    
    adjust_text(texts, precision=0.001,
            expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
            force_text=(0.01, 0.25), force_points=(0.01, 0.25),
            arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                            ))
    if save or save_all:
        plt.savefig(save_path+'term_2_alpha_cons.'+save_format,format=save_format)
    
    plt.show()
    
    #%% plot composition term function of alpha intermediate input
    
    # trade_baseline = trade[10].values.reshape((N,S,N))
    trade_baseline = b.iot.value.values.reshape((N,S,N,S)).sum(axis=-1)
    trade_cf = sol.iot.value.values.reshape((N,S,N,S)).sum(axis=-1)
    
    fig, ax = plt.subplots(figsize=(25,15))
    # ax2 = ax.twinx()
    colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    texts = []
    data_base = sector_map.copy()
    data_base['x'] = alpha_s(trade_baseline)
    data_base['y'] = alpha_s(trade_cf)-alpha_s(trade_baseline)
    for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
        data = data_base.loc[data_base['group_label'] == group_label]
        ax.scatter(data.x,data.y, label = group_label,color=colors[g])
        texts_group = [plt.text(data.x.iloc[i], 
                data.y.iloc[i], 
                industry,
                size=20,
                color=colors[g]) 
                for i,industry in enumerate(data.industry)]     # For kernel density
        texts = texts+texts_group
    # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # ax2.grid([])
    ax.set_xlabel('Alpha',fontsize = 30)
    ax.set_ylabel('d(alpha)',fontsize = 30)
    # ax.set_xlim([1e-2,3e1])
    plt.legend(fontsize = 25)
    plt.xscale('log')
    plt.title('Term 2 intermediate input only', fontsize = 30)
    
    adjust_text(texts, precision=0.001,
            expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
            force_text=(0.01, 0.25), force_points=(0.01, 0.25),
            arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                            ))
    if save or save_all:
        plt.savefig(save_path+'term_2_alpha_iot.'+save_format,format=save_format)
    
    plt.show()
    
    #%% plot composition term
    
    trade_baseline = trade['baseline'].values.reshape((N,S,N))
    trade_cf = trade[10].values.reshape((N,S,N)) 
    
    fig, ax = plt.subplots(figsize=(25,15))
    ax2 = ax.twinx()
    colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    texts = []
    data_base = sector_map.copy()
    data_base['y'] = np.einsum('s,s,s->s',
                       epsilon_s(trade_baseline,e),
                       alpha_s(trade_cf)-alpha_s(trade_baseline),
                       1/alpha_s(trade_baseline))
    data_base['y2'] = alpha_s(trade_cf)-alpha_s(trade_baseline)
    ax2.axhline(y=0, color='k', linestyle='-',lw=0.5,zorder=1000)
    ax.bar(0,0,color='grey',label = r'$\frac{dE_s}{E}$')
    ax.scatter([],[],color='grey',label = r'$d\alpha_s$',marker='^',s=1000)
    for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
        data = data_base.loc[data_base['group_label'] == group_label]
        bars = ax.bar(data.industry,data.y, label = group_label,color=colors[g])
        ax2.scatter(data.industry,data.y2,color=colors[g],
                    edgecolors='k',marker='^',s=1000)
        # ax.bar_label(bars,
        #              labels=data.industry,
        #              rotation=90,
        #               label_type = 'edge',
        #               padding=5,
        #               # color=colors[g],
        #               zorder=99)
        
        texts_group = [ax2.text(data.industry.iloc[i], 
                data.y2.iloc[i]-0.0005, 
                industry,
                size=20,
                # color=colors[g],
                rotation = 90,ha='center',va='top') 
                for i,industry in enumerate(data.industry)]     # For kernel density
        # texts = texts+texts_group
    # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    ax2.grid(None)
    
    
    # ax.set_xlabel('Sector',fontsize = 30)
    ax.set_ylabel(r'$\left(\frac{dE_s}{E}\right)_{_2}$',fontsize = 35,rotation=0,labelpad = 30,va='center')
    ax2.set_ylabel(r'$d\alpha_s$',fontsize = 35,rotation=0,labelpad = 30,va='center')
    ax.set_ylim([-0.08,0.08])
    ax2.set_ylim([-0.008,0.008])
    ax.legend(fontsize = 25)
    # plt.xscale('log')
    # plt.title('Term 2', fontsize = 30)
    ax2.set_xticklabels(['']
                        , rotation=45
                        , ha='right'
                        , rotation_mode='anchor'
                        ,fontsize=19)
    
    
    # adjust_text(texts, precision=0.001,
    #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    #         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                            # ))
    if save or save_all:
        plt.savefig(save_path+'term_2.'+save_format,format=save_format)
    
    plt.show()
    
    #%% sunburst term2 shares
    import kaleido
    import plotly
    from matplotlib import cm
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    scaling = 1
    # cmap = cm.get_cmap('Spectral')
    fig3 = make_subplots(rows=1, cols=2, 
                         specs=[[{"type": "sunburst"}, {"type": "sunburst"}]],
                         horizontal_spacing=0.01)
    
    trade_baseline = trade['baseline'].values.reshape((N,S,N))
    trade_cf = trade[10].values.reshape((N,S,N)) 
    
    data_base = pd.DataFrame(columns=['term','spec1','spec2','value']).set_index(['term','spec1','spec2'])
    
    # data_base.loc[('term 1','',''),'value'] = (X(trade_cf) - X(trade_baseline))/X(trade_baseline)
    
    term2 = np.einsum('s->s',
                       alpha_s(trade_cf)-alpha_s(trade_baseline))
    
    for i,sector in enumerate(b.sector_list):
        data_base.loc[('Expenditure decrease',sector_map.loc[sector,'industry'],''),'value'] = term2[i]*100
        
    # term3 = np.einsum('is->is',
    #            alpha_is(trade_cf)-alpha_is(trade_baseline))
    
    # for i,country in enumerate(b.country_list):
    #     for j,sector in enumerate(b.sector_list):
    #         data_base.loc[('term 3 reduction',sector_map.loc[sector,'industry'],country),'value'] = term3[i,j]
        
    # term4 = np.einsum('isj,->is',
    #            alpha_isj(trade_cf)-alpha_isj(trade_baseline))
    
    # for i,exp in tqdm(enumerate(b.country_list)):
    #     for j,sector in enumerate(b.sector_list):
    #         data_base.loc[('term 4 reduction',sector_map.loc[sector,'industry'],exp),'value'] = term4[i,j]
    # index = pd.MultiIndex.from_product([['term 4'],b.country_list,sector_map.industry.to_list(),b.country_list])
    # for j,i in tqdm(enumerate(index)):
    #     data_base.loc[i,'value'] = term4.ravel()[j]
    # data_base.value = np.abs(data_base.value)
    # data_base1 = data_base.loc[data_base.value<0].copy()
    # data_base2 = data_base.loc[data_base.value>0].copy()
    
    data_base.reset_index(inplace = True)
    data_base.loc[data_base.value>0, 'term'] = data_base.loc[data_base.value>0, 'term'].str.replace('decrease','increase')
    data_base.set_index(['term','spec1','spec2'],inplace=True)
    data_base.sort_index(inplace=True)
    data_base.value = np.abs(data_base.value)
    data_base.reset_index(inplace = True)
    data_base = data_base.replace('',None)
    
    
    import plotly.io as pio
    import plotly.express as px
    
    pio.renderers.default='browser'
    color_discrete_map = {
        'Expenditure decrease':sns.color_palette().as_hex()[3],
        'Expenditure increase':sns.color_palette().as_hex()[2],
        }
    
    fig1 = px.sunburst(data_base, path=['term', 'spec1','spec2'], values='value', color='term',
                        color_discrete_map=color_discrete_map)
    # fig1.update_traces(sort=False, selector=dict(type='sunburst')) 
    fig1.update_traces(textinfo="label+value",
                       texttemplate="%{label}<br>%{value:.2f}%")
    fig1.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig3.add_trace(fig1.data[0],
        row=1, col=1)
    # fig1.show()
    
    # sunburst term2 shares
    # cmap = cm.get_cmap('Spectral')
    
    
    trade_baseline = trade['baseline'].values.reshape((N,S,N))
    trade_cf = trade[10].values.reshape((N,S,N)) 
    
    data_base = pd.DataFrame(columns=['term','spec1','spec2','value']).set_index(['term','spec1','spec2'])
    
    # data_base.loc[('term 1','',''),'value'] = (X(trade_cf) - X(trade_baseline))/X(trade_baseline)
    
    term2 = np.einsum('s,s,s->s',
                       epsilon_s(trade_baseline,e),
                       alpha_s(trade_cf)-alpha_s(trade_baseline),
                       1/alpha_s(trade_baseline))
    
    for i,sector in enumerate(b.sector_list):
        data_base.loc[('Emissions decrease',sector_map.loc[sector,'industry'],''),'value'] = term2[i]*100
    
    data_base.reset_index(inplace = True)
    data_base.loc[data_base.value>0, 'term'] = data_base.loc[data_base.value>0, 'term'].str.replace('decrease','increase')
    data_base.set_index(['term','spec1','spec2'],inplace=True)
    data_base.sort_index(inplace=True)
    data_base.value = np.abs(data_base.value)
    data_base.reset_index(inplace = True)
    data_base = data_base.replace('',None)
    
    
    
    
    pio.renderers.default='browser'
    color_discrete_map = {
        'Emissions decrease':sns.color_palette().as_hex()[3],
        'Emissions increase':sns.color_palette().as_hex()[2],
        }
    
    fig2 = px.sunburst(data_base, path=['term', 'spec1','spec2'], values='value', color='term',
                        color_discrete_map=color_discrete_map)
    # fig2.update_traces(sort=False, selector=dict(type='sunburst')) 
    fig2.update_traces(rotation = 93) 
    fig2.update_traces(textinfo="label+value",
                       texttemplate="%{label}<br>%{value:.2f}%")
    fig2.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )
    # fig2.show()
    
    
    # fig3.add_trace(px.sunburst(data_base, path=['term', 'spec1','spec2'], values='value', color='term',
    #                     color_discrete_map=color_discrete_map).data[0],
    #     row=1, col=2)
    fig3.add_trace(fig2.data[0],
        row=1, col=2)
    
    fig3.update_layout(
        font=dict(
            # family="Courier New, monospace",
            size=60,  # Set the font size here
            # color="RebeccaPurple"
        )
    )
    
    fig3.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )
    
    pio.write_image(fig3,save_path+'term_2_sunburst.pdf', format='pdf', engine='orca',width=4000*scaling,height=4000*scaling*21/29.7)
    # fig3.write_html(save_path+'term_2_sunburst.html')
    fig3.show()
    
    #%% sunburst term3 shares
    from plotly.subplots import make_subplots
    import plotly.io as pio
    import plotly.express as px
    
    pio.templates[pio.templates.default].layout.colorway = [sns.color_palette("deep").as_hex()[i-2] for i in range(3)]
    
    scaling = 1
    # cmap = cm.get_cmap('Spectral')
    fig3 = make_subplots(rows=1, cols=2, 
                         specs=[[{"type": "sunburst"}, {"type": "sunburst"}]],
                         horizontal_spacing=0.01)
    
    trade_baseline = trade['baseline'].values.reshape((N,S,N))
    trade_cf = trade[10].values.reshape((N,S,N)) 
    
    data_base = pd.DataFrame(columns=['term','spec1','spec2','value']).set_index(['term','spec1','spec2'])
    
    term3 = np.einsum('is->is',
                       alpha_is(trade_cf)-alpha_is(trade_baseline))
    
    for i,sector in enumerate(b.sector_list):
        for j, country in enumerate(b.country_list):
            data_base.loc[('Expenditure share decrease',sector_map.loc[sector,'industry'],country),'value'] = term3[j,i]*100
        
    
    total = data_base.loc[data_base.value>0, 'value'].sum()
    
    data_base.reset_index(inplace = True)
    data_base['d(alpha)'] = data_base['value']
    data_base.loc[data_base.value>0, 'term'] = data_base.loc[data_base.value>0, 'term'].str.replace('decrease','increase')
    data_base.set_index(['term','spec1','spec2'],inplace=True)
    data_base.sort_index(inplace=True)
    data_base.value = np.abs(data_base.value)
    data_base.reset_index(inplace = True)
    data_base = data_base.replace('',None)
    data_base['value'] = data_base['value']*total/data_base['value'].sum()
    
    import plotly.io as pio
    import plotly.express as px
    
    pio.renderers.default='browser'
    color_discrete_map = {
        'Expenditure share decrease':sns.color_palette().as_hex()[3],
        'Expenditure share increase':sns.color_palette().as_hex()[2],
        }
    
    hover_data = {'term':False
                  , 'spec1':False
                  , 'spec2':False
                  , 'value':True
                  , 'd(alpha)':True}
    
    fig1 = px.sunburst(data_base, path=['spec1', 'spec2'], values='value', color='term',
                        color_discrete_map=color_discrete_map,
                        hover_data = hover_data,
                        # labels=[{'0': 'Deceased', '1': 'Survived'}]
                        )
    fig1.update_traces(textinfo="label+value",
                       texttemplate="%{label}<br>%{value:.2f}%")
    fig1.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )
    
    fig3.add_trace(fig1.data[0],
        row=1, col=1)
    
    
    trade_baseline = trade['baseline'].values.reshape((N,S,N))
    trade_cf = trade[10].values.reshape((N,S,N)) 
    
    data_base2 = pd.DataFrame(columns=['term','spec1','spec2','value']).set_index(['term','spec1','spec2'])
    
    term3 = np.einsum('is,is,is->is',
                       epsilon_is(trade_baseline,e),
                       alpha_is(trade_cf)-alpha_is(trade_baseline),
                       1/alpha_is(trade_baseline))
    
    for i,sector in enumerate(b.sector_list):
        for j, country in enumerate(b.country_list):
            data_base2.loc[('Emissions decrease',sector_map.loc[sector,'industry'],country),'value'] = term3[j,i]*100
    
    data_base2.sort_index(inplace=True)
    em_saved = data_base2.loc[data_base2.value>0].groupby(level=[0,1])['value'].sum() + data_base2.loc[data_base2.value<0].groupby(level=[0,1])['value'].sum()
    data_base2['value'] = data_base2['value']*np.abs(em_saved/data_base2.groupby(level=[0,1])['value'].sum())
    data_base2.reset_index(inplace = True)
    data_base2['dE/E'] = data_base2['value']
    data_base2.loc[data_base2.value>0, 'term'] = data_base2.loc[data_base2.value>0, 'term'].str.replace('decrease','increase')
    data_base2.value = np.abs(data_base2.value)
    data_base2 = data_base2.replace('',None)
    # data_base2['total'] = 'total'
    
    
    import plotly.io as pio
    import plotly.express as px
    
    pio.renderers.default='browser'
    color_discrete_map = {
        'Emissions decrease':sns.color_palette().as_hex()[3],
        'Emissions increase':sns.color_palette().as_hex()[2],
        }
    
    hover_data = {'term':False
                  , 'spec1':False
                  , 'spec2':False
                  , 'value':True
                  # , 'value disp':True
                  , 'dE/E':True}
    
    
    fig2 = px.sunburst(data_base2, path=['spec1','spec2'], values='value', color='term',
                       # template = 'plotly_dark_custom',
                       hover_data = hover_data,
                        color_discrete_map=color_discrete_map)
    fig2.update_traces(textinfo="label+value",
                       texttemplate="%{label}<br>%{value:.2f}%")
    fig2.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )
    # fig2.update_traces(sort=False, selector=dict(type='sunburst')) 
    # fig2.update_traces(rotation = 90) 
    # fig2.show()
    # color_mapping = {'0': "#708090", '1': "#006400", 'female': "#c71585", 'male': "#0000cd"}
    
    # fig2.update_traces(marker_colors=[color_mapping[cat] for cat in fig.data[-1].labels])
    
        
    
    # fig3.add_trace(px.sunburst(data_base, path=['term', 'spec1','spec2'], values='value', color='term',
    #                     color_discrete_map=color_discrete_map).data[0],
    #     row=1, col=2)
    fig3.add_trace(fig2.data[0],
        row=1, col=2)
    
    fig3.update_layout(
        font=dict(
            # family="Courier New, monospace",
            size=30,  # Set the font size here
            # color="RebeccaPurple"
        )
    )
    
    # colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    # color_mapping = {'0': "#708090", '1': "#006400", 'female': "#c71585", 'male': "#0000cd"}
    
    # fig3.update_traces(
    #     marker_colors=[colors[sector_map.loc[sector_map.industry == cat,'group_code'].iloc[0]]
    #                    for cat in fig3.data[-1].labels]
    #     )
    scaling = 1/2
    fig3.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )
    
    # pio.write_image(fig3,save_path+'term_3_sunburst.pdf', format='pdf', engine='orca',width=4000*scaling,height=4000*scaling*21/29.7)
    fig3.write_html(save_path+'term_3_sunburst.html')
    fig3.show()
    
    #%% plot term 3 epsilon(alpha) i specific
    
    trade_baseline = trade['baseline'].values.reshape((N,S,N))
    trade_cf = trade[10].values.reshape((N,S,N)) 
    
    fig, ax = plt.subplots(figsize=(25,15))
    # colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    # texts = []
    countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    #                                                               b.sector_list],
    #                                                             names = ['country','sector'])).reset_index()
    # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    data_base = countries.copy()
    
    data_base['y'] = np.einsum('is->i',epsilon_is(trade_baseline,e)).ravel()
    data_base['x'] = np.einsum('is->i',alpha_is(trade_baseline)).ravel()/S
    # for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    # data = data_base.loc[data_base['group_label'] == group_label]
    data= data_base
    ax.scatter(data.x,data.y
               # , label = group_label,color=colors[g]
               )
    texts = [plt.text(data.x.loc[i], 
            data.y.loc[i], 
            data.loc[i].country_name,
            size=15,
            # color=colors[g]
            ) 
            for i in data_base.index]     # For kernel density
    # texts = texts+texts_group
    # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # ax2.grid([])
    ax.set_xlabel(r'$\sum_s \alpha_{is}$',fontsize = 30)
    ax.set_ylabel(r'$\sum_s \epsilon_{is}$',fontsize = 30,rotation=0,labelpad=30)
    ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
    # ax.set_xlim([1e-2,3e1])
    # plt.legend(fontsize = 25)
    plt.xscale('log')
    plt.yscale('log')
    # plt.title('Term 3')
    ax.tick_params(axis='both', which='major', labelsize=25)
    adjust_text(texts, precision=0.001,
            expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
            force_text=(0.01, 0.25), force_points=(0.01, 0.25),
            arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                            ))
    if save or save_all:
        plt.savefig(save_path+'term_3_eps_alpha_i.'+save_format,format=save_format)
    
    plt.show()
    
    #%% plot term 3 function of epsilon / alpha
    
    # trade_baseline = trade[10].values.reshape((N,S,N))
    trade_baseline = trade['baseline'].values.reshape((N,S,N))
    trade_cf = trade[10].values.reshape((N,S,N)) 
    
    fig, ax = plt.subplots(figsize=(25,15))
    # ax2 = ax.twinx()
    colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    texts = []
    countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    #                                                               b.sector_list],
    #                                                             names = ['country','sector'])).reset_index()
    # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    data_base = countries.copy()
    
    data_base['y'] = np.einsum('is->i',alpha_is(trade_cf)-alpha_is(trade_baseline)).ravel()
    data_base['x'] = np.einsum('is,is->i',
                               epsilon_is(trade_baseline,e),
                               1/alpha_is(trade_baseline)).ravel()/S
    data = data_base
    ax.scatter(data.x,data.y)
    texts = [plt.text(data.x.iloc[i], 
            data.y.iloc[i], 
            country,
            size=20,
            # color=colors[g]
            ) 
            for i,country in enumerate(data.index)]     # For kernel density
        # texts = texts+texts_group
    # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # ax2.grid([])
    ax.set_xlabel(r'$\sum_s \frac{\epsilon_{is}}{\alpha_{is}}$',fontsize = 40)
    ax.set_ylabel(r'$\sum_s d\alpha_{is}$',fontsize = 40, labelpad=20,rotation=0)
    # ax.set_xlim([1e-2,3e1])
    # plt.legend(fontsize = 25)
    plt.xscale('log')
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(axis='both', which='minor', labelsize=25)
    
    plt.axhline(y=0, color='k', linestyle='-',lw=0.5)
    
    adjust_text(texts, precision=0.001,
            expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
            force_text=(0.01, 0.25), force_points=(0.01, 0.25),
            arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                            ))
    if save or save_all:
        plt.savefig(save_path+'term_3_eps_over_alpha_i.'+save_format,format=save_format)
    
    plt.show()
    
    #%% plot term 3 dalpha/ alpha function of epsilon
    
    # trade_baseline = trade[10].values.reshape((N,S,N))
    trade_baseline = trade['baseline'].values.reshape((N,S,N))
    trade_cf = trade[10].values.reshape((N,S,N)) 
    
    fig, ax = plt.subplots(figsize=(25,15))
    # ax2 = ax.twinx()
    colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    texts = []
    countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    #                                                               b.sector_list],
    #                                                             names = ['country','sector'])).reset_index()
    # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    data_base = countries.copy()
    
    data_base['y'] = np.einsum('is,is->i',alpha_is(trade_cf)-alpha_is(trade_baseline),
    1/alpha_is(trade_baseline)).ravel()
    data_base['x'] = np.einsum('is->i',
                               epsilon_is(trade_baseline,e)).ravel()/S
    data = data_base
    ax.scatter(data.x,data.y)
    texts = [plt.text(data.x.iloc[i], 
            data.y.iloc[i], 
            country,
            size=20,
            # color=colors[g]
            ) 
            for i,country in enumerate(data.country_name)]     # For kernel density
        # texts = texts+texts_group
    # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # ax2.grid([])
    ax.set_ylabel(r'$\left(\sum_s \frac{d\alpha_{is}}{\alpha_{is}}\right)_{_3}$',
                  fontsize = 40,rotation=0,labelpad =80,va='bottom')
    ax.set_xlabel(r'$\sum_s \epsilon_{is}$',fontsize = 40, labelpad=20,rotation=0)
    # ax.set_xlim([1e-2,3e1])
    # plt.legend(fontsize = 25)
    plt.xscale('log')
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(axis='both', which='minor', labelsize=25)
    
    plt.axhline(y=0, color='k', linestyle='-',lw=0.5)
    
    adjust_text(texts, precision=0.001,
            expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
            force_text=(0.01, 0.25), force_points=(0.01, 0.25),
            arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                            ))
    if save or save_all:
        plt.savefig(save_path+'term_3_dalpha_over_alpha_func_epsilon_i.'+save_format,format=save_format)
    
    plt.show()
    
    #%% plot term 3
    
    trade_baseline = trade['baseline'].values.reshape((N,S,N))
    trade_cf = trade[10].values.reshape((N,S,N)) 
    
    fig, ax = plt.subplots(figsize=(25,15))
    ax2 = ax.twinx()
    colors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    texts = []
    countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    #                                                               b.sector_list],
    #                                                             names = ['country','sector'])).reset_index()
    # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    data_base = countries.copy()
    data_base['y'] = np.einsum('is,is,is->i',
                       epsilon_is(trade_baseline,e),
                       alpha_is(trade_cf)-alpha_is(trade_baseline),
                       1/alpha_is(trade_baseline))
    data_base['y2'] = (alpha_is(trade_cf)-alpha_is(trade_baseline)).sum(axis=1)
    ax2.axhline(y=0, color='k', linestyle='-',lw=0.5,zorder=1000)
    # ax.bar(0,0,color='grey',label = r'$\sum_s dE_{is}$')
    # ax.scatter([],[],color='grey',label = r'$\sum_s d\alpha_{is}$',marker='^',s=1000)
    data = data_base
    bars = ax.bar(data.country_name,data.y)
    ax2.scatter(data.country_name,data.y2,color=sns.color_palette()[1],
                edgecolors='k',marker='^',s=1000)
    # ax.bar_label(bars,
    #              labels=data.industry,
    #              rotation=90,
    #               label_type = 'edge',
    #               padding=5,
    #               # color=colors[g],
    #               zorder=99)
    
    texts_group = [ax2.text(data.country_name.iloc[i], 
            data.y2.iloc[i]-0.02, 
            country,
            size=15,
            # color=colors[g],
            rotation = 90,ha='center',va='top') 
            for i,country in enumerate(data.country_name)]     # For kernel density
        # texts = texts+texts_group
    # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    ax2.grid(None)
    
    
    # ax.set_xlabel('Sector',fontsize = 30)
    ax.set_ylabel(r'$\left(\sum_s \frac{dE_{is}}{E_{is}} \right)_{_3}$',fontsize = 35,rotation=0,labelpad = 80,va='center')
    ax2.set_ylabel(r'$\left(\sum_s d\alpha_{is}\right)_3$',fontsize = 35,rotation=0,labelpad = 80,va='center')
    ax.set_ylim([-0.025,0.025])
    ax2.set_ylim([-0.4,0.4])
    # ax.legend(fontsize = 25)
    # plt.xscale('log')
    # plt.title('Term 2', fontsize = 30)
    ax2.set_xticklabels(['']
                        , rotation=45
                        , ha='right'
                        , rotation_mode='anchor'
                        ,fontsize=19)
    
    
    # adjust_text(texts, precision=0.001,
    #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    #         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                            # ))
    if save or save_all:
        plt.savefig(save_path+'term_3_i.'+save_format,format=save_format)
    
    plt.show()
    
    #%% Scatter plot of gross ouput change with kernel density by coarse industry
    
    print('Plotting scatter plot of output changes for every country x sector according to produciton intensity with kernel density estimates for categories of sectors')
    
    save = False
    
    sol = sols[10]
    
    country_list = b.country_list
    C = len(country_list)
    sector_list = b.sector_list
    
    p_hat_sol = sol.res.price_hat.to_numpy().reshape(C,S)
    E_hat_sol = sol.res.output_hat.to_numpy().reshape(C,S)
    E_hat_sol_percent = (E_hat_sol-1)*100
    q_hat_sol = E_hat_sol / p_hat_sol
    q_hat_sol_percent = (q_hat_sol-1)*100
    
    sector_map = pd.read_csv(data_path+'industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')
    
    data = pd.DataFrame(data = E_hat_sol_percent.ravel(),
                        index = pd.MultiIndex.from_product([country_list,sector_map.index.get_level_values(level=0).to_list()],
                                                           names=['country','sector']),
                        columns=['value'])
    data = data.reset_index().merge(sector_map.reset_index(),how='left',left_on='sector',right_on='ind_code').set_index(['country','sector']).drop('ind_code',axis=1)
    data['co2_intensity'] = b.co2_intensity.value.values
    data['output'] = b.output.value.values
    data=data.sort_values('group_code')
    data = data.loc[data.group_label.isin(['Agro-food','Raw materials','Energy'])]
    
    sector_list_full = []
    for sector in sector_list:
        sector_list_full.append(sector_map.loc['D'+sector].industry)
    
    group_labels_sorted = data.group_label.drop_duplicates().to_list()
    
    data_no_z =data.copy()
    data_no_z = data_no_z[data_no_z['co2_intensity'] != 0]
    # data_no_z = data_no_z[data_no_z['co2_intensity'] < 1e4]
    # data_no_z['co2_intensity'] = np.log(data_no_z['co2_intensity'])
    data_no_z = data_no_z[['co2_intensity','value','group_label','group_code','output']]
    
    data_no_z_1 = data_no_z[data_no_z['co2_intensity'] < 100].copy()
    data_no_z_2 = data_no_z[data_no_z['co2_intensity'] >= 100].copy()
    
    fig, ax = plt.subplots(figsize=(12,8),constrained_layout=True)
    # # sns.move_legend(plot, "lower left", bbox_to_anchor=(.55, .45), title='Species')
    
    # plot.fig.get_axes()[0].legend(loc='lower left')
    # # plt.legend(loc='lower left')
    
    palette = [sns.color_palette('bright')[i] for i in [2,4,0,3,1,7]]
    palette[0] = sns.color_palette()[2]
    palette[1] = sns.color_palette("hls", 8)[-2]
    for data_no_z_i in [data_no_z_1,data_no_z_2] :
    # for data_no_z_i in [data_no_z] :
        plot2 = sns.kdeplot(data=data_no_z_i,
                    x='co2_intensity',
                    y="value",
                    hue = 'group_label',
                    fill = True,
                    alpha = 0.25,
                    log_scale=(True, False),
                    # height=10,
                    # ratio=5,
                    # bw_adjust=0.5,
                    weights = 'output',
                    legend=False,
                    levels = 2,
                    palette = palette,
                    common_norm = True,
                    # shade=True,
                    thresh = 0.2,
                    # fill = False,
                    # alpha=0.6,
                    # hue_order = data.group_label.drop_duplicates().to_list()[::-1],
                    ax = ax
                    )
    for data_no_z_i in [data_no_z_1,data_no_z_2] :
        for i,group in enumerate(data_no_z_i.group_code.drop_duplicates().to_list()):
            ax.scatter(data_no_z_i[data_no_z_i['group_code'] == group].co2_intensity,
                       data_no_z_i[data_no_z_i['group_code'] == group].value,
                       color=palette[i],
                       s=(data_no_z_i[data_no_z_i['group_code'] == group].output)/1e3,zorder=1-i)
    
    ax.set_ylabel('Gross output changes (%)',
                    fontsize=20
                    )
    ax.set_xscale('log')
    ax.set_ylim(-100,+50)
    # ax.set_ylim(-80, +80)
    
    # ax.set_xlim(0.5,20000)
    ax.set_xlim(data_no_z.co2_intensity.min(),3e4)
    ax.margins(x=0)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    
    ax.set_xlabel('Carbon intensity of production (Tons / Mio.$)',
                    fontsize=20)
    
    handles = [mpatches.Patch(color=palette[ind], label=group_labels_sorted[ind]) for ind,group in enumerate(group_labels_sorted)]
    ax.legend(handles=handles,fontsize=20, loc = 'lower left')
    
    ax.xaxis.set_major_formatter(ScalarFormatter())
    
    ax.hlines(0,xmin=b.co2_intensity.value.min(),xmax=1e5,colors='black',ls='--',lw=1)
    
    # sec = '20'
    # sector = sector_map.loc['D' + sec].industry
    # sector_index = sector_list.index(sec)
    
    # country = 'RUS'
    # country_index = country_list.index(country)
    
    # ax.annotate(country + ' - ' + sector,
    #             xy=(b.co2_intensity.loc[country, sec].value, q_hat_sol_percent[country_index, sector_index]),
    #             xycoords='data',
    #             xytext=(-250, 0),
    #             textcoords='offset points',
    #             va='center',
    #             arrowprops=dict(arrowstyle="->",
    #                             connectionstyle="arc3", color='black'),
    #             bbox=dict(boxstyle="round", fc="w"), zorder=10
    #             )
    
    # sec = '28'
    # sector = sector_map.loc['D' + sec].industry
    # sector_index = sector_list.index(sec)
    
    # country = 'CHN'
    # country_index = country_list.index(country)
    
    # ax.annotate(country + ' - ' + sector,
    #             xy=(b.co2_intensity.loc[country, sec].value, q_hat_sol_percent[country_index, sector_index]),
    #             xycoords='data',
    #             xytext=(-50, -80),
    #             textcoords='offset points',
    #             va='center',
    #             arrowprops=dict(arrowstyle="->",
    #                             connectionstyle="arc3", color='black'),
    #             bbox=dict(boxstyle="round", fc="w"), zorder=10
    #             )
    
    # sec = '35'
    # sector = sector_map.loc['D' + sec].industry
    # sector_index = sector_list.index(sec)
    
    # country = 'NOR'
    # country_index = country_list.index(country)
    
    # ax.annotate(country + ' - ' + sector,
    #             xy=(b.co2_intensity.loc[country, sec].value, q_hat_sol_percent[country_index, sector_index]),
    #             xycoords='data',
    #             xytext=(20, 80),
    #             textcoords='offset points',
    #             va='center',
    #             arrowprops=dict(arrowstyle="->",
    #                             connectionstyle="arc3", color='black'),
    #             bbox=dict(boxstyle="round", fc="w"), zorder=10
    #             )
    
    # sec = '50'
    # sector = sector_map.loc['D' + sec].industry
    # sector_index = sector_list.index(sec)
    
    # country = 'DEU'
    # country_index = country_list.index(country)
    
    # ax.annotate(country + ' - ' + sector,
    #             xy=(b.co2_intensity.loc[country, sec].value, q_hat_sol_percent[country_index, sector_index]),
    #             xycoords='data',
    #             xytext=(80, 15),
    #             textcoords='offset points',
    #             va='center',
    #             arrowprops=dict(arrowstyle="->",
    #                             connectionstyle="arc3", color='black'),
    #             bbox=dict(boxstyle="round", fc="w"), zorder=10
    #             )
    
    # sec = '01T02'
    # sector = sector_map.loc['D' + sec].industry
    # sector_index = sector_list.index(sec)
    
    # country = 'BRA'
    # country_index = country_list.index(country)
    
    # ax.annotate(country + ' - ' + sector,
    #             xy=(b.co2_intensity.loc[country, sec].value, q_hat_sol_percent[country_index, sector_index]),
    #             xycoords='data',
    #             xytext=(-250, -5),
    #             textcoords='offset points',
    #             va='center',
    #             arrowprops=dict(arrowstyle="->",
    #                             connectionstyle="arc3", color='black'),
    #             bbox=dict(boxstyle="round", fc="w"), zorder=10
    #             )
    
    # sec = '01T02'
    # sector = sector_map.loc['D' + sec].industry
    # sector_index = sector_list.index(sec)
    
    # country = 'CHE'
    # country_index = country_list.index(country)
    
    # ax.annotate(country + ' - ' + sector,
    #             xy=(b.co2_intensity.loc[country, sec].value, q_hat_sol_percent[country_index, sector_index]),
    #             xycoords='data',
    #             xytext=(100, -35),
    #             textcoords='offset points',
    #             va='center',
    #             arrowprops=dict(arrowstyle="->",
    #                             connectionstyle="arc3", color='black'),
    #             bbox=dict(boxstyle="round", fc="w"), zorder=10
    #             )
    
    # plt.tight_layout()
    
    sec = '24'
    sector = sector_map.loc['D' + sec].industry
    sector_index = sector_list.index(sec)
    
    country = 'CHL'
    country_index = country_list.index(country)
    
    ax.annotate(country + ' - ' + sector,
                xy=(b.co2_intensity.loc[country, sec].value, E_hat_sol_percent[country_index, sector_index]),
                xycoords='data',
                xytext=(-250, 0),
                textcoords='offset points',
                va='center',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3", color='black'),
                bbox=dict(boxstyle="round", fc="w"), zorder=10
                )
    
    sec = '24'
    sector = sector_map.loc['D' + sec].industry
    sector_index = sector_list.index(sec)
    
    country = 'IND'
    country_index = country_list.index(country)
    
    ax.annotate(country + ' - ' + sector,
                xy=(b.co2_intensity.loc[country, sec].value, E_hat_sol_percent[country_index, sector_index]),
                xycoords='data',
                xytext=(-250, 0),
                textcoords='offset points',
                va='center',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3", color='black'),
                bbox=dict(boxstyle="round", fc="w"), zorder=10
                )
    
    sec = '24'
    sector = sector_map.loc['D' + sec].industry
    sector_index = sector_list.index(sec)
    
    country = 'PER'
    country_index = country_list.index(country)
    
    ax.annotate(country + ' - ' + sector,
                xy=(b.co2_intensity.loc[country, sec].value, E_hat_sol_percent[country_index, sector_index]),
                xycoords='data',
                xytext=(-250, 20),
                textcoords='offset points',
                va='center',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3", color='black'),
                bbox=dict(boxstyle="round", fc="w"), zorder=10
                )
    sec = '19'
    sector = sector_map.loc['D' + sec].industry
    sector_index = sector_list.index(sec)
    
    country = 'IRL'
    country_index = country_list.index(country)
    
    ax.annotate(country + ' - ' + sector,
                xy=(b.co2_intensity.loc[country, sec].value, E_hat_sol_percent[country_index, sector_index]),
                xycoords='data',
                xytext=(-250, 0),
                textcoords='offset points',
                va='center',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3", color='black'),
                bbox=dict(boxstyle="round", fc="w"), zorder=10
                )
    
    sec = '35'
    sector = sector_map.loc['D' + sec].industry
    sector_index = sector_list.index(sec)
    
    country = 'RUS'
    country_index = country_list.index(country)
    
    ax.annotate(country + ' - ' + sector,
                xy=(b.co2_intensity.loc[country, sec].value, E_hat_sol_percent[country_index, sector_index]),
                xycoords='data',
                xytext=(-200, -100),
                textcoords='offset points',
                va='center',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3", color='black'),
                bbox=dict(boxstyle="round", fc="w"), zorder=10
                )
    
    sec = '01T02'
    sector = sector_map.loc['D' + sec].industry
    sector_index = sector_list.index(sec)
    
    country = 'BRA'
    country_index = country_list.index(country)
    
    ax.annotate(country + ' - ' + sector,
                xy=(b.co2_intensity.loc[country, sec].value, E_hat_sol_percent[country_index, sector_index]),
                xycoords='data',
                xytext=(-200, -100),
                textcoords='offset points',
                va='center',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3", color='black'),
                bbox=dict(boxstyle="round", fc="w"), zorder=10
                )
    
    sec = '05T06'
    sector = sector_map.loc['D' + sec].industry
    sector_index = sector_list.index(sec)
    
    country = 'SAU'
    country_index = country_list.index(country)
    
    ax.annotate(country + ' - ' + sector,
                xy=(b.co2_intensity.loc[country, sec].value, E_hat_sol_percent[country_index, sector_index]),
                xycoords='data',
                xytext=(150, 0),
                textcoords='offset points',
                va='center',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3", color='black'),
                bbox=dict(boxstyle="round", fc="w"), zorder=10
                )
    
    if save or save_all:
        plt.savefig(save_path+'micro_effect.pdf',format='pdf')
    plt.show()
    
    #%% local social cost of carbon
    
    scc_df = pd.read_csv('data/cscc-database-2018-master/cscc_db_v2.csv')
    
    fig, ax = plt.subplots(figsize = (16,12))
    
    # for i in range(5):
    scc = scc_df.groupby('ISO3')['50%'].first().reset_index()
    error = np.abs(scc_df.groupby('ISO3')['50%'].max() - scc_df.groupby('ISO3')['50%'].min())/5
    error.rename('error',inplace = True)
    scc = scc.loc[scc.ISO3.isin(b.country_list)]
    
    sol = sols[10]
    countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    
    data = countries.copy()
    data = data.join(scc.set_index('ISO3').rename_axis('country')).dropna()
    
    data = data.join(sol.utility)
    
    data = data.join(error.rename_axis('country'))
    
    data.columns = ['country_name','scc','welfare','error']
    
    
    
    # y = sol.utility.reset_index()
    # y = y.loc[y.country.isin()]
    
    
    
    # ax.scatter(data.scc,data.welfare,label = str(i))
    ax.errorbar(data.scc,data.welfare,xerr=data.error, fmt='o')
    # if i == 0:
    texts = [plt.text(data.scc.loc[i], 
            data.welfare.loc[i], 
            data.country_name.loc[i],
            size=15,
            # color=colors[g]
            ) 
            for i in data.index]     # For kernel density
        
    ax.set_xlabel('Local social cost of carbon',fontsize = 30)
    ax.set_ylabel('Real income change from climate action',fontsize = 30)
    
    plt.xscale('log')
    
    # plt.title('Term 3')
    # ax.tick_params(axis='both', which='major', labelsize=25)
    adjust_text(texts, precision=0.001,
            expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
            force_text=(0.01, 0.25), force_points=(0.01, 0.25),
            arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                            ))
    
    
    plt.show()
    
    
    # #%% OLD
    
    
    #  #%% plot term 3 epsilon(alpha) all terms
    
    # trade_baseline = trade['baseline'].values.reshape((N,S,N))
    # trade_cf = trade[10].values.reshape((N,S,N)) 
    
    # fig, ax = plt.subplots(figsize=(25,15))
    # # colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    # # texts = []
    # countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    #                                                               b.sector_list],
    #                                                             names = ['country','sector'])).reset_index()
    # # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    # # data_base = countries.copy()
    
    # data_base['y'] = np.einsum('sj->sj',epsilon_sj(trade_baseline,e)).ravel()
    # data_base['x'] = np.einsum('sj->sj',alpha_sj(trade_baseline)).ravel()
    # # for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    # # data = data_base.loc[data_base['group_label'] == group_label]
    # data= data_base
    # ax.scatter(data.x,data.y
    #            # , label = group_label,color=colors[g]
    #            )
    # # texts_group = [plt.text(data.x.iloc[i], 
    # #         data.y.iloc[i], 
    # #         data.iloc[i].country+','+data.iloc[i].sector,
    # #         size=10,
    # #         # color=colors[g]
    # #         ) 
    # #         for i in data_base.index]     # For kernel density
    # # texts = texts+texts_group
    # # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # # ax2.grid([])
    # ax.set_xlabel('Alpha',fontsize = 30)
    # ax.set_ylabel('Epsilon',fontsize = 30)
    # ax.plot([0,data_base.x.max()],[0,data_base.x.max()/S],ls='--',color='k')
    # # ax.set_xlim([1e-2,3e1])
    # # plt.legend(fontsize = 25)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title('Term 3')
    # ax.tick_params(axis='both', which='major', labelsize=25)
    
    
    # # adjust_text(texts, precision=0.001,
    # #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    # #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    # #         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
    # #                         ))
    # if save or save_all:
    #     plt.savefig(save_path+'term_3_eps_alpha.'+save_format,format=save_format)
    
    # plt.show()
    
    # #%% plot term 3 function of epsilon over alpha all terms
    
    # trade_baseline = trade['baseline'].values.reshape((N,S,N))
    # trade_cf = trade[10].values.reshape((N,S,N)) 
    
    # fig, ax = plt.subplots(figsize=(25,15))
    # # colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    # # texts = []
    # countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    #                                                               b.sector_list],
    #                                                             names = ['country','sector'])).reset_index()
    # # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    # # data_base = countries.copy()
    
    # data_base['x'] = np.einsum('sj,sj->sj',
    #                            epsilon_sj(trade_baseline,e),
    #                            1/alpha_sj(trade_baseline)).ravel()
    # data_base['y'] = np.einsum('sj->sj',alpha_sj(trade_cf)-alpha_sj(trade_baseline)).ravel()
    # # for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    # # data = data_base.loc[data_base['group_label'] == group_label]
    # data= data_base
    # ax.scatter(data.x,data.y
    #            # , label = group_label,color=colors[g]
    #            )
    # # texts_group = [plt.text(data.x.iloc[i], 
    # #         data.y.iloc[i], 
    # #         data.iloc[i].country+','+data.iloc[i].sector,
    # #         size=10,
    # #         # color=colors[g]
    # #         ) 
    # #         for i in data_base.index]     # For kernel density
    # # texts = texts+texts_group
    # # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # # ax2.grid([])
    # ax.set_xlabel('Epsilon/Alpha',fontsize = 30)
    # ax.set_ylabel('d(alpha)',fontsize = 30)
    # # ax.plot([0,data_base.x.max()],[0,data_base.x.max()/S/N],ls='--',color='k')
    # # ax.set_xlim([1e-2,3e1])
    # # plt.legend(fontsize = 25)
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.title('Term 3')
    
    
    # # adjust_text(texts, precision=0.001,
    # #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    # #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    # #         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
    # #                         ))
    # if save or save_all:
    #     plt.savefig(save_path+'term_3_eps_over_alpha.'+save_format,format=save_format)
    
    # plt.show()
    
    # #%% plot term 3 function of epsilon over alpha by sector all terms
    
    # trade_baseline = trade['baseline'].values.reshape((N,S,N))
    # trade_cf = trade[10].values.reshape((N,S,N)) 
    
    
    # # colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    # # texts = []
    # countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    #                                                               b.sector_list],
    #                                                             names = ['country','sector'])).reset_index()
    # # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    # # data_base = countries.copy()
    
    # data_base['x'] = np.einsum('sj,sj->sj',
    #                            epsilon_sj(trade_baseline,e),
    #                            1/alpha_sj(trade_baseline)).ravel()
    # data_base['y'] = np.einsum('sj->sj',alpha_sj(trade_cf)-alpha_sj(trade_baseline)).ravel()
    # data_base = pd.merge(data_base,sector_map.reset_index(),on='sector')
    # # for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    # # data = data_base.loc[data_base['group_label'] == group_label]
    # for sector in data_base.sector.drop_duplicates():   
    #     fig, ax = plt.subplots(figsize=(25,15))
    #     data= data_base.loc[data_base.sector == sector]
    #     ax.scatter(data.x,data.y
    #            # , label = group_label,color=colors[g]
    #            )
    #     texts_group = [plt.text(data.x.loc[i], 
    #             data.y.loc[i], 
    #             data.loc[i].country,
    #             size=20,
    #             # color=colors[g]
    #             ) 
    #             for i in data.index]     # For kernel density
    # # texts = texts+texts_group
    # # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # # ax2.grid([])
    #     ax.set_xlabel('Epsilon/Alpha',fontsize = 30)
    #     ax.set_ylabel('d(alpha)',fontsize = 30)
    # # ax.plot([0,data_base.x.max()],[0,data_base.x.max()/S/N],ls='--',color='k')
    # # ax.set_xlim([1e-2,3e1])
    # # plt.legend(fontsize = 25)
    #     plt.xscale('log')
    #     # plt.yscale('log')
    #     plt.title('Term 3 '+sector_map.loc[sector].industry)
    
    
    # # adjust_text(texts, precision=0.001,
    # #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    # #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    # #         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
    # #                         ))
    # # if save or save_all:
    # #     plt.savefig(save_path+'term_3_eps_over_alpha.'+save_format,format=save_format)
    
    #     plt.show()
    
    # #%% plot term 3 epsilon(alpha) j specific
    
    # trade_baseline = trade['baseline'].values.reshape((N,S,N))
    # trade_cf = trade[10].values.reshape((N,S,N)) 
    
    # fig, ax = plt.subplots(figsize=(25,15))
    # # colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    # # texts = []
    # countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    # #                                                               b.sector_list],
    # #                                                             names = ['country','sector'])).reset_index()
    # # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    # data_base = countries.copy()
    
    # data_base['y'] = np.einsum('sj->j',epsilon_sj(trade_baseline,e)).ravel()
    # data_base['x'] = np.einsum('sj->j',alpha_sj(trade_baseline)).ravel()/S
    # # for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    # # data = data_base.loc[data_base['group_label'] == group_label]
    # data= data_base
    # ax.scatter(data.x,data.y
    #            # , label = group_label,color=colors[g]
    #            )
    # texts = [plt.text(data.x.loc[i], 
    #         data.y.loc[i], 
    #         data.loc[i].country_name,
    #         size=15,
    #         # color=colors[g]
    #         ) 
    #         for i in data_base.index]     # For kernel density
    # # texts = texts+texts_group
    # # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # # ax2.grid([])
    # ax.set_xlabel('Alpha',fontsize = 30)
    # ax.set_ylabel('Epsilon',fontsize = 30)
    # ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
    # # ax.set_xlim([1e-2,3e1])
    # # plt.legend(fontsize = 25)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title('Term 3')
    
    # adjust_text(texts, precision=0.001,
    #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    #         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
    #                         ))
    # if save or save_all:
    #     plt.savefig(save_path+'term_3_eps_alpha_j.'+save_format,format=save_format)
    
    # plt.show()
    
    # #%% plot term 3 epsilon(alpha) cons only j specific
    
    # trade_baseline = b.cons.value.values.reshape((N,S,N))
    # trade_cf = sol.cons.value.values.reshape((N,S,N)) 
    
    # fig, ax = plt.subplots(figsize=(25,15))
    # # colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    # # texts = []
    # countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    # #                                                               b.sector_list],
    # #                                                             names = ['country','sector'])).reset_index()
    # # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    # data_base = countries.copy()
    
    # data_base['y'] = np.einsum('sj->j',epsilon_sj(trade_baseline,e)).ravel()
    # data_base['x'] = np.einsum('sj->j',alpha_sj(trade_baseline)).ravel()/S
    # # for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    # # data = data_base.loc[data_base['group_label'] == group_label]
    # data= data_base
    # ax.scatter(data.x,data.y
    #            # , label = group_label,color=colors[g]
    #            )
    # texts = [plt.text(data.x.loc[i], 
    #         data.y.loc[i], 
    #         data.loc[i].country_name,
    #         size=15,
    #         # color=colors[g]
    #         ) 
    #         for i in data_base.index]     # For kernel density
    # # texts = texts+texts_group
    # # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # # ax2.grid([])
    # ax.set_xlabel('Alpha',fontsize = 30)
    # ax.set_ylabel('Epsilon',fontsize = 30)
    # ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
    # # ax.set_xlim([1e-2,3e1])
    # # plt.legend(fontsize = 25)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title('Term 3 cons only')
    
    
    # adjust_text(texts, precision=0.001,
    #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    #         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
    #                         ))
    # if save or save_all:
    #     plt.savefig(save_path+'term_3_eps_alpha_j_cons_only.'+save_format,format=save_format)
    
    # plt.show()
    
    # #%% plot term 3
    # trade_baseline = trade['baseline'].values.reshape((N,S,N))
    # trade_cf = trade[10].values.reshape((N,S,N)) 
    
    # fig, ax = plt.subplots(figsize=(25,15))
    # # colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    # # texts = []
    # ax2 = ax.twinx()
    # countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    # #                                                              b.sector_list],
    # #                                                             names = ['country','sector']))
    # # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    # data_base = countries.copy()
    # data_base['y'] = np.einsum('sj,sj,sj->j',
    #            epsilon_sj(trade_baseline,e),
    #            alpha_sj(trade_cf)-alpha_sj(trade_baseline),
    #            1/alpha_sj(trade_baseline))
    # data_base['y2'] = np.einsum('sj->j',alpha_sj(trade_cf)-alpha_sj(trade_baseline)).ravel()
    # data = data_base
    # bars = ax.bar(data.country_name,data.y)
    # ax.bar_label(bars,
    #              labels=data.country_name,
    #              rotation=90,
    #               label_type = 'edge',
    #               padding=2,
    #               # color=colors[g],
    #               zorder=10)
    # ax2.scatter(data.country_name,data.y2,edgecolors='k',color='g')
    # ax2.grid([])
    # ax2.set_ylim([-0.25,0.25])
    # ax.set_ylim([-0.025,0.025])
    # ax2.scatter([],[],color='grey',label = 'change in share')
    # ax2.bar(0,0, label = 'change in emissions\nassociated', color = 'grey')
    # ax.set_xticklabels(['']
    #                     , rotation=45
    #                     , ha='right'
    #                     , rotation_mode='anchor'
    #                     ,fontsize=19)
    # ax.set_ylabel('Change in emissions associated')
    # ax2.set_ylabel('Change in share')
    # plt.title('Term 3')
    # plt.legend(fontsize = 20)
    
    # if save or save_all:
    #     plt.savefig(save_path+'term_3_j.'+save_format,format=save_format)
    
    # plt.show()
    
    # #%% plot term 3 d(alpha) function of epsilon/alpha j specific
    
    # trade_baseline = trade['baseline'].values.reshape((N,S,N))
    # trade_cf = trade[10].values.reshape((N,S,N)) 
    
    # fig, ax = plt.subplots(figsize=(25,15))
    # # colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    # # texts = []
    # countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    # #                                                               b.sector_list],
    # #                                                             names = ['country','sector'])).reset_index()
    # # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    # data_base = countries.copy()
    
    # data_base['x'] = np.einsum('sj,sj->j',epsilon_sj(trade_baseline,e),1/alpha_sj(trade_baseline)).ravel()
    # data_base['y'] = np.einsum('sj->j',alpha_sj(trade_cf)-alpha_sj(trade_baseline)).ravel()
    # # for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    # # data = data_base.loc[data_base['group_label'] == group_label]
    # data= data_base
    # ax.scatter(data.x,data.y
    #            # , label = group_label,color=colors[g]
    #            )
    # texts = [plt.text(data.x.loc[i], 
    #         data.y.loc[i], 
    #         data.loc[i].country_name,
    #         size=15,
    #         # color=colors[g]
    #         ) 
    #         for i in data_base.index]     # For kernel density
    # # texts = texts+texts_group
    # # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # # ax2.grid([])
    # ax.set_ylabel('d(alpha)',fontsize = 30)
    # ax.set_xlabel('Epsilon/alpha',fontsize = 30)
    # # ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
    # # ax.set_xlim([1e-2,3e1])
    # # plt.legend(fontsize = 25)
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.title('Term 3')
    
    # adjust_text(texts, precision=0.001,
    #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    #         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
    #                         ))
    # if save or save_all:
    #     plt.savefig(save_path+'term_3_eps_over_alpha_j.'+save_format,format=save_format)
    
    # plt.show()
    
    # #%% plot term 3 d(alpha) function of epsilon/alpha j specific cons only
    
    # trade_baseline = b.cons.value.values.reshape((N,S,N))
    # trade_cf = sol.cons.value.values.reshape((N,S,N)) 
    
    # fig, ax = plt.subplots(figsize=(25,15))
    # # colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    # # texts = []
    # countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    # #                                                               b.sector_list],
    # #                                                             names = ['country','sector'])).reset_index()
    # # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    # data_base = countries.copy()
    
    # data_base['x'] = np.einsum('sj,sj->j',epsilon_sj(trade_baseline,e),1/alpha_sj(trade_baseline)).ravel()
    # data_base['y'] = np.einsum('sj->j',alpha_sj(trade_cf)-alpha_sj(trade_baseline)).ravel()
    # # for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    # # data = data_base.loc[data_base['group_label'] == group_label]
    # data= data_base
    # ax.scatter(data.x,data.y
    #            # , label = group_label,color=colors[g]
    #            )
    # texts = [plt.text(data.x.loc[i], 
    #         data.y.loc[i], 
    #         data.loc[i].country_name,
    #         size=15,
    #         # color=colors[g]
    #         ) 
    #         for i in data_base.index]     # For kernel density
    # # texts = texts+texts_group
    # # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # # ax2.grid([])
    # ax.set_ylabel('d(alpha)',fontsize = 30)
    # ax.set_xlabel('Epsilon/alpha',fontsize = 30)
    # # ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
    # # ax.set_xlim([1e-2,3e1])
    # # plt.legend(fontsize = 25)
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.title('Term 3 cons only')
    
    # adjust_text(texts, precision=0.001,
    #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    #         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
    #                         ))
    # if save or save_all:
    #     plt.savefig(save_path+'term_3_eps_over_alpha_j_cons_only.'+save_format,format=save_format)
    
    # plt.show()
    
    # #%% plot term 3 d(alpha) function of alpha j specific
    
    # trade_baseline = trade['baseline'].values.reshape((N,S,N))
    # trade_cf = trade[10].values.reshape((N,S,N)) 
    
    # fig, ax = plt.subplots(figsize=(25,15))
    # # colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    # # texts = []
    # countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    # #                                                               b.sector_list],
    # #                                                             names = ['country','sector'])).reset_index()
    # # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    # data_base = countries.copy()
    
    # data_base['x'] = np.einsum('sj->j',alpha_sj(trade_baseline)).ravel()
    # data_base['y'] = np.einsum('sj->j',alpha_sj(trade_cf)-alpha_sj(trade_baseline)).ravel()
    # # for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    # # data = data_base.loc[data_base['group_label'] == group_label]
    # data= data_base
    # ax.scatter(data.x,data.y
    #            # , label = group_label,color=colors[g]
    #            )
    # texts = [plt.text(data.x.loc[i], 
    #         data.y.loc[i], 
    #         data.loc[i].country_name,
    #         size=15,
    #         # color=colors[g]
    #         ) 
    #         for i in data_base.index]     # For kernel density
    # # texts = texts+texts_group
    # # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # # ax2.grid([])
    # ax.set_ylabel('d(alpha)',fontsize = 30)
    # ax.set_xlabel('Alpha',fontsize = 30)
    # # ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
    # # ax.set_xlim([1e-2,3e1])
    # # plt.legend(fontsize = 25)
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.title('Term 3')
    
    # adjust_text(texts, precision=0.001,
    #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    #         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
    #                         ))
    # if save or save_all:
    #     plt.savefig(save_path+'term_3_alpha_j.'+save_format,format=save_format)
    
    # plt.show()
    
    # #%% plot term 4 epsilon(alpha) all terms
    
    # trade_baseline = trade['baseline'].values.reshape((N,S,N))
    # trade_cf = trade[10].values.reshape((N,S,N)) 
    
    # fig, ax = plt.subplots(figsize=(25,15))
    # # colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    # # texts = []
    # countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    #                                                               b.sector_list,
    #                                                               b.country_list],
    #                                                             names = ['row_country','row_sector','col_country'])).reset_index()
    # # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    # # data_base = countries.copy()
    
    # data_base['y'] = np.einsum('isj->isj',epsilon_isj(trade_baseline,e)).ravel()
    # data_base['x'] = np.einsum('isj->isj',alpha_isj(trade_baseline)).ravel()
    # # for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    # # data = data_base.loc[data_base['group_label'] == group_label]
    # data= data_base
    # ax.scatter(data.x,data.y
    #            # , label = group_label,color=colors[g]
    #            )
    # # texts_group = [plt.text(data.x.iloc[i], 
    # #         data.y.iloc[i], 
    # #         data.iloc[i].row_country+','+data.iloc[i].row_sector+','+data.iloc[i].row_sector,
    # #         size=10,
    # #         # color=colors[g]
    # #         ) 
    # #         for i in data_base.index]     # For kernel density
    # # texts = texts+texts_group
    # # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # # ax2.grid([])
    # ax.set_xlabel('Alpha',fontsize = 30)
    # ax.set_ylabel('Epsilon',fontsize = 30)
    # ax.plot([0,data_base.x.max()],[0,data_base.x.max()/S/N],ls='--',color='k')
    # # ax.set_xlim([1e-2,3e1])
    # # plt.legend(fontsize = 25)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title('Term 4')
    
    
    # # adjust_text(texts, precision=0.001,
    # #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    # #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    # #         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
    # #                         ))
    # if save or save_all:
    #     plt.savefig(save_path+'term_4_eps_alpha.'+save_format,format=save_format)
    
    # plt.show()
    
    # #%% plot term 4 function of epsilon/alpha all terms
    
    # trade_baseline = trade['baseline'].values.reshape((N,S,N))
    # trade_cf = trade[10].values.reshape((N,S,N)) 
    
    # fig, ax = plt.subplots(figsize=(25,15))
    # # colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    # # texts = []
    # countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    #                                                               b.sector_list,
    #                                                               b.country_list],
    #                                                             names = ['row_country','row_sector','col_country'])).reset_index()
    # # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    # # data_base = countries.copy()
    
    # data_base['x'] = np.einsum('isj,isj->isj',
    #                            epsilon_isj(trade_baseline,e),
    #                            np.divide(1, 
    #                                      alpha_isj(trade_baseline), 
    #                                      out = np.zeros_like(alpha_isj(trade_baseline)), 
    #                                      where = alpha_isj(trade_baseline)!=0 )
    #                            ).ravel()
    # data_base['y'] = np.einsum('isj->isj',alpha_isj(trade_cf)-alpha_isj(trade_baseline)).ravel()
    # # for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    # # data = data_base.loc[data_base['group_label'] == group_label]
    # data= data_base
    # ax.scatter(data.x,data.y
    #            # , label = group_label,color=colors[g]
    #            )
    # # texts_group = [plt.text(data.x.iloc[i], 
    # #         data.y.iloc[i], 
    # #         data.iloc[i].row_country+','+data.iloc[i].row_sector+','+data.iloc[i].row_sector,
    # #         size=10,
    # #         # color=colors[g]
    # #         ) 
    # #         for i in data_base.index]     # For kernel density
    # # texts = texts+texts_group
    # # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # # ax2.grid([])
    # ax.set_xlabel('Epsilon/Alpha',fontsize = 30)
    # ax.set_ylabel('d(alpha)',fontsize = 30)
    # ax.plot([0,data_base.x.max()],[0,data_base.x.max()/S/N],ls='--',color='k')
    # # ax.set_xlim([1e-2,3e1])
    # # plt.legend(fontsize = 25)
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.title('Term 4')
    
    
    # # adjust_text(texts, precision=0.001,
    # #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    # #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    # #         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
    # #                         ))
    # if save or save_all:
    #     plt.savefig(save_path+'term_4_eps_over_alpha_all_terms.'+save_format,format=save_format)
    
    # plt.show()
    
    # #%% plot term 4 function of epsilon/alpha all terms
    
    # trade_baseline = trade['baseline'].values.reshape((N,S,N))
    # trade_cf = trade[10].values.reshape((N,S,N)) 
    
    
    # # colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    # # texts = []
    # countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    #                                                               b.sector_list,
    #                                                               b.country_list],
    #                                                             names = ['row_country','row_sector','col_country'])).reset_index()
    # # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    # # data_base = countries.copy()
    
    # data_base['x'] = np.einsum('isj,isj->isj',
    #                            epsilon_isj(trade_baseline,e),
    #                            np.divide(1, 
    #                                      alpha_isj(trade_baseline), 
    #                                      out = np.zeros_like(alpha_isj(trade_baseline)), 
    #                                      where = alpha_isj(trade_baseline)!=0 )
    #                            ).ravel()
    # data_base['y'] = np.einsum('isj->isj',alpha_isj(trade_cf)-alpha_isj(trade_baseline)).ravel()
    # # for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    # # data = data_base.loc[data_base['group_label'] == group_label]
    # # data_base = pd.merge
    # for sector in data_base.row_sector.drop_duplicates():
    #     fig, ax = plt.subplots(figsize=(25,15))
    #     data= data_base.loc[data_base.row_sector == sector]
    #     ax.scatter(data.x,data.y
    #                , label = sector
    #                )
    #     plt.xscale('log')
    #     plt.show()
        
    # # texts_group = [plt.text(data.x.iloc[i], 
    # #         data.y.iloc[i], 
    # #         data.iloc[i].row_country+','+data.iloc[i].row_sector+','+data.iloc[i].row_sector,
    # #         size=10,
    # #         # color=colors[g]
    # #         ) 
    # #         for i in data_base.index]     # For kernel density
    # # texts = texts+texts_group
    # # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # # ax2.grid([])
    # ax.set_xlabel('Epsilon/Alpha',fontsize = 30)
    # ax.set_ylabel('d(alpha)',fontsize = 30)
    # ax.plot([0,data_base.x.max()],[0,data_base.x.max()/S/N],ls='--',color='k')
    # # ax.set_xlim([1e-2,3e1])
    # plt.legend(fontsize = 25)
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.title('Term 4')
    
    
    # # adjust_text(texts, precision=0.001,
    # #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    # #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    # #         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
    # #                         ))
    # # if save or save_all:
    # #     plt.savefig(save_path+'term_4_eps_over_alpha_all_terms.'+save_format,format=save_format)
    
    # plt.show()
    
    
    # #%% plot term 4 epsilon(alpha) i specific
    
    # trade_baseline = trade['baseline'].values.reshape((N,S,N))
    # trade_cf = trade[10].values.reshape((N,S,N)) 
    
    # fig, ax = plt.subplots(figsize=(25,15))
    # # colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    # # texts = []
    # countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    # #                                                               b.sector_list],
    # #                                                             names = ['country','sector'])).reset_index()
    # # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    # data_base = countries.copy()
    
    # data_base['y'] = np.einsum('isj->i',epsilon_isj(trade_baseline,e)).ravel()
    # data_base['x'] = np.einsum('isj->i',alpha_isj(trade_baseline)).ravel()/N/S
    # # for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    # # data = data_base.loc[data_base['group_label'] == group_label]
    # data= data_base
    # ax.scatter(data.x,data.y
    #            # , label = group_label,color=colors[g]
    #            )
    # texts = [plt.text(data.x.loc[i], 
    #         data.y.loc[i], 
    #         data.loc[i].country_name,
    #         size=15,
    #         # color=colors[g]
    #         ) 
    #         for i in data_base.index]     # For kernel density
    # # texts = texts+texts_group
    # # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # # ax2.grid([])
    # ax.set_xlabel('Alpha',fontsize = 30)
    # ax.set_ylabel('Epsilon',fontsize = 30)
    # ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
    # # ax.set_xlim([1e-2,3e1])
    # # plt.legend(fontsize = 25)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title('Term 4')
    
    # # adjust_text(texts, precision=0.001,
    # #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    # #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    # #         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
    # #                         ))
    # if save or save_all:
    #     plt.savefig(save_path+'term_4_eps_alpha_i.'+save_format,format=save_format)
    
    # plt.show()
    
    # #%% plot term 4 epsilon(alpha) cons only i specific
    
    # trade_baseline = b.cons.value.values.reshape((N,S,N))
    # trade_cf = sol.cons.value.values.reshape((N,S,N)) 
    
    # fig, ax = plt.subplots(figsize=(25,15))
    # # colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    # # texts = []
    # countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    # #                                                               b.sector_list],
    # #                                                             names = ['country','sector'])).reset_index()
    # # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    # data_base = countries.copy()
    
    # data_base['y'] = np.einsum('isj->i',epsilon_isj(trade_baseline,e)).ravel()
    # data_base['x'] = np.einsum('isj->i',alpha_isj(trade_baseline)).ravel()/N/S
    # # for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    # # data = data_base.loc[data_base['group_label'] == group_label]
    # data= data_base
    # ax.scatter(data.x,data.y
    #            # , label = group_label,color=colors[g]
    #            )
    # texts = [plt.text(data.x.loc[i], 
    #         data.y.loc[i], 
    #         data.loc[i].country_name,
    #         size=15,
    #         # color=colors[g]
    #         ) 
    #         for i in data_base.index]     # For kernel density
    # # texts = texts+texts_group
    # # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # # ax2.grid([])
    # ax.set_xlabel('Alpha',fontsize = 30)
    # ax.set_ylabel('Epsilon',fontsize = 30)
    # ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
    # # ax.set_xlim([1e-2,3e1])
    # # plt.legend(fontsize = 25)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title('Term 4 cons only')
    
    
    # adjust_text(texts, precision=0.001,
    #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    #         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
    #                         ))
    # if save or save_all:
    #     plt.savefig(save_path+'term_4_eps_alpha_i_cons_only.'+save_format,format=save_format)
    
    # plt.show()
    
    # #%% plot term 4
    # trade_baseline = trade['baseline'].values.reshape((N,S,N))
    # trade_cf = trade[10].values.reshape((N,S,N)) 
    
    # fig, ax = plt.subplots(figsize=(25,15))
    # # colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    # # texts = []
    # ax2 = ax.twinx()
    # countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    # #                                                              b.sector_list],
    # #                                                             names = ['country','sector']))
    # # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    # data_base = countries.copy()
    # data_base['y'] = np.einsum('isj,isj,isj->i',
    #            epsilon_isj(trade_baseline,e),
    #            alpha_isj(trade_cf)-alpha_isj(trade_baseline),
    #            np.divide(1, 
    #                      alpha_isj(trade_baseline), 
    #                      out = np.zeros_like(alpha_isj(trade_baseline)), 
    #                      where = alpha_isj(trade_baseline)!=0 ))
    # data_base['y2'] = np.einsum('isj->i',alpha_isj(trade_cf)-alpha_isj(trade_baseline)).ravel()
    # data = data_base
    # bars = ax.bar(data.country_name,data.y)
    # ax.bar_label(bars,
    #              labels=data.country_name,
    #              rotation=90,
    #               label_type = 'edge',
    #               padding=2,
    #               # color=colors[g],
    #               zorder=10)
    # ax2.scatter(data.country_name,data.y2,edgecolors='k',color='g')
    # ax2.grid([])
    # ax2.set_ylim([-5,5])
    # ax.set_ylim([-0.008,0.008])
    # ax2.scatter([],[],color='grey',label = 'change in share')
    # ax2.bar(0,0, label = 'change in emissions\nassociated', color = 'grey')
    # ax.set_xticklabels(['']
    #                     , rotation=45
    #                     , ha='right'
    #                     , rotation_mode='anchor'
    #                     ,fontsize=19)
    # ax.set_ylabel('Change in emissions associated')
    # ax2.set_ylabel('Change in share')
    # plt.title('Term 4')
    # plt.legend(fontsize = 20)
    
    # if save or save_all:
    #     plt.savefig(save_path+'term_4_i.'+save_format,format=save_format)
    
    # plt.show()
    
    # #%% plot term 4 d(alpha) function of epsilon/alpha i specific
    
    # trade_baseline = trade['baseline'].values.reshape((N,S,N))
    # trade_cf = trade[10].values.reshape((N,S,N)) 
    
    # fig, ax = plt.subplots(figsize=(25,15))
    # # colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    # # texts = []
    # countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    # #                                                               b.sector_list],
    # #                                                             names = ['country','sector'])).reset_index()
    # # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    # data_base = countries.copy()
    
    # data_base['x'] = np.einsum('isj,isj->i',epsilon_isj(trade_baseline,e),
    #                            np.divide(1, 
    #                                      alpha_isj(trade_baseline), 
    #                                      out = np.zeros_like(alpha_isj(trade_baseline)), 
    #                                      where = alpha_isj(trade_baseline)!=0 )).ravel()
    # data_base['y'] = np.einsum('isj->i',alpha_isj(trade_cf)-alpha_isj(trade_baseline)).ravel()
    # # for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    # # data = data_base.loc[data_base['group_label'] == group_label]
    # data= data_base
    # ax.scatter(data.x,data.y
    #            # , label = group_label,color=colors[g]
    #            )
    # texts = [plt.text(data.x.loc[i], 
    #         data.y.loc[i], 
    #         data.loc[i].country_name,
    #         size=15,
    #         # color=colors[g]
    #         ) 
    #         for i in data_base.index]     # For kernel density
    # # texts = texts+texts_group
    # # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # # ax2.grid([])
    # ax.set_ylabel('d(alpha)',fontsize = 30)
    # ax.set_xlabel('Epsilon/alpha',fontsize = 30)
    # # ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
    # # ax.set_xlim([1e-2,3e1])
    # # plt.legend(fontsize = 25)
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.title('Term 4')
    
    # adjust_text(texts, precision=0.001,
    #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    #         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
    #                         ))
    # if save or save_all:
    #     plt.savefig(save_path+'term_4_eps_over_alpha_i.'+save_format,format=save_format)
    
    # plt.show()
    
    # #%% plot term 4 d(alpha) function of epsilon/alpha i specific cons only
    
    # trade_baseline = b.cons.value.values.reshape((N,S,N))
    # trade_cf = sol.cons.value.values.reshape((N,S,N)) 
    
    # fig, ax = plt.subplots(figsize=(25,15))
    # # colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    # # texts = []
    # countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    # #                                                               b.sector_list],
    # #                                                             names = ['country','sector'])).reset_index()
    # # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    # data_base = countries.copy()
    
    # data_base['x'] = np.einsum('isj,isj->i',epsilon_isj(trade_baseline,e),
    #                            np.divide(1, 
    #                                      alpha_isj(trade_baseline), 
    #                                      out = np.zeros_like(alpha_isj(trade_baseline)), 
    #                                      where = alpha_isj(trade_baseline)!=0 )).ravel()
    # data_base['y'] = np.einsum('isj->i',alpha_isj(trade_cf)-alpha_isj(trade_baseline)).ravel()
    # # for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    # # data = data_base.loc[data_base['group_label'] == group_label]
    # data= data_base
    # ax.scatter(data.x,data.y
    #            # , label = group_label,color=colors[g]
    #            )
    # texts = [plt.text(data.x.loc[i], 
    #         data.y.loc[i], 
    #         data.loc[i].country_name,
    #         size=15,
    #         # color=colors[g]
    #         ) 
    #         for i in data_base.index]     # For kernel density
    # # texts = texts+texts_group
    # # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # # ax2.grid([])
    # ax.set_ylabel('d(alpha)',fontsize = 30)
    # ax.set_xlabel('Epsilon/alpha',fontsize = 30)
    # # ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
    # # ax.set_xlim([1e-2,3e1])
    # # plt.legend(fontsize = 25)
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.title('Term 4 cons only')
    
    # adjust_text(texts, precision=0.001,
    #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    #         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
    #                         ))
    # if save or save_all:
    #     plt.savefig(save_path+'term_4_eps_over_alpha_i_cons_only.'+save_format,format=save_format)
    
    # plt.show()
    
    # #%% plot term 4 d(alpha) function of alpha i specific
    
    # trade_baseline = trade['baseline'].values.reshape((N,S,N))
    # trade_cf = trade[10].values.reshape((N,S,N)) 
    
    # fig, ax = plt.subplots(figsize=(25,15))
    # # colors_group_sectors = [sns.color_palette()[i] for i in [2,1,3,4,0,5]]
    # # texts = []
    # countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country').set_index('country')
    # # data_base = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list,
    # #                                                               b.sector_list],
    # #                                                             names = ['country','sector'])).reset_index()
    # # data_base = pd.DataFrame(index = pd.Index(b.country_list,name = 'country'))
    # data_base = countries.copy()
    
    # data_base['x'] = np.einsum('isj->i',alpha_isj(trade_baseline)).ravel()
    # data_base['y'] = np.einsum('isj->i',alpha_isj(trade_cf)-alpha_isj(trade_baseline)).ravel()
    # # for g,group_label in enumerate(data_base.group_label.drop_duplicates()):
    # # data = data_base.loc[data_base['group_label'] == group_label]
    # data= data_base
    # ax.scatter(data.x,data.y
    #            # , label = group_label,color=colors[g]
    #            )
    # texts = [plt.text(data.x.loc[i], 
    #         data.y.loc[i], 
    #         data.loc[i].country_name,
    #         size=15,
    #         # color=colors[g]
    #         ) 
    #         for i in data_base.index]     # For kernel density
    # # texts = texts+texts_group
    # # ax2.scatter(np.arange(len(epsilon_s(trade_baseline, e))),epsilon_s(trade_baseline, e)/alpha_s(trade_cf),color = 'r')
    # # ax2.grid([])
    # ax.set_ylabel('d(alpha)',fontsize = 30)
    # ax.set_xlabel('Alpha',fontsize = 30)
    # # ax.plot([0,data_base.x.max()],[0,data_base.x.max()],ls='--',color='k')
    # # ax.set_xlim([1e-2,3e1])
    # # plt.legend(fontsize = 25)
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.title('Term 4')
    
    # adjust_text(texts, precision=0.001,
    #         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
    #         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
    #         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
    #                         ))
    # if save or save_all:
    #     plt.savefig(save_path+'term_4_alpha_i.'+save_format,format=save_format)
    
    # plt.show()
    
    # #%% stacked all terms
    # from matplotlib import cm
    # cmap = cm.get_cmap('Spectral')
    
    
    # trade_baseline = trade['baseline'].values.reshape((N,S,N))
    # trade_cf = trade[10].values.reshape((N,S,N)) 
    
    # data_base = pd.DataFrame(columns=['term','spec1','spec2','value']).set_index(['term','spec1','spec2'])
    
    # data_base.loc[('term 1','',''),'value'] = (X(trade_cf) - X(trade_baseline))/X(trade_baseline)
    
    # term2 = np.einsum('s,s,s->s',
    #                    epsilon_s(trade_baseline,e),
    #                    alpha_s(trade_cf)-alpha_s(trade_baseline),
    #                    1/alpha_s(trade_baseline))
    
    # for i,sector in enumerate(b.sector_list):
    #     data_base.loc[('term 2 reduction',sector_map.loc[sector,'industry'],''),'value'] = term2[i]
        
    # term3 = np.einsum('sj,sj,sj->sj',
    #            epsilon_sj(trade_baseline,e),
    #            alpha_sj(trade_cf)-alpha_sj(trade_baseline),
    #            1/alpha_sj(trade_baseline))
    
    # for i,country in enumerate(b.country_list):
    #     for j,sector in enumerate(b.sector_list):
    #         data_base.loc[('term 3 reduction',sector_map.loc[sector,'industry'],country),'value'] = term3[j,i]
        
    # term4 = np.einsum('isj,isj,isj->is',
    #            epsilon_isj(trade_baseline,e),
    #            alpha_isj(trade_cf)-alpha_isj(trade_baseline),
    #            np.divide(1, 
    #                      alpha_isj(trade_baseline), 
    #                      out = np.zeros_like(alpha_isj(trade_baseline)), 
    #                      where = alpha_isj(trade_baseline)!=0 ))
    
    # for i,exp in tqdm(enumerate(b.country_list)):
    #     for j,sector in enumerate(b.sector_list):
    #         data_base.loc[('term 4 reduction',sector_map.loc[sector,'industry'],exp),'value'] = term4[i,j]
    # # index = pd.MultiIndex.from_product([['term 4'],b.country_list,sector_map.industry.to_list(),b.country_list])
    # # for j,i in tqdm(enumerate(index)):
    # #     data_base.loc[i,'value'] = term4.ravel()[j]
    # # data_base.value = np.abs(data_base.value)
    # # data_base1 = data_base.loc[data_base.value<0].copy()
    # # data_base2 = data_base.loc[data_base.value>0].copy()
    
    # data_base.reset_index(inplace = True)
    # data_base.loc[data_base.value>0, 'term'] = data_base.loc[data_base.value>0, 'term'].str.replace('reduction','increase')
    # data_base.set_index(['term','spec1','spec2'],inplace=True)
    # data_base.sort_index(inplace=True)
    # data_base.value = np.abs(data_base.value)
    # # data_base.loc[('untouched emissions','',''),'value'] = 1-data_base.value.sum()
    # data_base.reset_index(inplace = True)
    # data_base = data_base.replace('',None)
    
    # # data_base2.value = np.abs(data_base2.value)
    # # data_base2.reset_index(inplace = True)
    # # data_base2 = data_base2.replace('',None)
    
    
    # # fig, ax = plt.subplots(figsize=(25,15))
    # # print('here')
    # # ax.stackplot(data_base.value.to_list(),
    # #               labels=data_base.spec.to_list() )
    # # ax.plot(carb_taxes[1:],[l_em_incr[:i].sum() for i in range(len(l_em_incr))], 
    # #           label='Emissions',color='black'
    # #           ,lw=3)
    # # ax.plot(carb_taxes,norm_emissions-1, 
    # #           label='Emissions real',color='y'
    # #           ,lw=3)
    # # ax.legend(loc='lower left',fontsize = 20)
    # # ax.tick_params(axis='both', which='major', labelsize=15)
    # # ax.set_xlabel('Carbon tax',fontsize = 20)
    
    # # ax = data_base.T.plot(kind='bar', stacked=True,legend=False, cmap=cmap, linewidth=0.5,edgecolor='grey'
    # #                  , figsize=(20, 10))
    # # for c in ax.containers:
    
    # #     # Optional: if the segment is small or 0, customize the labels
    # #     # labels = [v.get_height() if v.get_height() > 0 else '' for v in c]
    # #     ax.bar_label(c, labels=data_base.index.get_level_values(1).to_list(), label_type='center')
    
    # # plt.show()
    # #%%
    # import plotly.io as pio
    # import plotly.express as px
    # # import plotly.graph_objects as go
    # pio.renderers.default='browser'
    # color_discrete_map = {
    #     'term 1':sns.color_palette("Paired").as_hex()[1],
    #     'term 2 reduction':sns.color_palette("Paired").as_hex()[3],
    #     'term 2 increase':sns.color_palette("Paired").as_hex()[2],
    #     'term 3 reduction':sns.color_palette("Paired").as_hex()[5],
    #     'term 3 increase':sns.color_palette("Paired").as_hex()[4],
    #     'term 4 reduction':sns.color_palette("Paired").as_hex()[7],
    #     'term 4 increase':sns.color_palette("Paired").as_hex()[6],
    #     # 'untouched emissions':sns.color_palette("Paired").as_hex()[8]
    #     }
    # # color_discrete_map = {
    # #     'term 1':sns.color_palette().as_hex()[0],
    # #     'term 2':sns.color_palette().as_hex()[1],
    # #     'term 2 positive':sns.color_palette().as_hex()[1],
    # #     'term 3':sns.color_palette().as_hex()[2],
    # #     'term 3 positive':sns.color_palette().as_hex()[2],
    # #     'term 4':sns.color_palette().as_hex()[3],
    # #     'term 4 positive':sns.color_palette().as_hex()[3],
    # #     'untouched emissions':sns.color_palette("Paired").as_hex()[8]
    # #     }
    
    # # color_discrete_map = {term:sns.color_palette("Paired").as_hex() for i,term in enumerate(data_base.reset_index().term.drop_duplicates())}
    
    # fig1 = px.sunburst(data_base, path=['term', 'spec1','spec2'], values='value', color='term',
    #                     color_discrete_map=color_discrete_map)
    # fig1.update_traces(sort=False, selector=dict(type='sunburst')) 
    # # fig1 = go.Figure()
    # # fig1.add_trace(go.Sunburst(ids=data_base1.spec2, values=data_base1.value))
    # # fig2.update_layout(title_text="Two-level Sunburst Diagram", font_size=10)
    # fig1.show()
    # # fig2 = px.sunburst(data_base2, path=['term', 'spec1','spec2'], values='value', color='term',
    # #                    color_discrete_map=color_discrete_map)
    # # # fig2.update_layout(title_text="Two-level Sunburst Diagram", font_size=10)
    # # fig2.show()
    
    # #%% stacked all terms
    # from matplotlib import cm
    # cmap = cm.get_cmap('Spectral')
    
    
    # trade_baseline = trade['baseline'].values.reshape((N,S,N))
    # trade_cf = trade[10].values.reshape((N,S,N)) 
    
    # data_base = pd.DataFrame(columns=['term','spec1','spec2','value']).set_index(['term','spec1','spec2'])
    
    # data_base.loc[('term 1','',''),'value'] = (X(trade_cf) - X(trade_baseline))/X(trade_baseline)
    
    # term2 = np.einsum('s->s',
    #                    alpha_s(trade_cf)-alpha_s(trade_baseline))
    
    # for i,sector in enumerate(b.sector_list):
    #     data_base.loc[('term 2 reduction',sector_map.loc[sector,'industry'],''),'value'] = term2[i]
        
    # term3 = np.einsum('sj,->sj',
    #            alpha_sj(trade_cf)-alpha_sj(trade_baseline))
    
    # for i,country in enumerate(b.country_list):
    #     for j,sector in enumerate(b.sector_list):
    #         data_base.loc[('term 3 reduction',sector_map.loc[sector,'industry'],country),'value'] = term3[j,i]
        
    # term4 = np.einsum('isj,->is',
    #            alpha_isj(trade_cf)-alpha_isj(trade_baseline))
    
    # for i,exp in tqdm(enumerate(b.country_list)):
    #     for j,sector in enumerate(b.sector_list):
    #         data_base.loc[('term 4 reduction',sector_map.loc[sector,'industry'],exp),'value'] = term4[i,j]
    # # index = pd.MultiIndex.from_product([['term 4'],b.country_list,sector_map.industry.to_list(),b.country_list])
    # # for j,i in tqdm(enumerate(index)):
    # #     data_base.loc[i,'value'] = term4.ravel()[j]
    # # data_base.value = np.abs(data_base.value)
    # # data_base1 = data_base.loc[data_base.value<0].copy()
    # # data_base2 = data_base.loc[data_base.value>0].copy()
    
    # data_base.reset_index(inplace = True)
    # data_base.loc[data_base.value>0, 'term'] = data_base.loc[data_base.value>0, 'term'].str.replace('reduction','increase')
    # data_base.set_index(['term','spec1','spec2'],inplace=True)
    # data_base.sort_index(inplace=True)
    # data_base.value = np.abs(data_base.value)
    # # data_base.loc[('untouched emissions','',''),'value'] = 1-data_base.value.sum()
    # data_base.reset_index(inplace = True)
    # data_base = data_base.replace('',None)
    
    # # data_base2.value = np.abs(data_base2.value)
    # # data_base2.reset_index(inplace = True)
    # # data_base2 = data_base2.replace('',None)
    
    
    # # fig, ax = plt.subplots(figsize=(25,15))
    # # print('here')
    # # ax.stackplot(data_base.value.to_list(),
    # #               labels=data_base.spec.to_list() )
    # # ax.plot(carb_taxes[1:],[l_em_incr[:i].sum() for i in range(len(l_em_incr))], 
    # #           label='Emissions',color='black'
    # #           ,lw=3)
    # # ax.plot(carb_taxes,norm_emissions-1, 
    # #           label='Emissions real',color='y'
    # #           ,lw=3)
    # # ax.legend(loc='lower left',fontsize = 20)
    # # ax.tick_params(axis='both', which='major', labelsize=15)
    # # ax.set_xlabel('Carbon tax',fontsize = 20)
    
    # # ax = data_base.T.plot(kind='bar', stacked=True,legend=False, cmap=cmap, linewidth=0.5,edgecolor='grey'
    # #                  , figsize=(20, 10))
    # # for c in ax.containers:
    
    # #     # Optional: if the segment is small or 0, customize the labels
    # #     # labels = [v.get_height() if v.get_height() > 0 else '' for v in c]
    # #     ax.bar_label(c, labels=data_base.index.get_level_values(1).to_list(), label_type='center')
    
    # # plt.show()
    
    # import plotly.io as pio
    # import plotly.express as px
    # # import plotly.graph_objects as go
    # pio.renderers.default='browser'
    # color_discrete_map = {
    #     'term 1':sns.color_palette("Paired").as_hex()[1],
    #     'term 2 reduction':sns.color_palette("Paired").as_hex()[3],
    #     'term 2 increase':sns.color_palette("Paired").as_hex()[2],
    #     'term 3 reduction':sns.color_palette("Paired").as_hex()[5],
    #     'term 3 increase':sns.color_palette("Paired").as_hex()[4],
    #     'term 4 reduction':sns.color_palette("Paired").as_hex()[7],
    #     'term 4 increase':sns.color_palette("Paired").as_hex()[6],
    #     # 'untouched emissions':sns.color_palette("Paired").as_hex()[8]
    #     }
    # # color_discrete_map = {
    # #     'term 1':sns.color_palette().as_hex()[0],
    # #     'term 2':sns.color_palette().as_hex()[1],
    # #     'term 2 positive':sns.color_palette().as_hex()[1],
    # #     'term 3':sns.color_palette().as_hex()[2],
    # #     'term 3 positive':sns.color_palette().as_hex()[2],
    # #     'term 4':sns.color_palette().as_hex()[3],
    # #     'term 4 positive':sns.color_palette().as_hex()[3],
    # #     'untouched emissions':sns.color_palette("Paired").as_hex()[8]
    # #     }
    
    # # color_discrete_map = {term:sns.color_palette("Paired").as_hex() for i,term in enumerate(data_base.reset_index().term.drop_duplicates())}
    
    # fig1 = px.sunburst(data_base, path=['term', 'spec1','spec2'], values='value', color='term',
    #                     color_discrete_map=color_discrete_map)
    # fig1.update_traces(sort=False, selector=dict(type='sunburst')) 
    # # fig1 = go.Figure()
    # # fig1.add_trace(go.Sunburst(ids=data_base1.spec2, values=data_base1.value))
    # # fig2.update_layout(title_text="Two-level Sunburst Diagram", font_size=10)
    # fig1.show()
    # # fig2 = px.sunburst(data_base2, path=['term', 'spec1','spec2'], values='value', color='term',
    # #                    color_discrete_map=color_discrete_map)
    # # # fig2.update_layout(title_text="Two-level Sunburst Diagram", font_size=10)
    # # fig2.show()
