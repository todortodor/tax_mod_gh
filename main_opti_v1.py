#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 14:17:32 2022

@author: simonl
"""

main_path = './'
import sys
sys.path.append(main_path+'lib/')
import solver_funcs as s
import data_funcs as d
from tqdm import tqdm
import numpy as np
# from deco import *
import pandas as pd
import time
from time import perf_counter
# from multiprocessing import Pool
# from multiprocessing import Manager

# dir_num = 227
data_path = main_path+'data/'
results_path = 'results/'

numeraire_type = 'wage'
numeraire_country = 'WLD'

EEA = d.countries_from_fta('EEA')
EU = d.countries_from_fta('EU')
EU_ETS = d.countries_from_fta('EU_ETS')
NAFTA = d.countries_from_fta('NAFTA')
ASEAN = d.countries_from_fta('ASEAN')
AANZFTA = d.countries_from_fta('AANZFTA')
APTA = d.countries_from_fta('APTA')
MERCOSUR = d.countries_from_fta('MERCOSUR')
G20 = d.countries_from_fta('G20')
CLUB1 = EU + ['CHN','USA','IND','CAN','GBR']
CLUB2 = EU + ['CHN','USA','IND','IDN','RUS','BRA']
CLUB3 = EU + ['CHN','USA','IND','IDN','RUS','BRA','ZAF']

# carb_cost_list = np.append(np.linspace(0,2.5e-4,251),np.linspace(2.5e-4,1e-3,76)[1:])[46:]
# carb_cost_list = np.concatenate([np.linspace(0,1e-4,101),np.linspace(1e-4,1e-3,91)[1:]])
# carb_cost_list = np.concatenate([np.array([0]),np.logspace(-6,-3,16)])
# carb_cost_list = np.linspace(0,1e-4,101)
# carb_cost_list = [1.4e-05]
carb_cost_list = [1e-4]
# carb_cost_list = [None]
# carb_cost_list = np.concatenate([np.array([0]),np.logspace(-4,2,61)])
# eta_path = ['uniform_elasticities_4.csv']
# sigma_path = ['uniform_elasticities_4.csv']
# eta_path = ['elasticities_agg1.csv','elasticities_agg2.csv','uniform_elasticities_4.csv']
# sigma_path = ['elasticities_agg1.csv','elasticities_agg2.csv','uniform_elasticities_4.csv']
# eta_path = ['rescaled_to_4elasticities_agg1.csv',
#  'rescaled_to_5elasticities_agg1.csv',
#  'rescaled_to_4_output_weightedelasticities_agg1.csv',
#  'rescaled_to_5_output_weightedelasticities_agg1.csv',
#  'rescaled_to_4elasticities_agg2.csv',
#  'rescaled_to_5elasticities_agg2.csv',
#  'rescaled_to_4_output_weightedelasticities_agg2.csv',
#  'rescaled_to_5_output_weightedelasticities_agg2.csv']
# sigma_path = ['rescaled_to_4elasticities_agg1.csv',
#  'rescaled_to_5elasticities_agg1.csv',
#  'rescaled_to_4_output_weightedelasticities_agg1.csv',
#  'rescaled_to_5_output_weightedelasticities_agg1.csv',
#  'rescaled_to_4elasticities_agg2.csv',
#  'rescaled_to_5elasticities_agg2.csv',
#  'rescaled_to_4_output_weightedelasticities_agg2.csv',
#  'rescaled_to_5_output_weightedelasticities_agg2.csv']
# eta_path = ["sigmaNewConcordance.csv",
# "sigmaNewConcordanceNoAgNoEl.csv",
# "sigmaOldConcordance.csv"]
# sigma_path = ["sigmaNewConcordance.csv",
# "sigmaNewConcordanceNoAgNoEl.csv",
# "sigmaOldConcordance.csv"]
eta_path = ["cp_estimate_allyears.csv"]
sigma_path = ["cp_estimate_allyears.csv"]
# eta_path = ["elasticities_agg1.csv"]
# sigma_path = ["elasticities_agg1.csv"]
# eta_path = ["fgo_estimate.csv"]
# sigma_path = ["fgo_estimate.csv"]
# eta_path = ["rescaled_to_4elasticities_agg1.csv",
# "rescaled_to_5elasticities_agg1.csv",
# "elasticities_agg1.csv",
# "uniform_elasticities_4.csv"]
# sigma_path = ["rescaled_to_4elasticities_agg1.csv",
# "rescaled_to_5elasticities_agg1.csv",
# "elasticities_agg1.csv",
# "uniform_elasticities_4.csv"]
# sigma_path = ['uniform_elasticities_4.csv']
# carb_cost_list = [4.6e-4]
taxed_countries_list = [None]
# taxed_countries_list = [EU,CLUB2]
# taxing_countries_list = [None,EU,NAFTA,ASEAN,AANZFTA,APTA,EEA,MERCOSUR,
#                           ['USA'],['CHN'],
#                           EEA+NAFTA,EEA+ASEAN,EEA+APTA,EEA+AANZFTA,EEA+['USA'],EEA+['CHN'],
#                           NAFTA+APTA,NAFTA+MERCOSUR,
#                           APTA+AANZFTA,EU+NAFTA+['CHN'],EU+NAFTA+APTA]
# taxing_countries_list = [high_income_countries,mid_income_countries,['EU']]
taxing_countries_list = [EU_ETS]
# taxing_countries_list = [CLUB2]
# taxing_countries_list = [CLUB3]
# taxing_countries_list = [None]

taxed_sectors_list = [None]

# spec_tax = pd.DataFrame(index = pd.MultiIndex.from_product([d.get_country_list(),
#                                                             d.get_sector_list(),
#                                                             d.get_country_list()],
#                                                             names = ['row_country',
#                                                                     'row_sector',
#                                                                     'col_country']),
#                         columns = ['value'])
# spec_tax['value'] = 0

# spec_tax.loc[spec_tax.query("row_country != col_country").index, 'value'] = 0

# if False:
#     baseline = d.baseline(2018, data_path)

# va = baseline.va.groupby('col_country').sum()
# labor = baseline.labor.set_index('country').rename_axis('col_country')['2018'].to_frame()
# labor.columns = ['value']

# gdp_p_c = va/labor

# poor_countries = gdp_p_c.loc[gdp_p_c.value < gdp_p_c.value.loc['PER']].index.to_list()
# emerging_countries = gdp_p_c.loc[(gdp_p_c.value >= gdp_p_c.value.loc['PER']) & 
#                                  (gdp_p_c.value < gdp_p_c.value.loc['POL'])].index.to_list()
# rich_countries = gdp_p_c.loc[gdp_p_c.value >= gdp_p_c.value.loc['POL']].index.to_list()

# spec_tax = spec_tax.reset_index()

# # spec_tax.loc[(spec_tax.col_country.isin(poor_countries)),'value'] = 2.5e-5
# # spec_tax.loc[(spec_tax.col_country.isin(emerging_countries)),'value'] = 5e-5
# # spec_tax.loc[(spec_tax.col_country.isin(rich_countries)),'value'] = 7.5e-5

# # spec_tax.loc[(spec_tax.col_country.isin(poor_countries)) & 
# #              (spec_tax.col_country.isin(G20)),'value'] = 2.5e-5
# # spec_tax.loc[(spec_tax.col_country.isin(emerging_countries)) & 
# #              (spec_tax.col_country.isin(G20)),'value'] = 5e-5
# # spec_tax.loc[(spec_tax.col_country.isin(rich_countries)) & 
# #              (spec_tax.col_country.isin(G20)),'value'] = 7.5e-5

# # spec_tax.loc[(spec_tax.col_country.isin(poor_countries)) & 
# #              (spec_tax.col_country.isin(CLUB1)),'value'] = 2.5e-5
# # spec_tax.loc[(spec_tax.col_country.isin(emerging_countries)) & 
# #              (spec_tax.col_country.isin(CLUB1)),'value'] = 5e-5
# # spec_tax.loc[(spec_tax.col_country.isin(rich_countries)) & 
# #              (spec_tax.col_country.isin(CLUB1)),'value'] = 7.5e-5

# # spec_tax.loc[(spec_tax.col_country.isin(poor_countries)) & 
# #              (spec_tax.col_country.isin(CLUB2)),'value'] = 2.5e-5
# # spec_tax.loc[(spec_tax.col_country.isin(emerging_countries)) & 
# #              (spec_tax.col_country.isin(CLUB2)),'value'] = 5e-5
# # spec_tax.loc[(spec_tax.col_country.isin(rich_countries)) & 
# #              (spec_tax.col_country.isin(CLUB2)),'value'] = 7.5e-5

# spec_tax.loc[(spec_tax.col_country.isin(poor_countries)) & 
#              (spec_tax.col_country.isin(CLUB3)),'value'] = 2.5e-5
# spec_tax.loc[(spec_tax.col_country.isin(emerging_countries)) & 
#              (spec_tax.col_country.isin(CLUB3)),'value'] = 5e-5
# spec_tax.loc[(spec_tax.col_country.isin(rich_countries)) & 
#              (spec_tax.col_country.isin(CLUB3)),'value'] = 7.5e-5

# # spec_tax.loc[(spec_tax.row_country.isin(EU)),'value'] = 1e-4
# # spec_tax.loc[(spec_tax.col_country.isin(EU)),'value'] = 100

# # spec_tax.loc[(spec_tax.col_country.isin(rich_countries)) & 
# #              (spec_tax.col_country.isin(CLUB2)),'value'] = 7.5e-5

# # specific_taxing_list = [spec_tax.set_index(['row_country', 'row_sector', 'col_country']
# #                                            )*x for x in np.linspace(0,1,101)]
# specific_taxing_list = [spec_tax.set_index(['row_country', 'row_sector', 'col_country']
#                                            )]

specific_taxing_list = [None]

# for cab_price in ['fair_cab_price.csv','fair_cab_price_2.csv','fair_cab_price_3.csv']:
#     spec_tax = pd.DataFrame(index = pd.MultiIndex.from_product([d.get_country_list(),
#                                                                 d.get_sector_list(),
#                                                                 d.get_country_list()],
#                                                                 names = ['row_country',
#                                                                         'row_sector',
#                                                                         'col_country'])).reset_index()
    
#     fair_carb_price = pd.read_csv('fair_cab_price.csv').set_index('col_country')#*1e6
#     # fair_carb_price.iloc[0].value = fair_carb_price.value.iloc[0]/2
#     # fair_carb_price = fair_carb_price*10
    
#     spec_tax = pd.merge(spec_tax,fair_carb_price,on='col_country')
    
#     spec_tax = spec_tax.set_index(['row_country',
#             'row_sector',
#             'col_country'])

#     specific_taxing_list.append(spec_tax)

# eta_path = ["elasticities_agg2.csv"]
# sigma_path = ["elasticities_agg2.csv"]
# eta_path = ["uniform_elasticities_4.csv"]
# sigma_path = ["uniform_elasticities_4.csv"]
# eta_path = ["rescaled_to_4elasticities_agg2.csv"]
# sigma_path = ["rescaled_to_4elasticities_agg2.csv"]
# eta_path = ["fgo_estimate.csv"]
# sigma_path = ["fgo_estimate.csv"]
# fair_tax_list = [False]
# pol_pay_tax_list = [False]
tax_scheme_list = ['consumer','producer','eu_style']
# tax_scheme_list = ['consumer']
# tax_scheme_list = ['eu_style']

dir_num = 57

numeraire_type = 'wage'
numeraire_country = 'WLD'

# fair_tax_list = [False,True]
# pol_pay_tax_list = [False,True]
# fair_tax_list = [True]
fair_tax_list = [False]
pol_pay_tax_list = [False]

tau_factor_list = [1]
# tau_factor_list = np.linspace(1,5,9)
# tau_factor_list = [1e5]
autarky = False

cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                      taxed_sectors_list,specific_taxing_list,fair_tax_list,pol_pay_tax_list,tax_scheme_list,tau_factor_list,
                      same_elasticities=True)
exclude_direct_emissions = False

years = [2018]

E_init = None

reload_baseline = True
      
for y in years:
    # y=2018         
    year=str(y)
    
    if reload_baseline:
    
        baseline = d.baseline(year, data_path, exclude_direct_emissions=exclude_direct_emissions)
        
        baseline.num_scale(numeraire_type, numeraire_country, inplace = True)
        
        # baseline.num_type = 'no_scaling'
        # baseline.num = 1
        
        baseline.make_np_arrays(inplace = True)
        
        baseline.compute_shares_and_gammas(inplace = True)
        
        # baseline.co2_intensity = baseline.co2_intensity/1e6
        # baseline.co2_intensity_np = baseline.co2_intensity_np/1e6
        # baseline.co2_prod = baseline.co2_prod/1e6
        # baseline.co2_prod_np = baseline.co2_prod_np/1e6
    
    #%%
    import solver_funcs as s
    import data_funcs as d
    import treatment_funcs as t
    vec_init = None
    for simulation_case in tqdm(cases):
        if simulation_case['eta_path'] == simulation_case['sigma_path']:
            
            params = d.params(data_path, **simulation_case)
            #
            params.num_scale_carb_cost(baseline.num, inplace = True)
            # # start = time.perf_counter()
            if not (params.fair_tax and params.pol_pay_tax):
                if not params.fair_tax and not params.pol_pay_tax and not autarky:
                    # print(params.carb_cost_df)
                    # res = pd.read_csv('results/2018_75/401_results.csv')
                    # vec_init = np.concatenate([res['output_hat'].values,
                    #                             res['price_hat'].values,
                    #                             res.groupby('country').first()['spending_hat'].values] )
                    results = s.solve_one_loop(params, baseline, vec_init = vec_init,
                                               # tol=1e-4,
                                               damping=5)
                    vec_init = np.concatenate([results['E_hat'].ravel(),
                                                results['p_hat'].ravel(),
                                                results['I_hat'].ravel()] )
                    vec_init = None
                # print(time.perf_counter() - start)
                # E_init = results['E_hat'].copy()
                
                # if autarky:
                #     results = s.solve_autarky(params, baseline, vec_init = vec_init)
                #     vec_init = np.concatenate([results['E_hat'].ravel(),
                #                                 results['p_hat'].ravel(),
                #                                 results['I_hat'].ravel()] )
                
                if params.fair_tax:
                    results = s.solve_fair_tax(params, baseline)
                    
                if params.pol_pay_tax:
                    results = s.solve_pol_pay_tax(params, baseline)
                
                
                # temp = {}
                # results = s.solve_fair_carb_price(params, baseline, temp)
                
                #compute some aggregated solution quantities to write directly in runs report
                emissions_sol, utility, utility_countries = s.compute_emissions_utility(results, params, baseline, autarky=autarky)
                
                d.write_solution_csv(results,results_path,dir_num,emissions_sol,utility,params,baseline,autarky)
