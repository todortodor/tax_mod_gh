#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 08:39:51 2022

@author: simonl
"""

import pandas as pd
import numpy as np
import lib.solver_funcs as s
import lib.data_funcs as d
import sys
from copy import deepcopy
import itertools
from tqdm import tqdm
from ast import literal_eval

def dir_path(results_path,year,dir_num):
    if isinstance(year,int):
        year = str(year)
    return results_path+year+'_'+str(dir_num)

class sol:
    def __init__(self, run, results_path, data_path):
        self.run = run
        self.res = pd.read_csv(results_path+run.path)
        self.results_path = results_path
        
        if self.run.fair_tax or self.run.pol_pay_tax:
            self.contrib = pd.read_csv(results_path+run.path[:-4]+'_contrib.csv',index_col=0)
        else:
            self.contrib = None

        try:
            taxed_countries = literal_eval(run.taxed_countries)
        except:
            taxed_countries = run.taxed_countries
        try:
            taxing_countries = literal_eval(run.taxing_countries)
        except:
            taxing_countries = run.taxing_countries
        try:
            taxed_sectors = literal_eval(run.taxed_sectors)
        except:
            taxed_sectors = run.taxed_sectors
        try:
            tax_scheme = run.tax_scheme
        except:
            tax_scheme = 'consumer'
        try:
            tau_factor = run.tau_factor
        except:
            tau_factor = 1
        
        if 'specific' in run.tax_type:
            self.params = d.params(data_path,
                                   eta_path =run.eta_path,
                                   sigma_path = run.sigma_path,
                                   specific_taxing = pd.read_csv(self.run.path_tax_scheme, 
                                                                 index_col=['row_country','row_sector','col_country']),
                                   fair_tax = run.fair_tax,
                                   pol_pay_tax = run.pol_pay_tax,
                                   tax_scheme = tax_scheme,
                                   tau_factor = tau_factor
                                   )
        
        if 'specific' not in run.tax_type:
            self.params = d.params(data_path,
                                   eta_path =run.eta_path,
                                   sigma_path = run.sigma_path,
                                   carb_cost = run.carb_cost,
                                   taxed_countries = taxed_countries,
                                   taxing_countries = taxing_countries,
                                   taxed_sectors = taxed_sectors,
                                   fair_tax = run.fair_tax,
                                   pol_pay_tax = run.pol_pay_tax,
                                   tax_scheme = tax_scheme,
                                   tau_factor = tau_factor
                                   )
            
    
    def elements(self):
        for key, item in sorted(self.__dict__.items()):
            print(key, ',', str(type(item))[8:-2])    
    
    def copy(self):
        frame = deepcopy(self)
        return frame
    
    def memory(self, details = False):
        print('Sol class takes up ', 
              sum([sys.getsizeof(x) for x in self.__dict__.values()])/1e6, 
              ' Mb')
        if details:
            for key,item in sorted(self.__dict__.items()):
                print(key, sys.getsizeof(item)/1e6, 'Mb')   
    
    def compute_solution(self,baseline,inplace=False,with_np_arrays=False):
        
        if inplace:
            frame = self
        else:
            frame = self.copy()
            
        p = frame.params
        b = baseline.make_np_arrays().compute_shares_and_gammas()
        
        E_hat_sol = frame.res.output_hat.values.reshape(b.country_number, b.sector_number)
        p_hat_sol = frame.res.price_hat.values.reshape(b.country_number, b.sector_number)
        I_hat_sol = frame.res.spending_hat.values.reshape(b.country_number, b.sector_number)[:,0]
        if frame.run.fair_tax:
            b.deficit_np = b.deficit_np + frame.contrib.value.values
        if frame.run.pol_pay_tax:
            b.deficit_np = b.deficit_np + frame.contrib.value.values
            
        iot_hat_unit = s.iot_eq_unit(p_hat_sol, p, b) 
        cons_hat_unit = s.cons_eq_unit(p_hat_sol, p, b)       
        beta = np.einsum('itj->tj',b.cons_np) / np.einsum('itj->j',b.cons_np)

        taxed_price = p_hat_sol[:,:,None]*p.tau_hat*(1+p.carb_cost_np*b.co2_intensity_np[:,:,None])
        if p.tax_scheme == 'eu_style':
            taxed_price = np.einsum('it,itj,itj->itj',
                                    p_hat_sol,
                                    p.tau_hat,
                                    (1+np.maximum(p.carb_cost_np,np.einsum('itj->jti',p.carb_cost_np))*b.co2_intensity_np[:,:,None]))
        consumer_price_agg = np.einsum('itj,itj->tj'
                                  ,taxed_price**(1-p.sigma[None,:,None]) 
                                  ,b.share_cons_o_np 
                                  ) ** (1/(1 - p.sigma[:,None]))
        price_agg_no_pow = np.einsum('itj,itjs->tjs'
                                  ,taxed_price**(1-p.eta[None,:,None]) 
                                  ,b.share_cs_o_np 
                                  )       
        producer_price_agg = np.divide(1, 
                        price_agg_no_pow , 
                        out = np.ones_like(price_agg_no_pow), 
                        where = price_agg_no_pow!=0 ) ** (1/(p.eta[:,None,None] - 1))  
        
        # I_hat_sol = s.compute_I_hat(p_hat_sol, E_hat_sol, p, b)
        
        iot = np.einsum('it,js,itjs,itjs -> itjs', p_hat_sol, E_hat_sol , iot_hat_unit , b.iot_np)
        cons = np.einsum('it,j,itj,itj -> itj', p_hat_sol, I_hat_sol , cons_hat_unit , b.cons_np)
        va = E_hat_sol * b.va_np
        output = E_hat_sol * b.output_np
        co2_prod = E_hat_sol * b.co2_prod_np / p_hat_sol
        
        cons_hat_sol = np.einsum('j,itj->itj',  I_hat_sol , cons_hat_unit)
        utility_cs_hat_sol = np.einsum('itj,itj->tj', 
                                        cons_hat_sol**((p.sigma[None,:,None]-1)/p.sigma[None,:,None]) , 
                                        b.share_cons_o_np ) ** (p.sigma[:,None] / (p.sigma[:,None]-1))
        utility = (utility_cs_hat_sol**beta).prod(axis=0)
        
        if with_np_arrays:
            frame.iot_np = iot
            frame.cons_np = cons
            frame.trade_np = iot.sum(axis=-1)+cons
        
        frame.iot = pd.DataFrame(index = b.iot.index,
                                    data = iot.ravel(),
                                    columns = ['value'])
        
        frame.cons = pd.DataFrame(index = b.cons.index,
                                    data = cons.ravel(),
                                    columns = ['value'])
        
        frame.va = pd.DataFrame(index = b.va.index,
                                    data = va.ravel(),
                                    columns = ['value'])
        
        frame.output = pd.DataFrame(index = b.output.index,
                                    data = output.ravel(),
                                    columns = ['value'])
        
        frame.co2_prod = pd.DataFrame(index = b.co2_prod.index,
                                    data = co2_prod.ravel(),
                                    columns = ['value'])
            
        frame.price = pd.DataFrame(index = pd.MultiIndex.from_product(
                                                        [b.country_list,b.sector_list],
                                                        names = ['row_country','row_sector']),
                                    data = p_hat_sol.ravel(),
                                    columns = ['hat'])#.reset_index()
        
        frame.taxed_price = pd.DataFrame(index = pd.MultiIndex.from_product(
                                                        [b.country_list,b.sector_list,b.country_list],
                                                        names = ['row_country','row_sector','col_country']),
                                    data = taxed_price.ravel(),
                                    columns = ['hat'])#.reset_index()
        
        frame.consumer_price_agg = pd.DataFrame(index = pd.MultiIndex.from_product(
                                                        [b.sector_list,b.country_list],
                                                        names = ['row_sector','col_country']), 
                                                data = consumer_price_agg.ravel(),
                                                columns = ['hat'])#.reset_index()
        
        frame.producer_price_agg = pd.DataFrame(index = pd.MultiIndex.from_product(
                                                        [b.sector_list,b.country_list,b.sector_list],
                                                        names = ['row_sector','col_country','col_sector']), 
                                                data = producer_price_agg.ravel(),
                                                columns = ['hat'])#.reset_index()
        
        frame.utility = pd.DataFrame(index = pd.Index(b.country_list,name='country'), 
                                    data = utility.ravel(),
                                    columns = ['hat'])#.reset_index()
        
        return frame
    
    def compute_trade_only(self,baseline):

        p = self.params
        b = baseline.make_np_arrays().compute_shares_and_gammas()
        
        E_hat_sol = self.res.output_hat.values.reshape(b.country_number, b.sector_number)
        p_hat_sol = self.res.price_hat.values.reshape(b.country_number, b.sector_number)
        I_hat_sol = self.res.spending_hat.values.reshape(b.country_number, b.sector_number)[:,0]
        if self.run.fair_tax:
            b.deficit_np = b.deficit_np + self.contrib.value.values
        if self.run.pol_pay_tax:
            b.deficit_np = b.deficit_np + self.contrib.value.values
            
        iot_hat_unit = s.iot_eq_unit(p_hat_sol, p, b) 
        cons_hat_unit = s.cons_eq_unit(p_hat_sol, p, b)       
        
        # I_hat_sol = s.compute_I_hat(p_hat_sol, E_hat_sol, p, b)
        
        iot = np.einsum('it,itj,js,itjs,itjs -> itjs', p_hat_sol, p.tau_hat, E_hat_sol , iot_hat_unit , b.iot_np)
        cons = np.einsum('it,itj,j,itj,itj -> itj', p_hat_sol, p.tau_hat, I_hat_sol , cons_hat_unit , b.cons_np)
        
        return iot.sum(axis=-1) + cons
    
    def compute_hat(self,baseline,inplace=False):
        
        if inplace:
            frame = self
        else:
            frame = self.copy()
        
        frame.iot['hat'] = frame.iot['value'] / baseline.iot['value']
        frame.cons['hat'] = frame.cons['value'] / baseline.cons['value']
        frame.va['hat'] = frame.va['value'] / baseline.va['value']
        frame.output['hat'] = frame.output['value'] / baseline.output['value']
        frame.co2_prod['hat'] = frame.co2_prod['value'] / baseline.co2_prod['value']
        
        return frame
        
    @staticmethod
    def load_sols(cases,
                  years,
                  dir_num,
                  results_path,
                  data_path,
                  baselines=None,
                  compute_sols = True,
                  compute_hats = False,
                  return_not_found_cases=False,
                  drop_duplicate_runs = False,
                  keep = 'last',
                  exclude_direct_emissions=False):

        if isinstance(dir_num,int):
            dir_num = [dir_num]
        if isinstance(years,int):
            years = [years]
        
        if compute_hats:
            assert compute_sols, "can't compute hats without computing the solution"
        
        relevant_runs, found_cases, not_found_cases = find_runs(cases,results_path,dir_num,years
                                                                ,drop_duplicate_runs = drop_duplicate_runs,keep=keep)
        
        assert len(relevant_runs) > 0, "No runs found"
        
        if baselines is None:
            baselines = {}
            for y in years:
                baselines[int(y)] = d.baseline(y,data_path,exclude_direct_emissions)
        
        sols = []
        for idx,run in relevant_runs.iterrows():
            sols.append(sol(run, results_path, data_path))
          
        if compute_sols:
            sols = [s.compute_solution(baselines[s.run.year]) for s in tqdm(sols)]
            
        if compute_hats:
            sols = [s.compute_hat(baselines[s.run.year]) for s in tqdm(sols)]
            
        if return_not_found_cases:
            return sols, baselines, relevant_runs, found_cases, not_found_cases
        else:
            return sols, baselines, relevant_runs, found_cases
        
def find_runs(cases,results_path,dir_num,years,drop_duplicate_runs = False,keep = 'first'):
    if isinstance(dir_num,int):
        dir_num = [dir_num]
    if isinstance(years,int):
        years = [years]
    if isinstance(cases,dict):
        cases = [cases]
    
    for i,y in enumerate(years):
        if isinstance(y,int):
            years[i] = str(y)
    
    temp_dfs = [pd.read_csv(dir_path(results_path,y,d_num)+'/runs.csv' ,
                 index_col=0)
     for y,d_num in itertools.product(years,dir_num)]    
    
    for temp_df in temp_dfs:
        if 'pol_pay_tax' not in temp_df.columns:
            temp_df['pol_pay_tax'] = False
        if 'tax_scheme' not in temp_df.columns:
            temp_df['tax_scheme'] = 'consumer'
        if 'tau_factor' not in temp_df.columns:
            temp_df['tau_factor'] = 1
    
    runs = pd.concat(
        temp_dfs
        )
    
    not_found_cases = []
    found_cases = []
    relevant_runs = []
    for cas in cases:
        condition = look_for_cas_in_runs(cas,runs,results_path)
        if condition.any():
            found_cases.append(cas)
            relevant_runs.append(runs[condition])
        else:
            not_found_cases.append(cas)
    
    assert len(relevant_runs) > 0, "No relevant runs found"
    relevant_runs = pd.concat(relevant_runs)
    
    if drop_duplicate_runs:
        relevant_runs.drop_duplicates(['year', 'carb_cost', 'tax_type', 'taxed_countries','taxing_countries', 
                                       'taxed_sectors','fair_tax', 'pol_pay_tax', 'tax_scheme', 'sigma_path', 
                                       'eta_path', 'path_tax_scheme'],
                                      inplace=True,
                                      keep = keep)
    
    print(str(len(found_cases))+' cases found out of '+str(len(found_cases)+len(not_found_cases)))
    print('Found cases for '+str(len(runs.year.drop_duplicates()))+' years')
    
    found_years = relevant_runs.year.drop_duplicates().to_list()
    not_found_years = [y for y in years if int(y) not in found_years]
    if not_found_years is not None:
      print('Years not found :',not_found_years)  
    
    return relevant_runs, found_cases, not_found_cases
        
def look_for_cas_in_runs(cas,runs,results_path):
        
    if cas['specific_taxing'] is None:
        condition1 = pd.Series(np.isclose(runs['carb_cost'].fillna(1e12), cas['carb_cost']), index = runs.index)
        
        if cas['taxed_countries'] is None:
            condition2 = runs['taxed_countries'].isna()
        else:
            condition2 = (runs['taxed_countries'] == str(sorted(cas['taxed_countries'])))
              
        if cas['taxing_countries'] is None:
            condition3 = runs['taxing_countries'].isna()
        else:
            condition3 = (runs['taxing_countries'] == str(sorted(cas['taxing_countries'])))
        
        if cas['taxed_sectors'] is None:
            condition4 = runs['taxed_sectors'].isna()
        else:
            condition4 = (runs['taxed_sectors'] == str(sorted(cas['taxed_sectors'])))
        
        condition5 = (runs['fair_tax'] == cas['fair_tax'])
        
        condition8 = (runs['pol_pay_tax'] == cas['pol_pay_tax'])
        
        condition9 = (runs['tax_scheme'] == cas['tax_scheme'])
        
        condition10 = (runs['tau_factor'] == cas['tau_factor'])
        
        condition6 = (runs['eta_path'] == cas['eta_path'])
        
        condition7 = (runs['sigma_path'] == cas['sigma_path'])
        
        # condition = condition1 & condition2 & condition3 & condition4 & condition5 & condition6 & condition7
        condition = condition1 * condition2 * condition3 * condition4 * condition5 * condition6 * condition7 * condition8 * condition9 * condition10
        
    if cas['specific_taxing'] is not None:
        condition1 = pd.Series([False]*len(runs), index = runs.index)
        for i,run in runs.iterrows():
            if 'specific' in run['tax_type']:
                condition1.iloc[i] = all(np.isclose(cas['specific_taxing'].value.values,
                                    pd.read_csv(run.path_tax_scheme,index_col=[0,1,2]).value.values))
        
        condition2 = (runs['eta_path'] == cas['eta_path'])
        
        condition3 = (runs['sigma_path'] == cas['sigma_path'])        
        
        condition = condition1 & condition2 & condition3 
        
    return condition