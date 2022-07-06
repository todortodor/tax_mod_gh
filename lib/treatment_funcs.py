#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 08:39:51 2022

@author: simonl
"""

import pandas as pd
import numpy as np
import solver_funcs as s
import data_funcs as d
import sys
from copy import deepcopy
import itertools
from tqdm import tqdm

def dir_path(results_path,year,dir_num):
    if isinstance(year,int):
        year = str(year)
    return results_path+year+'_'+str(dir_num)

class sol:
    def __init__(self, run, results_path):
        self.run = run
        self.res = pd.read_csv(results_path+run.path)
        self.results_path = results_path
        
        if self.run.fair_tax:
            self.contrib = pd.read_csv(results_path+run.path[:-4]+'_contrib.csv',index_col=0)
        else:
            self.contrib = None

        if 'specific' in run.tax_type:
            self.params = d.params(eta =run.eta,
                                   sigma = run.sigma,
                                   specific_taxing = pd.read_csv(self.results_path+run.path_tax_scheme, 
                                                                 index_col=['row_country','row_sector','col_country']),
                                   fair_tax = run.fair_tax)
            
        if 'specific' not in run.tax_type:
            self.params = d.params(eta =run.eta,
                                   sigma = run.sigma,
                                   carb_cost = run.carb_cost,
                                   taxed_countries = run.taxed_countries,
                                   taxed_sectors = run.taxed_sectors,
                                   fair_tax = run.fair_tax)
            
    
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
    
    def compute_solution(self,baseline,inplace=False):
        
        if inplace:
            frame = self
        else:
            frame = self.copy()
            
        p = frame.params
        b = baseline.make_np_arrays().compute_shares_and_gammas()
        
        E_hat_sol = frame.res.output_hat.values.reshape(b.country_number, b.sector_number)
        p_hat_sol = frame.res.price_hat.values.reshape(b.country_number, b.sector_number)
        if frame.run.fair_tax:
            b.deficit_np = b.deficit_np + frame.contrib.values
            
        iot_hat_unit = s.iot_eq_unit(p_hat_sol, p, b) 
        cons_hat_unit = s.cons_eq_unit(p_hat_sol, p, b)       
        beta = np.einsum('itj->tj',b.cons_np) / np.einsum('itj->j',b.cons_np)

        taxed_price = p_hat_sol[:,:,None]*(1+p.carb_cost_np*b.co2_intensity_np[:,:,None])
        consumer_price_agg = np.einsum('itj,itj->tj'
                                  ,taxed_price**(1-p.sigma) 
                                  ,b.share_cons_o_np 
                                  ) ** (1/(1 - p.sigma))
        price_agg_no_pow = np.einsum('itj,itjs->tjs'
                                  ,taxed_price**(1-p.eta) 
                                  ,b.share_cs_o_np 
                                  )       
        producer_price_agg = np.divide(1, 
                        price_agg_no_pow , 
                        out = np.ones_like(price_agg_no_pow), 
                        where = price_agg_no_pow!=0 ) ** (1/(p.eta - 1))  
        
        I_hat_sol = s.compute_I_hat(p_hat_sol, E_hat_sol, p, b)
        
        iot = np.einsum('it,js,itjs,itjs -> itjs', p_hat_sol, E_hat_sol , iot_hat_unit , b.iot_np)
        cons = np.einsum('it,j,itj,itj -> itj', p_hat_sol, I_hat_sol , cons_hat_unit , b.cons_np)
        va = E_hat_sol * b.va_np
        output = E_hat_sol * b.output_np
        co2_prod = E_hat_sol * b.co2_prod_np / p_hat_sol
        
        cons_hat_sol = np.einsum('j,itj->itj',  I_hat_sol , cons_hat_unit)
        utility_cs_hat_sol = np.einsum('itj,itj->tj', 
                                        cons_hat_sol**((p.sigma-1)/p.sigma) , 
                                        b.share_cons_o_np ) ** (p.sigma / (p.sigma-1))
        utility = (utility_cs_hat_sol**beta).prod(axis=0)
        
        # for var in ['iot','cons','va','output','co2_prod']:
        #     setattr(frame,var,pd.DataFrame(index = getattr(baseline,var).index, 
        #                                     data = locals()[var].ravel(),
        #                                     columns = ['value']))
        # for var in ['iot','cons','va','output','co2_prod']:
        #     setattr(frame,var,getattr(baseline,var))
        #     getattr(frame,var).value = locals()[var].ravel()
        frame.iot = b.iot[['row_country','row_sector','col_country','col_sector']]
        frame.iot['value'] = iot.ravel()
        
        frame.cons = b.cons[['row_country','row_sector','col_country']]
        frame.cons['value'] = cons.ravel()
        
        frame.va = b.va[['col_country','col_sector']]
        frame.va['value'] = va.ravel()
        
        frame.output = b.output[['country','sector']]
        frame.output['value'] = output.ravel()
        
        frame.co2_prod = b.co2_prod[['country','sector']]
        frame.co2_prod['value'] = co2_prod.ravel()
            
        frame.price = pd.DataFrame(index = pd.MultiIndex.from_product(
                                                        [b.country_list,b.sector_list],
                                                        names = ['row_country','row_sector']),
                                    data = p_hat_sol.ravel(),
                                    columns = ['hat']).reset_index()
        frame.taxed_price = pd.DataFrame(index = pd.MultiIndex.from_product(
                                                        [b.country_list,b.sector_list,b.country_list],
                                                        names = ['row_country','row_sector','col_country']),
                                    data = taxed_price.ravel(),
                                    columns = ['hat']).reset_index()
        frame.consumer_price_agg = pd.DataFrame(index = pd.MultiIndex.from_product(
                                                        [b.sector_list,b.country_list],
                                                        names = ['row_sector','col_country']), 
                                                data = consumer_price_agg.ravel(),
                                                columns = ['hat']).reset_index()
        frame.producer_price_agg = pd.DataFrame(index = pd.MultiIndex.from_product(
                                                        [b.sector_list,b.country_list,b.sector_list],
                                                        names = ['row_sector','col_country','col_sector']), 
                                                data = producer_price_agg.ravel(),
                                                columns = ['hat']).reset_index()
        frame.utility = pd.DataFrame(index = pd.Index(b.country_list,name='country'), 
                                    data = utility.ravel(),
                                    columns = ['hat']).reset_index()
        
        return frame
    
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
                  keep = 'first'):

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
                baselines[int(y)] = d.baseline(y,data_path)
        
        sols = []
        for idx,run in relevant_runs.iterrows():
            sols.append(sol(run, results_path))
          
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
        
    runs = pd.concat(
        # [pd.read_csv(dir_path(results_path,y,d_num)+'/runs.csv' ,
        #              index_col=0, 
        #              keep_default_na=False,
        #              ).replace('',None)
        [pd.read_csv(dir_path(results_path,y,d_num)+'/runs.csv' ,
                     index_col=0)
         for y,d_num in itertools.product(years,dir_num)]
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
                                       'taxed_sectors','fair_tax', 'sigma', 'eta', 'path_tax_scheme'],
                                      inplace=True,
                                      keep = keep)
    
    print(str(len(found_cases))+' cases found out of '+str(len(found_cases)+len(not_found_cases)))
    
    return relevant_runs, found_cases, not_found_cases
        
def look_for_cas_in_runs(cas,runs,results_path):
        
    if cas['specific_taxing'] is None:
        condition1 = pd.Series(np.isclose(runs['carb_cost'].fillna(1e12), cas['carb_cost']))
        
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
        
        condition = condition1 & condition2 & condition3 & condition4 & condition5
        
    if cas['specific_taxing'] is not None:
        condition = pd.Series([False]*len(runs))
        for i,run in runs.iterrows():
            if 'specific' in run['tax_type']:
                condition.iloc[i] = all(np.isclose(cas['specific_taxing'].value.values,
                                    pd.read_csv(results_path+run.path_tax_scheme,index_col=[0,1,2]).value.values))
    return condition