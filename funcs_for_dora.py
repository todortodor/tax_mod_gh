#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:20:44 2024

@author: slepot
"""

from copy import deepcopy
import pandas as pd
import numpy as np
import os, sys
from pathlib import Path
import itertools
import matplotlib.pyplot as plt

def get_country_list():
    country_list = ['ARG', 'AUS', 'AUT', 'BEL', 'BGR', 'BRA', 'BRN', 'CAN', 
                    'CHE', 'CHL', 'CHN', 'COL', 'CRI', 'CYP', 'CZE', 'DEU', 
                    'DNK', 'ESP', 'EST', 'FIN', 'FRA', 'GBR', 'GRC', 'HRV', 
                    'HUN', 'IDN', 'IND', 'IRL', 'ISL', 'ISR', 'ITA', 'JPN', 
                    'KAZ', 'KHM', 'KOR', 'LAO', 'LTU', 'LVA', 'MAR', 'MEX', 
                    'MLT', 'MMR', 'NLD', 'NOR', 'NZL', 'PER', 'PHL', 'POL', 
                    'PRT', 'ROU', 'ROW', 'RUS', 'SAU', 'SGP', 'SVK', 'SVN', 
                    'SWE', 'THA', 'TUN', 'TUR', 'TWN', 'USA', 'VNM', 'ZAF']
    return country_list

def get_sector_list():
    sector_list = ['01T02', '03', '05T06', '07T08', '10T12', '13T15', '16', 
                   '17T18', '19', '20', '21', '22', '23', '24', '25', '26', 
                   '27', '28', '29T30', '31T33', '35', '36T39', '41T43', 
                   '45T47', '49', '50', '51', '52', '53', '55T56', 
                   '58T60', '61', '62T63', '64T66', '68', '69T75', 
                   '77T82', '84', '85', '86T88', '90T93', '94T98']
    return sector_list

def countries_from_fta(fta = None, return_dict = False):
    fta_dict = {
     'EU':['AUT', 'BEL', 'BGR','CYP', 'CZE', 'DEU','DNK', 'ESP', 'EST', 'FIN',
           'FRA','GRC','HUN','IRL','ITA','LTU','LVA','MLT','NLD',
           'POL','PRT','ROU','SVK','SVN','SWE'],
     'EU_ETS':['AUT', 'BEL', 'BGR', 'CHE', 'CYP', 
                'CZE', 'DEU', 'DNK', 'ESP', 'EST', 
                'FIN', 'FRA', 'GRC', 'HRV', 'HUN', 
                'IRL', 'ISL', 'ITA', 'LTU', 'LVA', 
                'MLT', 'NLD', 'NOR', 'POL', 'PRT', 
                'ROU', 'SVK', 'SVN', 'SWE'
                ],
     'NAFTA':['CAN','MEX','USA'],
     'ASEAN':['BRN', 'IDN', 'KHM', 'LAO', 'MMR', 'PHL', 'SGP', 'THA', 'VNM'],
     'AANZFTA':['AUS', 'BRN', 'IDN', 'KHM', 'LAO', 'MMR', 'NZL', 'PHL', 'SGP', 'THA', 'VNM'],
     'APTA':['CHN','IND','KOR','LAO'],
     'EEA':['AUT', 'BEL', 'BGR', 'CHE', 'CYP', 'CZE', 'DEU','DNK', 'ESP', 'EST', 'FIN',
            'FRA','GRC','HRV','HUN','IRL','ISL','ITA','LTU','LVA','MLT','NLD',
            'NOR','POL','PRT','ROU','SVK','SVN','SWE'],
     'MERCOSUR':['ARG','BRA','CHL','COL','PER']}
    
    fta_dict['G20'] = fta_dict['EU'] + [c for c in ['ARG','AUS','BRA','CAN','CHN', 
             'FRA','DEU','IND','IDN','ITA','JPN','MEX','RUS','SAU','ZAF','KOR',  
             'TUR','GBR','USA'] if c not in fta_dict['EU']]
    
    if fta is None:
        if not return_dict:
            for key,item in fta_dict.items():
                print(key)
                print(item)
        if return_dict:
            return fta_dict
    else:
        return fta_dict[fta]

class params:
    """ Contains the tax matrix and the elasticities
    """
    country_list = get_country_list()
    country_number = len(country_list)
    
    sector_list = get_sector_list()
    sector_number = len(sector_list)
    
    def __init__(self,
                 data_path,
                 eta_path, 
                 sigma_path,
                 tax_matrix):
        
        self.eta_path = eta_path
        eta_df = pd.read_csv(data_path+'elasticities/'+eta_path,index_col=0)
        assert eta_df.index.to_list() == self.sector_list , "wrong sectors for eta"
        assert len(eta_df.columns) == 1, "wrong nbr of columns of eta"
        assert np.all(eta_df.values > 0), "negative or null eta"
        self.eta = eta_df[eta_df.columns[0]].values
            
        self.sigma_path = sigma_path
        sigma_df = pd.read_csv(data_path+'elasticities/'+sigma_path,index_col=0)
        assert sigma_df.index.to_list() == self.sector_list , "wrong sectors for sigma"
        assert len(sigma_df.columns) == 1, "wrong nbr of columns of sigma"
        assert np.all(sigma_df.values > 0), "negative or null sigma"
        self.sigma = sigma_df[sigma_df.columns[0]].values
            
        self.tax_matrix = tax_matrix
        self.tax_matrix_np = self.tax_matrix.value.values.reshape((self.country_number,self.sector_number,self.country_number))
        self.num_scaled = False
        
    def copy(self):
        frame = deepcopy(self)
        return frame
    
    def elements(self):
        for key, item in sorted(self.__dict__.items()):
            print(key, ',', str(type(item))[8:-2])
            
class baseline:
    """ Contains the data
    """
    def __init__(self,year,data_path,exclude_direct_emissions=False):
        year = str(year)
        print('Loading baseline data '+year)
        self.path = data_path+'yearly_CSV_agg_treated/datas'+year
        
        cons = pd.read_csv (self.path+'/consumption_'+year+'.csv'
                            ,index_col = ['row_country','row_sector','col_country'])
        iot = pd.read_csv (self.path+'/input_output_'+year+'.csv'
                           ,index_col = ['row_country','row_sector','col_country','col_sector'])
        output = pd.read_csv (self.path+'/output_'+year+'.csv'
                              ,index_col = ['row_country','row_sector'])
        va = pd.read_csv (self.path+'/VA_'+year+'.csv'
                          ,index_col = ['col_country','col_sector'])
        if not exclude_direct_emissions:
            co2_intensity = pd.read_csv(self.path+'/co2_intensity_prod_with_agri_ind_proc_fug_'+year+'.csv'
                                        ,index_col = ['country','sector'])
            co2_prod = pd.read_csv(self.path+'/prod_CO2_with_agri_agri_ind_proc_fug_'+year+'.csv'
                                   ,index_col = ['country','sector'])
        else:
            co2_intensity = pd.read_csv(self.path+'/co2_intensity_prod_'+year+'.csv'
                                        ,index_col = ['country','sector'])
            co2_prod = pd.read_csv(self.path+'/prod_CO2_'+year+'.csv'
                                   ,index_col = ['country','sector'])
        labor = pd.read_csv(data_path+'/World bank/labor_force/labor.csv')
        
        cumul_emissions_share = pd.read_csv(data_path+'share_of_cumulative_co2_treated.csv',
                                            index_col = [0,1]
                                            ).loc[int(year)]
        
        self.sector_list = get_sector_list()
        self.sector_number = len(self.sector_list)
        self.country_list = get_country_list()
        self.country_number = len(self.country_list)
        self.iot = iot
        self.cons = cons
        self.output = output.rename_axis(['country','sector'])
        self.va = va
        self.co2_intensity = co2_intensity
        self.co2_prod = co2_prod
        self.labor = labor
        self.cumul_emissions_share = cumul_emissions_share
        self.deficit = pd.DataFrame(self.cons.groupby(level=2)['value'].sum()
            - self.va.groupby(level=0)['value'].sum())
        self.num_scaled = False
        self.year = year
        
        
    def elements(self):
        for key, item in sorted(self.__dict__.items()):
            print(key, ',', str(type(item))[8:-2])
    
    def get_elements(self):
        elements_list = []
        for key, item in sorted(self.__dict__.items()):
            elements_list.append(key)
        return elements_list
    
    def copy(self):
        frame = deepcopy(self)
        return frame
    
    def memory(self, details = False):
        print('Baseline class takes up ', 
              sum([sys.getsizeof(x) for x in self.__dict__.values()])/1e6, 
              ' Mb')
        if details:
            for key,item in sorted(self.__dict__.items()):
                print(key, sys.getsizeof(item)/1e6, 'Mb')            
    
    def num_scale(self, 
                  numeraire_type = 'wage', 
                  numeraire_country = 'USA', 
                  inplace = False):
        if inplace:
            frame = self
        else:
            frame = self.copy()
        
        if frame.num_scaled == False:
            if numeraire_type == 'output':
                if numeraire_country == 'WLD':
                    num = frame.output.value.sum()
                else:
                    num = frame.output.loc[numeraire_country].value.sum()
            if numeraire_type == 'wage':
                if numeraire_country == 'WLD':
                    num = frame.va.value.sum()
                else:
                    num = frame.va.loc[numeraire_country].value.sum() \
                        / frame.labor.loc[frame.labor.country == numeraire_country, frame.year].to_numpy()[0]
            
            frame.cons.value = frame.cons.value / num
            frame.iot.value = frame.iot.value / num
            frame.output.value = frame.output.value / num
            frame.va.value = frame.va.value / num
            frame.deficit.value = frame.deficit.value / num
            frame.co2_intensity.value = frame.co2_intensity.value * num
            frame.num_scaled = True
            frame.num = num
            frame.num_type = numeraire_type
            frame.num_country = numeraire_country
            if not inplace:
                return frame
            
        else:
            print('Baseline was already scaled by numeraire, did nothing')
    
    def num_scale_back(self, inplace = False):
        if inplace:
            frame = self
        else:
            frame = self.copy()
        
        if frame.num_scaled == True:
            
            num = frame.num
            
            frame.cons = frame.cons * num
            frame.iot = frame.iot * num
            frame.output = frame.output * num
            frame.va = frame.va * num
            frame.deficit = frame.deficit * num
            frame.co2_intensity = frame.co2_intensity / num
            frame.num_scaled = False
            del frame.num, frame.num_type, frame.num_country
            if not inplace:
                return frame
            
        else:
            print('Baseline was not scaled by numeraire, did nothing')
    
    def make_np_arrays(self,inplace=False):
        if inplace:
            frame = self
        else:
            frame = self.copy()   
            
        C = frame.country_number
        S = frame.sector_number
        frame.cons_np = frame.cons.value.values.reshape(C,S,C)
        frame.iot_np = frame.iot.value.values.reshape(C,S,C,S)
        frame.output_np = frame.output.value.values.reshape(C,S)
        frame.co2_intensity_np = frame.co2_intensity.value.values.reshape(C,S) 
        frame.co2_prod_np = frame.co2_prod.value.values.reshape(C,S)
        frame.va_np = frame.va.value.values.reshape(C,S)
        frame.deficit_np = frame.deficit.value.values
        frame.cumul_emissions_share_np = frame.cumul_emissions_share.value.values
        
        return frame
        
    def compute_shares_and_gammas(self,inplace = False):
        if inplace:
            frame = self
        else:
            frame = self.copy()  
            
        frame.gamma_labor_np = frame.va_np / frame.output_np
        frame.gamma_sector_np = frame.iot_np.sum(axis = 0) / frame.output_np   
        frame.cons_tot_np = frame.cons_np.sum(axis=(0,1))
        with np.errstate(invalid='ignore'):
            frame.share_cs_o_np = np.nan_to_num(frame.iot_np / frame.iot_np.sum(axis = 0)[None,:,:,:])
            frame.share_cons_o_np = np.nan_to_num(frame.cons_np / frame.cons_np.sum(axis = 0)[None,:,:])
        frame.va_share_np = frame.va_np / frame.va_np.sum(axis=1)[:,None]   
        
        return frame

def cons_eq_unit(price, params, baseline):    
    p = params
    b = baseline

    taxed_price = np.einsum('it,itj->itj',
                            price,
                            (1+p.tax_matrix_np))
        
    price_agg_no_pow = np.einsum('itj,itj->tj'
                          ,taxed_price**(1-p.sigma[None,:,None]) 
                          ,b.share_cons_o_np 
                          )
    
    Q = np.einsum('tj,itj -> itj' , 
                  np.divide(1, 
                            price_agg_no_pow , 
                            out = np.ones_like(price_agg_no_pow), 
                            where = price_agg_no_pow!=0 ) ,  
                  taxed_price ** (-p.sigma[None,:,None]))
    return Q   

def iot_eq_unit(price, params, baseline):    
    p = params
    b = baseline
    
    taxed_price = np.einsum('it,itj->itj',
                            price,
                            (1+p.tax_matrix_np))
        
    price_agg_no_pow = np.einsum('itj,itjs->tjs'
                          ,taxed_price**(1-p.eta[None,:,None]) 
                          ,b.share_cs_o_np 
                          )
    
    M = np.einsum('tjs,itj -> itjs' , 
                  np.divide(1, 
                            price_agg_no_pow , 
                            out = np.ones_like(price_agg_no_pow), 
                            where = price_agg_no_pow!=0 ) , 
                  taxed_price ** (-p.eta[None,:,None]))
    return M 

def solve_one_loop(params, baseline, vec_init = None, tol=1e-9, damping=5):
    C = baseline.country_number
    S = baseline.sector_number
    p = params
    b = baseline  

    tol = tol

    count = 0
    condition = True

    vec_new = None
    p_new = None
    E_new = None
    I_new = None
    if vec_init is None:
        E_old = np.ones(C*S).reshape(C,S)
        p_old = np.ones(C*S).reshape(C,S)
        I_old = np.ones(C).reshape(C)
    else:
        E_old = vec_init[:C*S].reshape(C,S)
        p_old = vec_init[C*S:2*C*S].reshape(C,S)
        I_old = vec_init[2*C*S:].reshape(C)

    convergence = None

    while condition:
        if count>0:
            vec_new = np.concatenate([E_new.ravel(),p_new.ravel(),I_new.ravel()])
            vec_old = np.concatenate([E_old.ravel(),p_old.ravel(),I_old.ravel()])
            
            vec_new[vec_new<0]=0
            vec_old = (vec_new + (damping-1)*vec_old)/damping
            
            E_old = vec_old[:C*S].reshape(C,S)
            p_old = vec_old[C*S:2*C*S].reshape(C,S)
            I_old = vec_old[2*C*S:].reshape(C)
        
        iot_hat_unit = iot_eq_unit(p_old, params, baseline) 
        cons_hat_unit = cons_eq_unit(p_old, params, baseline)    
        
        # E hat equation
        A = np.einsum('j,it,itj,itj->it',
                      I_old,
                      p_old,
                      cons_hat_unit,
                      b.cons_np)
        
        B = np.einsum('js,it,itjs,itjs->it',
                      E_old,
                      p_old,
                      iot_hat_unit,
                      b.iot_np)
        
        E_new = (A+B)/b.output_np
        
        # I hat equation
        A = np.einsum('js,js->j',
                      E_new,
                      b.va_np)
        
        B = np.einsum('j,it,itj,itj,itj->j',
                      I_old,
                      p_old,
                      p.tax_matrix_np,
                      cons_hat_unit,
                      b.cons_np)
        
        K = np.einsum('js,it,itj,itjs,itjs->j',
                      E_old,
                      p_old,
                      p.tax_matrix_np,
                      iot_hat_unit,
                      b.iot_np)
        
        I_new = (A+B+K+b.deficit_np) / b.cons_tot_np
        
        # p hat equation
        taxed_price = np.einsum('it,itj->itj',
                                p_old,
                                (1+p.tax_matrix_np))
                
        price_agg_no_pow = np.einsum('itj,itjs->tjs'
                                  ,taxed_price**(1-p.eta[None,:,None]) 
                                  ,b.share_cs_o_np 
                                  )       
        price_agg = np.divide(1, 
                        price_agg_no_pow , 
                        out = np.ones_like(price_agg_no_pow), 
                        where = price_agg_no_pow!=0 ) ** (1/(p.eta[:,None,None] - 1))
  
        prod = ( price_agg ** b.gamma_sector_np ).prod(axis = 0)
        wage_hat = np.einsum('js,js->j', E_old , b.va_share_np )    
        
        p_new = wage_hat[:,None]**b.gamma_labor_np * prod
        
        # compute convergence
        vec_new = np.concatenate([E_new.ravel(),I_new.ravel(),p_new.ravel()])
        vec_old = np.concatenate([E_old.ravel(),I_old.ravel(),p_old.ravel()])
        convergence = np.append(convergence , 
                                np.linalg.norm(vec_new - vec_old)/np.linalg.norm(vec_old) )
        
        condition = convergence[-1] > tol
        count += 1

        if count % 100 == 0 and count>0:
            plt.semilogy(convergence)
            plt.show()
            print(count,convergence[-1])
    
    norm_factor = np.einsum('js,js,j->', 
                            E_new , 
                            baseline.va_share_np, 
                            baseline.va_np.sum(axis=1)
                            ) / baseline.va_np.sum()
    
    E_new = E_new / norm_factor
    p_new = p_new / norm_factor
    I_new = I_new / norm_factor
    
    results = {'E_hat': E_new,'p_hat':p_new,'I_hat':I_new}
    return results

def write_solution_csv(results,
                       results_path, 
                       run_name,
                       params):
    
    p = params
    
    E_hat_sol = results['E_hat']
    p_hat_sol = results['p_hat']
    I_hat_sol = results['I_hat']
    
    path = results_path+run_name
    
    results_data_frame = pd.DataFrame(index = pd.MultiIndex.from_product([p.country_list, p.sector_list]
                                                                         ,names=['country','sector']),
                                      columns = ['output_hat','price_hat'])
    results_data_frame['output_hat'] = E_hat_sol.ravel()
    results_data_frame['price_hat'] = p_hat_sol.ravel()
    results_data_frame['spending_hat'] = np.repeat(I_hat_sol[:, np.newaxis], 
                                                   p.sector_number, axis=1).ravel()
    results_data_frame.to_csv(path+'_results.csv')
    
    params.tax_matrix.to_csv(path+'_tax_matrix.csv')
    
class sol:
    def __init__(self, results_path, tax_matrix_path, baseline, data_path,
                 eta_path, sigma_path):
        
        self.res = pd.read_csv(results_path)
        self.tax_matrix = pd.read_csv(tax_matrix_path)
        self.results_path = results_path
        
        self.params = params(data_path,
                            eta_path = eta_path,
                            sigma_path = sigma_path,
                            tax_matrix = self.tax_matrix)
            
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
    
    def compute_solution(self,baseline,inplace=True,with_np_arrays=False):
        if inplace:
            frame = self
        else:
            frame = self.copy()
            
        p = frame.params
        b = baseline.make_np_arrays().compute_shares_and_gammas()
        
        E_hat_sol = frame.res.output_hat.values.reshape(b.country_number, b.sector_number)
        p_hat_sol = frame.res.price_hat.values.reshape(b.country_number, b.sector_number)
        I_hat_sol = frame.res.spending_hat.values.reshape(b.country_number, b.sector_number)[:,0]
            
        iot_hat_unit = iot_eq_unit(p_hat_sol, p, b) 
        cons_hat_unit = cons_eq_unit(p_hat_sol, p, b)     
        
        beta = np.einsum('itj->tj',b.cons_np) / np.einsum('itj->j',b.cons_np)

        taxed_price = p_hat_sol[:,:,None]*(1+p.tax_matrix_np)
        
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
        
        iot = np.einsum('it,js,itjs,itjs -> itjs', p_hat_sol, E_hat_sol , iot_hat_unit , b.iot_np)
        cons = np.einsum('it,j,itj,itj -> itj', p_hat_sol, I_hat_sol , cons_hat_unit , b.cons_np)
        
        va = E_hat_sol * b.va_np
        output = E_hat_sol * b.output_np
        co2_prod = E_hat_sol * b.co2_prod_np / p_hat_sol

        labor_hat = E_hat_sol / (va.sum(axis=1)/b.va_np.sum(axis=1))[:,None]
        
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
        
        frame.labor_hat = pd.DataFrame(index = b.output.index,
                                    data = labor_hat.ravel(),
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