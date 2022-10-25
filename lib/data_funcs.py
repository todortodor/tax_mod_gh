#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 16:08:02 2022

@author: simonl
"""

from copy import deepcopy
import pandas as pd
import numpy as np
import os, sys
from pathlib import Path
import itertools

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
     'NAFTA':['CAN','MEX','USA'],
     'ASEAN':['BRN', 'IDN', 'KHM', 'LAO', 'MMR', 'PHL', 'SGP', 'THA', 'VNM'],
     'AANZFTA':['AUS', 'BRN', 'IDN', 'KHM', 'LAO', 'MMR', 'NZL', 'PHL', 'SGP', 'THA', 'VNM'],
     'APTA':['CHN','IND','KOR','LAO'],
     'EEA':['AUT', 'BEL', 'BGR', 'CHE', 'CYP', 'CZE', 'DEU','DNK', 'ESP', 'EST', 'FIN',
            'FRA','GRC','HRV','HUN','IRL','ISL','ITA','LTU','LVA','MLT','NLD',
            'NOR','POL','PRT','ROU','SVK','SVN','SWE'],
     'MERCOSUR':['ARG','BRA','CHL','COL','PER']}
    if fta is None:
        if not return_dict:
            for key,item in fta_dict.items():
                print(key)
                print(item)
        if return_dict:
            return fta_dict
    else:
        return fta_dict[fta]

def fta_from_countries(countries):
    try:
        countries = eval(countries)
    except:
        pass
    
    if type(countries) is not list:
        return countries
    
    else:
        fta_dict = countries_from_fta(return_dict=True)
        fta_dict = dict(sorted(fta_dict.items(), key= lambda x: len(x[1]), reverse=True))
        
        out_countries = set(countries)
        out_ftas = []
        for fta in fta_dict:
            if set(fta_dict[fta]).issubset(set(countries)):
                if not any([set(fta_dict[fta]).issubset(set(fta_dict[f])) for f in out_ftas]):
                    out_ftas.append(fta)
                    out_countries = out_countries-set(fta_dict[fta])
        
        return sorted(out_ftas)+sorted(list(out_countries))

class baseline:
    def __init__(self,year,data_path):
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
        co2_intensity = pd.read_csv(self.path+'/co2_intensity_prod_with_agri_ind_proc_fug_'+year+'.csv'
                                    ,index_col = ['country','sector'])
        co2_prod = pd.read_csv(self.path+'/prod_CO2_with_agri_agri_ind_proc_fug_'+year+'.csv'
                               ,index_col = ['country','sector'])
        labor = pd.read_csv(data_path+'/World bank/labor_force/labor.csv')
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
                num = frame.output.loc[numeraire_country].value.sum()
            if numeraire_type == 'wage':
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
 
class params:
    """ Builds parameters and taxation scheme
    
    Default behavior : uniform tax for all countries and sectors.
    
    Taxed countries or sectors need to be a list of strings found in 
    baseline.country_list or baseline.sector_list
    If taxed_countries or taxed_sectors are specified, the tax will
    only apply to those countries and/or sectors and will be the same.
    
    Can also apply specific_taxing with variable tax values depending on countries / sectors.
    The input needs to be a DataFrame with one column containing the tax value
    and a multi-index country/sector.
    """
    country_list = get_country_list()
    country_number = len(country_list)
    
    sector_list = get_sector_list()
    sector_number = len(sector_list)
    
    def __init__(self,
                 data_path,
                 eta_path, 
                 sigma_path,
                 carb_cost = None,  
                 taxed_countries = None, 
                 taxing_countries = None, 
                 taxed_sectors = None,
                 specific_taxing = None,
                 fair_tax = False):
        
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
            
        self.taxed_sectors = taxed_sectors
        self.taxed_countries = taxed_countries
        self.taxing_countries = taxing_countries
        self.specific_taxing = specific_taxing
        self.fair_tax = fair_tax
        self.carb_cost = carb_cost
        
        if specific_taxing is None:
            self.carb_cost_df = pd.DataFrame(
                index = pd.MultiIndex.from_product(
                    [self.country_list,self.sector_list,self.country_list], 
                    names = ['row_country','row_sector','col_country']),
                columns = ['value'],
                data = np.ones(self.country_number * self.sector_number * self.country_number)*carb_cost
                )
            tax_type = 'uniform'
            
            # if taxed_countries is not None:
            #     non_taxed_countries = [c for c in self.country_list if c not in taxed_countries]
            #     self.carb_cost_df.loc[
            #         (non_taxed_countries,self.sector_list),'value'
            #         ] = 0
            #     tax_type = tax_type+'_countries'
            try:
                non_taxed_countries = [c for c in self.country_list if c not in taxed_countries]
                self.carb_cost_df.loc[
                    (non_taxed_countries,self.sector_list,self.country_list),'value'
                    ] = 0
                tax_type = tax_type+'_taxed_countries'
            except:
                pass
            
            try:
                non_taxing_countries = [c for c in self.country_list if c not in taxing_countries]
                self.carb_cost_df.loc[
                    (self.country_list,self.sector_list,non_taxing_countries),'value'
                    ] = 0
                tax_type = tax_type+'_taxing_countries'
            except:
                pass
                        
            # if taxed_sectors is not None:
            #     non_taxed_sectors = [s for s in self.sector_list if s not in taxed_sectors]
            #     self.carb_cost_df.loc[
            #         (self.country_list,non_taxed_sectors),'value'
            #         ] = 0
            #     tax_type = tax_type+'_sectors'
            try:
                non_taxed_sectors = [s for s in self.sector_list if s not in taxed_sectors]
                self.carb_cost_df.loc[
                    (self.country_list,non_taxed_sectors,self.country_list),'value'
                    ] = 0
                tax_type = tax_type+'_sectors'
            except:
                pass
        
        if specific_taxing is not None:
            self.carb_cost = None
            self.taxed_countries = None
            self.taxing_countries = None
            self.taxed_sectors = None
            self.carb_cost_df = specific_taxing
            self.carb_cost_df.sort_index(inplace = True)
            self.carb_cost_df.index.rename(['row_country','row_sector','col_country'], inplace = True)
            assert np.all( self.carb_cost_df.index == pd.MultiIndex.from_product(
                [self.country_list,
                self.sector_list,
                self.country_list]) ), "Incorrect taxing input, index isn't correct"
            
            assert len(self.carb_cost_df.columns) == 1, "Incorrect taxing input, wrong nbr of columns"
            self.carb_cost_df.columns = ['value']
            tax_type = 'specific'
            
        if fair_tax:
            tax_type = tax_type+'_fair'

        self.carb_cost_np = self.carb_cost_df.value.values.reshape(self.country_number,
                                                                   self.sector_number,
                                                                   self.country_number)
        self.num_scaled = False
        self.tax_type = tax_type
    
    def copy(self):
        frame = deepcopy(self)
        return frame
    
    def elements(self):
        for key, item in sorted(self.__dict__.items()):
            print(key, ',', str(type(item))[8:-2])
    
    def num_scale_carb_cost(self, num, inplace = False):
        if inplace:
            frame = self
        else:
            frame = self.copy()
        
        if frame.num_scaled == False:
            
            try:
                frame.carb_cost = frame.carb_cost / num
            except:
                pass
            frame.carb_cost_np = frame.carb_cost_np / num
            frame.carb_cost_df = frame.carb_cost_df / num
            frame.num = num
            frame.num_scaled = True
            if not inplace:
                return frame
            
        else:
            print('Carbon tax was already scaled by numeraire, did nothing')
            
    def num_scale_back_carb_cost(self, inplace = False):
        if inplace:
            frame = self
        else:
            frame = self.copy()
        
        if frame.num_scaled == True:
            num = frame.num    
            try:
                frame.carb_cost = frame.carb_cost * num
            except:
                pass
            frame.carb_cost_np = frame.carb_cost_np * num
            frame.carb_cost_df = frame.carb_cost_df * num
            frame.num_scaled = False
            del frame.num
            if not inplace:
                return frame
            
        else:
            print('Carbon tax was not scaled by numeraire, did nothing')
    

def build_cases(eta_path_list,sigma_path_list,carb_cost_list,taxed_countries_list, taxing_countries_list,
                taxed_sectors_list, specific_taxing_list,fair_tax_list):
        
    cases = []
    taxed_countries_list = [sorted(t) if t is not None else None for t in taxed_countries_list]
    taxing_countries_list = [sorted(t) if t is not None else None for t in taxing_countries_list]
    taxed_sectors_list = [sorted(t) if t is not None else None for t in taxed_sectors_list]
    if isinstance(eta_path_list,str):
        eta_path_list = [eta_path_list]
    if isinstance(sigma_path_list,str):
        sigma_path_list = [sigma_path_list]
    
    for eta_path,sigma_path,carb_cost,taxed_countries,taxing_countries,taxed_sectors,specific_taxing,fair_tax \
        in itertools.product(eta_path_list,sigma_path_list,carb_cost_list,taxed_countries_list,taxing_countries_list,
                             taxed_sectors_list,specific_taxing_list,fair_tax_list): 
            assert carb_cost is None or specific_taxing is None, 'carb_cost and specific taxing are mutually exclusive parameters'
            assert carb_cost is not None or specific_taxing is not None, 'carb_cost or specific taxing need to be specified'
            
            cases.append({'eta_path':eta_path,
                      'sigma_path':sigma_path,
                      'carb_cost':carb_cost,
                      'taxed_countries': taxed_countries,
                      'taxing_countries': taxing_countries,
                      'taxed_sectors':taxed_sectors,
                      'specific_taxing':specific_taxing,
                      'fair_tax':fair_tax})
    return cases
            
def write_solution_csv(results,
                       results_path, 
                       dir_num,
                       emissions_sol, 
                       utility,
                       params, 
                       baseline):
    p = params
    b = baseline    
    
    E_hat_sol = results['E_hat']
    p_hat_sol = results['p_hat']
    
    path = results_path+b.year+'_'+str(dir_num)
    Path(path).mkdir(parents=True, exist_ok=True)
    
    runs_path = path+'/runs.csv'
    
    if not os.path.exists(runs_path):
        runs = pd.DataFrame(columns = ['year',
                                        'carb_cost',
                                        'tax_type',
                                        'taxed_countries',
                                        'taxing_countries',
                                        'taxed_sectors',
                                        'fair_tax',
                                        'sigma_path',
                                        'eta_path',
                                        'path',
                                        'path_tax_scheme',
                                        'num',
                                        'num_type',
                                        'num_country',
                                        'emissions',
                                        'utility'])
        runs.to_csv(runs_path)
            
    runs = pd.read_csv(runs_path,index_col=0)
    
    files_in_dir = os.listdir(path)
    highest_existing_run_number = max(
                                      [int(f.split('.')[0].split('_')[0]) 
                                      for f in files_in_dir 
                                      if f.split('.')[0].split('_')[0].isnumeric()]
                                      +[0])
    run_path = b.year+'_'+str(dir_num)+'/'+str(highest_existing_run_number+1)+'_results.csv'
        
    if 'specific' in p.tax_type:       
        tax_scheme_path = results_path+run_path[:-4]+'_tax_scheme.csv'
        p.num_scale_back_carb_cost().carb_cost_df.to_csv(tax_scheme_path)
    else:
        tax_scheme_path = None
    
    if p.fair_tax:
        contrib_data_frame = pd.DataFrame(index = pd.Index(b.country_list, name='country') , 
                                          columns = ['value'], 
                                          data = results['contrib']*b.num)
        contrib_data_frame.to_csv(results_path+run_path[:-4]+'_contrib.csv')
    
    run = pd.DataFrame(data = [b.year,
                    p.num_scale_back_carb_cost().carb_cost,
                    p.tax_type,
                    p.taxed_countries,
                    p.taxing_countries,
                    p.taxed_sectors,
                    p.fair_tax,
                    p.sigma_path,
                    p.eta_path,
                    run_path,
                    tax_scheme_path,
                    b.num,
                    b.num_type,
                    b.num_country,
                    emissions_sol,
                    utility], 
                    index = runs.columns).T
    runs = pd.concat([runs, run],ignore_index=True)
    runs.to_csv(runs_path)
    
    results_data_frame = pd.DataFrame(index = pd.MultiIndex.from_product([b.country_list, b.sector_list]
                                                                         ,names=['country','sector']),
                                      columns = ['output_hat','price_hat'])
    results_data_frame['output_hat'] = E_hat_sol.ravel()
    results_data_frame['price_hat'] = p_hat_sol.ravel()
    results_data_frame.to_csv(results_path+run_path)
    
    

    