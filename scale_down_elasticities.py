#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 18:06:11 2023

@author: slepot
"""

import pandas as pd
import numpy as np

elasticities_path = 'data/elasticities/'

# elasticities_run = 'elasticities_agg1.csv'

list_elast = {}

for elasticities_run in ['elasticities_agg1.csv','elasticities_agg2.csv']:

    elasticities_unscaled = pd.read_csv(elasticities_path+elasticities_run).set_index('sector')
    
    output = pd.read_csv('data/yearly_CSV_agg_treated/datas2018/output_2018.csv'
                         ).groupby('row_sector')['value'].sum(
                             ).rename_axis('sector')
    
    elasticities_scaled_to_4 = elasticities_unscaled*4/elasticities_unscaled['epsilon'].mean()
    elasticities_scaled_to_5 = elasticities_unscaled*5/elasticities_unscaled['epsilon'].mean()
    
    list_elast['rescaled_to_4'+elasticities_run[:-4]] = elasticities_scaled_to_4
    list_elast['rescaled_to_5'+elasticities_run[:-4]] = elasticities_scaled_to_5
    
    output_weighted_mean = (elasticities_unscaled['epsilon']*output).sum()/output.sum()
    
    elasticities_scaled_to_4_output_weighted = elasticities_unscaled*4/output_weighted_mean
    elasticities_scaled_to_5_output_weighted = elasticities_unscaled*5/output_weighted_mean
    
    list_elast['rescaled_to_4_output_weighted'+elasticities_run[:-4]] = elasticities_scaled_to_4_output_weighted
    list_elast['rescaled_to_5_output_weighted'+elasticities_run[:-4]] = elasticities_scaled_to_5_output_weighted
    
    # elasticities_scaled_to_4.to_csv(elasticities_path+'rescaled_to_4'+elasticities_run)
    # elasticities_scaled_to_5.to_csv(elasticities_path+'rescaled_to_5'+elasticities_run)
    
    # elasticities_scaled_to_4_output_weighted.to_csv(elasticities_path+'rescaled_to_4_output_weighted'+elasticities_run)
    # elasticities_scaled_to_5_output_weighted.to_csv(elasticities_path+'rescaled_to_5_output_weighted'+elasticities_run)
    
    # list_elast.append('rescaled_to_4'+elasticities_run)
    # list_elast.append('rescaled_to_5'+elasticities_run)
    # list_elast.append('rescaled_to_4_output_weighted'+elasticities_run)
    # list_elast.append('rescaled_to_5_output_weighted'+elasticities_run)


df = pd.DataFrame(index=list_elast['rescaled_to_4_output_weighted'+elasticities_run[:-4]].index)

for k in list_elast:
    df[k] = list_elast[k]['epsilon']
    
df_names = pd.read_csv('data/industry_labels_after_agg_expl.csv',sep=';')
df_names['sector'] = df_names['ind_code'].str.replace('D','')

df_names = df_names[['sector','industry']].set_index('sector')

df_names = df_names.join(df)
