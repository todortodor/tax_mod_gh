#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 17:46:57 2022

@author: simonl
"""

import pandas as pd

main_path = './'
data_path = main_path+'data/'
eta_path = 'uniform_elasticities_4.csv'
sigma_path = 'uniform_elasticities_4.csv'

for y in range(2018,2019):
    year = str(y)
    runs_path = '/Users/simonl/Dropbox/Mac/Documents/taff/tax_mod/results/'+year+'_2/runs.csv'
    runs = pd.read_csv(runs_path,index_col = 0)
    runs.rename(columns = {'eta':'eta_path','sigma':'sigma_path'}, inplace = True)
    runs['eta_path'] = eta_path
    runs['sigma_path'] = sigma_path
    runs.to_csv(runs_path)