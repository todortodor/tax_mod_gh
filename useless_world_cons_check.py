#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:43:21 2022

@author: simonl
"""
main_path = '/Users/simonl/Dropbox/Mac/Documents/taff/tax_mod/'
import sys
sys.path.append(main_path+"lib/")
import solver_funcs as s
import data_funcs as d
import treatment_funcs as t
import pandas as pd
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm as tqdm

data_path = main_path+'data/'
year = '2018'

ba = d.baseline(year, data_path)
cons_sector = ba.cons.groupby('row_sector').value.sum().reset_index()


sector_map = pd.read_csv(data_path+'industry_labels_after_agg_expl.csv',sep=';')
sector_map['sector'] = cons_sector.row_sector.drop_duplicates()[:42].values
sector_map.set_index('sector',inplace=True)
cons_sector['row_sector'] = cons_sector['row_sector'].replace(sector_map['industry'])
cons_sector.set_index('row_sector',inplace=True)

#%%

cons_sector.sort_values('value',ascending=False).plot(kind='bar',figsize = (18,12))