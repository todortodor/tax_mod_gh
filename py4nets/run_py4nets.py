#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 08:07:29 2022

@author: simonl
"""

import sys
main_path = ("../")
# sys.path.append(main_path+"lib/")
import py4nets_funcs as py4nets
data_path = main_path+"data/"
results_path = main_path+"results/"
import data_funcs as d
import treatment_funcs as t
dir_num = 1

y=2018 #year
carb_cost = 1e-2 #carbon tax in million dollars by ton of CO2
taxed_countries = ['CHN']
taxed_sectors = None
specific_taxing = None
fair_tax = False

simulation_case = {'carb_cost':carb_cost,
                    'taxed_countries': taxed_countries,
                    'taxed_sectors':taxed_sectors,
                    'specific_taxing':specific_taxing,
                    'fair_tax':fair_tax}
final_or_inter_or_total = 'total'  # 'inter' or 'final' or 'total'
""""trade flows for final demand or trade flows of intermediate inputs or total trade flows
This is not a very interesting quantity to look at"""

baseline = d.baseline(y, data_path)
relevant_runs,found_cases,not_found_cases = t.find_runs(simulation_case,
                                                        results_path,
                                                        dir_num,
                                                        y,
                                                        drop_duplicate_runs = True,
                                                        keep='last')
sol = t.sol(relevant_runs.squeeze(), results_path).compute_solution(baseline)

nodes_unpivoted, edges_unpivoted, nodes_total, edges_total, world = \
    py4nets.load_baseline_compute_initial(sol, baseline, data_path, final_or_inter_or_total)

#%% build and display the network

sector = 'Total' #'a sector' (see list) or 'Total'
country = None #'a country' (see list) or None to build the whole network
input_or_output = 'output' #'input' or 'output'
nominal_values_or_shares = 'values' #'values' or 'shares'
write_title_in_antartica = False #True or False

py4nets.build_network_and_style(sector,country,input_or_output,nodes_unpivoted, edges_unpivoted, nodes_total, 
                                edges_total, world, write_title_in_antartica, nominal_values_or_shares,final_or_inter_or_total)

"""""
Countries :
['ARG','AUS', 'AUT', 'BEL', 'BGR', 'BRA','BRN', 'CAN', 'CHE', 'CHL', 'CHN','COL', 'CRI', 
 'CYP', 'CZE', 'DEU', 'DNK', 'ESP', 'EST', 'FIN', 'FRA', 'GBR', 'GRC', 'HRV', 'HUN', 
 'IDN', 'IND', 'IRL', 'ISL', 'ISR', 'ITA', 'JPN', 'KAZ', 'KHM', 'KOR', 'LAO', 'LTU', 
 'LVA', 'MAR', 'MEX', 'MLT', 'MMR', 'NLD', 'NOR', 'NZL', 'PER', 'PHL', 'POL', 'PRT', 
 'ROU', 'ROW', 'RUS', 'SAU', 'SGP', 'SVK', 'SVN', 'SWE', 'THA', 'TUN', 'TUR', 'TWN', 
 'USA', 'VNM', 'ZAF']

Sectors :
['Agriculture', 'Fishing', 'Mining, energy', 
'Mining, non-energy', 'Food products', 'Textiles', 
'Wood', 'Paper', 'Coke, petroleum', 'Chemicals', 
'Pharmaceuticals', 'Plastics', 'Non-metallic minerals', 
'Basic metals', 'Fabricated metals', 'Electronic', 
'Electrical equipment', 'Machinery', 'Transport equipments', 
'Manufacturing nec', 'Energy', 'Water supply', 'Construction', 
'Wholesale, retail', 'Land transport', 'Water transport', 'Air transport', 
'Warehousing', 'Post', 'Tourism', 'Media', 'Telecom', 'IT', 
'Finance, insurance', 'Real estate', 'R&D', 'Administration', 'Public sector', 
'Education', 'Health', 'Entertainment', 'Other service']


The size of the edge is always proportional to the nominal value of the trade flow in the specified sector.
The node and edge sizes are scaled so that the size max is always the same.
Nodes have a minimum size of size_max/5, Edges don't have minimum size, a $0 trade flow will not appear.
The color scale are always for changes, e.g. [(quantity)_counterfactual/(quantity)_baseline] and distributed 
in deciles around the median projected on a linear color map, so the outliers will be squished.
The median of the change of the quantity is white/transparent.


For the whole network (Country = None):
If 'output', the network will be built on exports :
    The size of the node is the baseline nominal value of production of traded goods (exports)
    If 'values' the network is built on nominal values:
        - the color of the node is the change of production of traded goods (prod_traded_goods)
        - the color of the edge is the trade flow change in nominal value (trade_flow)
    If 'shares' the network will be built on shares of the total output of the exporter:
        - the color of the node is the change in share of production traded (prod_traded_goods / prod_total_of_exporter)
            (so red means the country should use a bigger share of their own production domestically, green means they should export more)
        - the color of the edge is the change in (trade_flow / prod_total_of_exporter)
            (so red means this trade flow should decrease even more than the reduction of production of the exporter
             this really represents reorganisation of trade. If there was no reorganisation of trade, all edges would be white/transparent) 
            
If 'input', the network will be built on imports :
    The size of the node is the baseline volume of demand for traded goods (imports)
    If 'values' the network is built on nominal values:
        - the color of the node is the change of total imports of goods (imports_traded_goods)
        - the color of the edge is the trade flow change in nominal value (trade_flow)
    If 'shares' the network will be built on shares of the total demand of the importer:
        - the color of the node is the change in share of demand that comes from traded goods (imports_traded_goods / total_demand_of_importer)
            (so red means the country should use a bigger share of their production domestically, green means they should import more)
        - the color of the edge is the change in (trade_flow / total_demand_of_importer)
            (so red means this trade flow should decrease even more than the reduction of demand of the importer
             this represents a reorganisation net of income effects.)         


For a highlighted country C :
It works the same, but it is all relative to exports / imports of the chosen country C
Note : In that case then 'shares' and 'values' maps will look exactly the same because the shares are all relative
to the same quantity : the exports or imports of country C, so the ranking of flows isn't affected and therefore 
the colors either.
The size of C doesn't represent anything, it is arbitrary and always the same.
The size of the other nodes are proportional to the baseline Imports('input')/Exports('output') from C in nominal value
If 'output':
    The color of C is the change in share of production that is traded (prod_traded_goods)
        (so if C is red, C should export less of his production and use it domestically more
         if C is green, C should export more of his production and use it domestically less)
    The color of the other nodes is the change in exports from C to the node
    The color of the edges is the change in the trade flow (trade_flow)
        Note : visually the other nodes and the corresponding edges will have the same colors
If 'input':
    The color of C is the change in share of demand that is traded (demand_traded_goods)
        (so if C is green, C should import more of his demand from outside
         if C is red, C should import less of his demand from outside)
    The color of the other nodes is the change in imports of C from the node
    The color of the edges is the change in the trade flow (trade_flow)
        Note : visually the other nodes and the corresponding edges will have the same colors    

"""""

#%% export as an image

path = main_path+'graph_images/'
height = 4000
width = 8000
py4nets.export_as_image(path,height,width)