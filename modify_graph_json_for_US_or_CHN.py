#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 18:10:29 2023

@author: slepot
"""

import json
from tqdm import tqdm

nodes = []
edges = []

country = 'CHN'

# Opening JSON file
with open('/Users/slepot/Documents/taff/todortodor.github.io/data/graph.json') as json_file:
    data = json.load(json_file)
 
    # # Print the type of data variable
    # print("Type:", type(data))
 
    # # Print the data of dictionary
    # print("\nPeople1:", data['people1'])
    # print("\nPeople2:", data['people2'])
    
    nodes = data['nodes']
    
    for edge in tqdm(data['edges']):
        if country in edge['id']:
            edges.append(edge)
            
    data_country = {
        'nodes' : nodes,
        'edges' : edges
        }

with open('/Users/slepot/Documents/taff/todortodor.github.io/data/graph_'+country+'.json', 'w') as f:
    json.dump(data_country, f)
    
#%% no TWN
    
import json
from tqdm import tqdm

nodes = []
edges = []

country = 'USA'

# Opening JSON file
with open('/Users/slepot/Documents/taff/todortodor.github.io/data/graph.json') as json_file:
    data = json.load(json_file)
 
    # # Print the type of data variable
    # print("Type:", type(data))
 
    # # Print the data of dictionary
    # print("\nPeople1:", data['people1'])
    # print("\nPeople2:", data['people2'])
    
    for node in tqdm(data['nodes']):
        if 'TWN' not in node['id']:
            nodes.append(node)
    
    for edge in tqdm(data['edges']):
        if country in edge['id'] and 'TWN' not in edge['id']:
            edges.append(edge)
            
    data_country = {
        'nodes' : nodes,
        'edges' : edges
        }

with open('/Users/slepot/Documents/taff/todortodor.github.io/data/graph_no_TWN_'+country+'.json', 'w') as f:
# with open('/Users/slepot/Documents/taff/todortodor.github.io/data/graph_no_TWN.json', 'w') as f:
    json.dump(data_country, f)
