#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 10:21:25 2022

@author: simonl
"""

import pandas as pd
import seaborn as sns
import numpy as np
import py4cytoscape as p4c
# import data_funcs as d
import treatment_funcs as t

#%% functions

def pivot_df(df,columns):
    #makes columns with sectors
    df = df.pivot(index=columns,columns=['sector'])
    df.columns = df.columns.reorder_levels([1,0])
    df.sort_index(axis=1,inplace = True,ascending=False)
    df.columns = df.columns.to_flat_index()
    df.columns = [', '.join(c) for c in df.columns]
    df.reset_index(inplace=True)
    return df

def map_sectors(df, data_path, column):
    #replace ISIC codes with more explicit sector names
    sector_map = pd.read_csv(data_path+'industry_labels_after_agg_expl.csv',sep=';')
    sector_map['sector'] = df.sector.drop_duplicates()[:42].values
    sector_map.set_index('sector',inplace=True)
    df[column] = df[column].replace(sector_map['industry'])
    return df

def concatenate_total(df,columns):
    #compute the total of sectors, columns are the country columns
    #that will be used for the groupby
    total = df.groupby(columns,as_index = False).sum()
    total['sector'] = 'Total'
    df = pd.concat([df,total])
    return df

def compute_share_of_exports(traded,target_column,new_column_name):
    traded.set_index(['source','sector','target'],inplace=True)
    traded[new_column_name] = (traded[target_column].reorder_levels([1,2,0]).div(traded.groupby(level=[1,2]).sum()[target_column])).reorder_levels([2,0,1])
    traded.reset_index(inplace=True)
    return traded

def compute_share_of_output(traded,tot,target_column,new_column_name,data_path):
    output = concatenate_total(
                map_sectors(tot.groupby(['row_country', 'row_sector']).sum()
                            .rename_axis(['source','sector'])
                            .reset_index(),data_path,'sector'),['source']
                ).set_index(['source','sector'])
    traded.set_index(['source','sector','target'],inplace=True)
    traded[new_column_name] = traded[target_column].div(output[target_column])
    traded.reset_index(inplace=True)
    return traded

def compute_share_of_imports(traded,target_column,new_column_name):
    traded.set_index(['source','sector','target'],inplace=True)
    traded[new_column_name] = (traded[target_column].reorder_levels([1,2,0]).div(traded.groupby(level=[1,2]).sum()[target_column])).reorder_levels([2,0,1])
    traded.reset_index(inplace=True)
    return traded

def compute_share_of_demand(traded,tot,target_column,new_column_name,data_path):
    inputs = concatenate_total(
                map_sectors(tot.groupby(['row_sector', 'col_country']).sum()
                            .rename_axis(['sector','target'])
                            .reset_index(),data_path,'sector'),['target']
                ).set_index(['sector','target'])
    traded.set_index(['source','sector','target'],inplace=True)
    traded[new_column_name] = (traded[target_column].reorder_levels([1,2,0]).div(inputs[target_column])).reorder_levels([2,0,1])
    traded.reset_index(inplace=True)
    return traded

def load_world_data(data_path):
    world = pd.read_csv(data_path+'countries_codes_and_coordinates.csv')[['Alpha-3 code','Latitude (average)', 'Longitude (average)']]
    world.columns = ['country','latitude','longitude']
    for column in world.columns: 
        world[column] = world[column].str.replace('"','')
        world[column] = world[column].str.replace(' ','')
    world['country'] = world['country'].astype('string')
    world.drop_duplicates('country',inplace=True)
    world['latitude'] = -pd.to_numeric(world['latitude'])*10
    world['longitude'] = pd.to_numeric(world['longitude'])*10
    world.loc[world.index.max()+1] = ['ROW',0,0]
    return world

def compute_nodes_and_edges_baseline(traded, tot, sol, data_path, world=None):
    #main function, computes the nodes and edges data from traded and sol_all
    #returns nodes and edges unpivoted = with a column "sectors" and the qty in columns
    #and nodes and edges pivoted = with columns for every sector X qty
    if world is None:
        world = load_world_data(data_path)
        
    # nodes   
    print('Computing nodes')
    nodes = traded.groupby(['row_country','row_sector'],as_index=False).sum().copy()
    nodes.columns = ['id','sector','output value','output new']
    nodes = pd.merge(nodes,traded.groupby(['row_sector','col_country'],as_index = False).sum()
                                 .rename(columns={'row_sector':'sector','col_country':'id'})
                     ,on = ['id','sector'])
    nodes.columns = ['id','sector','output value','output new','input value','input new']
    nodes = map_sectors(nodes, data_path, 'sector')
    nodes = concatenate_total(nodes,['id'])
    
    nodes['output hat'] = nodes['output new']/nodes['output value']
    nodes['input hat'] = nodes['input new']/nodes['input value']
    
    output_share_traded = (concatenate_total(traded.groupby(['row_country','row_sector'], as_index=False).sum()
                                              .rename(columns={'row_sector':'sector','row_country':'id'})
                                              ,['id']).set_index(['id','sector'])
                            / 
                            concatenate_total(tot.groupby(['row_country','row_sector'], as_index=False).sum()
                                              .rename(columns={'row_sector':'sector','row_country':'id'})
                                              ,['id']).set_index(['id','sector'])).reset_index()
    
    output_share_traded.columns = ['id','sector','share output traded value','share output traded new']
    output_share_traded['share output traded hat'] = output_share_traded['share output traded new']/output_share_traded['share output traded value']
    
    input_share_traded = (concatenate_total(traded.groupby(['row_sector','col_country'], as_index=False).sum()
                                              .rename(columns={'row_sector':'sector','col_country':'id'})
                                              ,['id']).set_index(['id','sector'])
                            / 
                            concatenate_total(tot.groupby(['row_sector','col_country'], as_index=False).sum()
                                              .rename(columns={'row_sector':'sector','col_country':'id'})
                                               ,['id']).set_index(['id','sector'])).reset_index()

    input_share_traded.columns = ['id','sector','share input traded value','share input traded new']
    input_share_traded['share input traded hat'] = input_share_traded['share input traded new']/input_share_traded['share input traded value']
    
    output_share_traded = map_sectors(output_share_traded,data_path,'sector')
    input_share_traded = map_sectors(input_share_traded,data_path,'sector')
    
    nodes = pd.merge(nodes,output_share_traded)
    nodes = pd.merge(nodes,input_share_traded)
    
    nodes = nodes.fillna(1)
    nodes = nodes[['id', 'sector', 'output value', 'output hat', 'input value', 'input hat'
                   ,'share output traded hat', 'share input traded hat']]
    
    nodes_unpivoted = nodes
    nodes = pivot_df(nodes,['id'])
    nodes = nodes.merge(world,left_on='id',right_on='country').drop('country',axis=1)
    nodes['latitude'] = nodes['latitude']*1.2
    
    #edges    
    print('Computing edges')
    edges = traded.groupby(['row_country', 'row_sector', 'col_country'],as_index=False).sum().copy()
    edges.columns = ['source','sector','target', 'value', 'new']
    
    edges = map_sectors(edges,data_path,'sector')
    
    prices = sol.price
    prices.columns = ['source','sector','price']
    prices = map_sectors(prices,data_path,'sector')
    edges = pd.merge(edges,prices,how='left',on=['source','sector'])
    edges['real new'] = edges['new'] / edges['price']
    
    edges = concatenate_total(edges,['source','target']).drop('price',axis=1)
    
    edges = compute_share_of_output(edges,tot,'value','share of output',data_path)
    edges = compute_share_of_output(edges,tot,'new','share of output new',data_path)
    
    edges = compute_share_of_exports(edges,'value','share of exports')
    edges = compute_share_of_exports(edges,'new','share of exports new')
    
    edges = compute_share_of_demand(edges,tot,'value','share of input',data_path)
    edges = compute_share_of_demand(edges,tot,'new','share of input new',data_path)
    
    edges = compute_share_of_imports(edges,'value','share of imports')
    edges = compute_share_of_imports(edges,'new','share of imports new')
    
    edges['hat'] = edges['new']/edges['value']
    edges['real hat'] = edges['real new']/edges['value']
    
    edges['share of output hat'] = edges['share of output new']/edges['share of output']
    edges['share of exports hat'] = edges['share of exports new']/edges['share of exports']
    edges['share of input hat'] = edges['share of input new']/edges['share of input']
    edges['share of imports hat'] = edges['share of imports new']/edges['share of imports']
    
    edges = edges.fillna(1)
    edges = edges[['source', 'sector', 'target', 'value','hat', 'real hat',
           'share of output hat', 'share of exports hat', 'share of input hat',
           'share of imports hat']]
    
    edges_unpivoted = edges
    edges = pivot_df(edges,['source','target'])
    
    return nodes_unpivoted, edges_unpivoted, nodes, edges
    
def create_baseline_network(nodes, edges, network_title):    
    #create network
    print('Creating network')
    network = p4c.create_network_from_data_frames(nodes = nodes, edges = edges, title=network_title)    
    
    return network
    
def create_network_for_specific_country(nodes_unpivoted, 
                                        edges_unpivoted, 
                                        country, 
                                        network_title, 
                                        input_or_output,
                                        sector,
                                        world=None):
    nodes = nodes_unpivoted
    
    if world is None:
        world = load_world_data()
        
    if input_or_output == 'input':
        nodes = pd.merge(nodes,
                         edges_unpivoted[edges_unpivoted.target == country][['source','sector','share of input hat']], 
                         left_on = ['id','sector'],
                         right_on = ['source','sector'],
                         how='left'
                         ).drop(columns='source')
        nodes.loc[nodes['id'] == country, 'share of input hat'] = nodes.loc[nodes['id'] == country, 'share input traded hat']
        for sector in nodes_unpivoted.sector.drop_duplicates():
            nodes.loc[(nodes['id'] != country) & (nodes['sector'] == sector), 'input value'] = edges_unpivoted.loc[(edges_unpivoted['target'] == country) & (edges_unpivoted['sector'] == sector),'value'].values
            nodes.loc[(nodes['id'] == country) & (nodes['sector'] == sector), 'input value'] = nodes.loc[(nodes['id'] != country) & (nodes['sector'] == sector), 'input value'].mean()
        
        nodes_country = nodes[['id','sector','input value','share of input hat']]   
        edges_country = edges_unpivoted[edges_unpivoted.target == country]
        
    if input_or_output == 'output':
        nodes = pd.merge(nodes,
                         edges_unpivoted[edges_unpivoted.source == country][['target','sector','share of output hat']], 
                         left_on = ['id','sector'],
                         right_on = ['target','sector'],
                         how='left'
                         ).drop(columns='target')
        nodes.loc[nodes['id'] == country, 'share of output hat'] = nodes.loc[nodes['id'] == country, 'share output traded hat']
        for sector in nodes_unpivoted.sector.drop_duplicates():
            nodes.loc[(nodes['id'] != country) & (nodes['sector'] == sector), 'output value'] = edges_unpivoted.loc[(edges_unpivoted['source'] == country) & (edges_unpivoted['sector'] == sector),'value'].values
            nodes.loc[(nodes['id'] == country) & (nodes['sector'] == sector), 'output value'] = nodes.loc[(nodes['id'] != country) & (nodes['sector'] == sector), 'output value'].mean()        
 
        nodes_country = nodes[['id','sector','output value','share of output hat']]   
        edges_country = edges_unpivoted[edges_unpivoted.source == country]
    
    
             
    nodes_country = pivot_df(nodes_country,['id'])
    nodes_country = nodes_country.merge(world,left_on='id',right_on='country').drop('country',axis=1)
    nodes_country['latitude'] = nodes_country['latitude']*1.2
        
    
    edges_country = pivot_df(edges_country,['source','target'])
    
    print('Creating network')
    network_country = p4c.create_network_from_data_frames(nodes = nodes_country, 
                                                          edges = edges_country, 
                                                          title = network_title)
        
    return nodes_country, edges_country, network_country

def c_map(df,sector,qty):
    data_points = [df[sector+', '+qty].quantile(i/10) for i in range(1,10)]
    color_points = sns.diverging_palette(15,120, s=65,l=50, n=9).as_hex()
    return data_points,color_points

# def size_map_node(df,sector,qty,size_max):
#     data_points = [df[sector+', '+qty].quantile(i/10) for i in range(0,11)]
#     size_points = list(np.linspace(size_max/5,size_max,len(data_points)))
#     return data_points, size_points

# def size_map_node_label(df,sector,qty,size_max):
#     data_points = [df[sector+', '+qty].quantile(i/10) for i in range(0,11)]
#     size_points = list(np.linspace(size_max/15,size_max/2.5,len(data_points)))
#     return data_points, size_points

class nodes_p:
    #class computes and outputs the actual values that will be mapped for colors and sizes for nodes
    def __init__(self,nodes,sector,node_size_max,
                 node_color_property_to_map,node_size_property_to_map):
        self.size_table_values = [0,nodes[sector+', '+node_size_property_to_map].max()]
        self.label_size_table_values = [0,nodes[sector+', '+node_size_property_to_map].max()]
        self.size_visual_values = [node_size_max/5,node_size_max]
        self.label_size_visual_values = [node_size_max/15,node_size_max/2.5]
        # self.size_table_values = size_map_node(nodes,sector,node_size_property_to_map,node_size_max)[0]
        # self.size_visual_values = size_map_node(nodes,sector,node_size_property_to_map,node_size_max)[1]
        # self.label_size_table_values = size_map_node_label(nodes,sector,node_size_property_to_map,node_size_max)[0]
        # self.label_size_visual_values = size_map_node_label(nodes,sector,node_size_property_to_map,node_size_max)[1]
        # print(size_map_node(nodes,sector,node_size_property_to_map,node_size_max)[0])
        # print(size_map_node(nodes,sector,node_size_property_to_map,node_size_max)[1])
        self.color_table_values = c_map(nodes,sector,node_color_property_to_map)[0]
        self.color_visual_values = c_map(nodes,sector,node_color_property_to_map)[1]
        
class edges_p:
    #class computes and outputs the actual values that will be mapped for colors and sizes for edges
    def __init__(self,edges,sector,edge_size_max,input_or_output,edge_color_property_to_map):
        self.size_table_values = [0,edges[sector+', value'].max()]
        self.size_visual_values = [0,edge_size_max]
        self.color_table_values = c_map(edges,sector,edge_color_property_to_map)[0]
        self.color_visual_values = c_map(edges,sector,edge_color_property_to_map)[1]
        self.transparency_table_values = [0,edges[sector+', value'].max()]
        self.transparency_visual_values = [100,255]
        if input_or_output == 'input':
            self.edge_source_arrow_shape = None
            self.edge_target_arrow_shape = 'DELTA'
        if input_or_output == 'output':
            self.edge_source_arrow_shape = 'CIRCLE'
            self.edge_target_arrow_shape = None
 
def make_title(sector, country, input_or_output, nominal_values_or_shares, final_or_inter_or_total):
    if country is None:
        if input_or_output == 'output':
            title = 'Exports'+' '+sector+' '+ nominal_values_or_shares+' '+final_or_inter_or_total
        elif input_or_output == 'input':
            title = 'Imports'+' '+sector+' '+ nominal_values_or_shares+' '+final_or_inter_or_total
    else:
        if input_or_output == 'output':
            title = 'Exports '+country+' '+sector+' '+ nominal_values_or_shares+' '+final_or_inter_or_total
        elif input_or_output == 'input':
            title = 'Imports '+country+' '+sector+' '+ nominal_values_or_shares+' '+final_or_inter_or_total
    return title        
 
#%% compute nodes and edges network

def load_baseline_compute_initial(sol, baseline, data_path, final_or_inter_or_total):  
    sector_list = sol.output['sector'].drop_duplicates().to_list()
    S = len(sector_list)
    country_list = sol.output['country'].drop_duplicates().to_list()
    C = len(country_list)
    
    baseline_iot = baseline.iot
    iot = sol.iot
    cons_temp = sol.cons
    baseline_cons_temp = baseline.cons
    cons_temp['col_sector'] = 'cons'
    baseline_cons_temp['col_sector'] = 'cons'
    print('Computing trade for '+final_or_inter_or_total)
    if final_or_inter_or_total == 'total':
        tot = pd.concat([baseline_iot,baseline_cons_temp])
        tot['new'] = pd.concat([iot,cons_temp])['value']
        traded = tot[tot['row_country'] != tot['col_country']]#.set_index(['row_country', 'row_sector', 'col_country','col_sector'])
    if final_or_inter_or_total == 'final':
        tot = baseline_cons_temp
        tot['new'] = cons_temp['value']
        traded = tot[tot['row_country'] != tot['col_country']]#.set_index(['row_country', 'row_sector', 'col_country','col_sector'])
    if final_or_inter_or_total == 'inter':
        tot = baseline_iot
        tot['new'] = iot['value']
        traded = tot[tot['row_country'] != tot['col_country']]#.set_index(['row_country', 'row_sector', 'col_country','col_sector'])
    
    world = load_world_data(data_path)
    nodes_unpivoted, edges_unpivoted, nodes_total, edges_total = \
        compute_nodes_and_edges_baseline(traded, tot, sol, data_path, world)
        
    return nodes_unpivoted, edges_unpivoted, nodes_total, edges_total, world, traded
    
#%% create and apply mapping
        
""" Sectors : ['Agriculture', 'Fishing', 'Mining, energy', 
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
"""

bypass_existence_check = True
remake_network = True
remake_style = True
fit_view = True

def build_network_and_style(sector,country,input_or_output,nodes_unpivoted, edges_unpivoted, nodes_total, 
                            edges_total, world, write_title, nominal_values_or_shares,final_or_inter_or_total):
    print(sector)
    print(country)
    # check if the network exists
    title = make_title(sector, country, input_or_output, nominal_values_or_shares,final_or_inter_or_total)
    
    if not bypass_existence_check: #check for already existing network if not bypassed
        if title not in p4c.get_network_list() or remake_network:
            try:
                p4c.delete_network(title)
            except:
                pass
            if country is None:
                network = create_baseline_network(nodes_total, edges_total, title)
            else:
                nodes_country, edges_country, network_country = \
                    create_network_for_specific_country(nodes_unpivoted, 
                                                        edges_unpivoted, 
                                                        country, 
                                                        title, 
                                                        input_or_output, 
                                                        world)
        else:
            p4c.set_current_network(network=title)
            
    if bypass_existence_check:
        if remake_network: #if check bypassed, can force the remake of the network
            if country is None:
                network = create_baseline_network(nodes_total, edges_total, title)
            else:
                nodes_country, edges_country, network_country = \
                    create_network_for_specific_country(nodes_unpivoted, 
                                                        edges_unpivoted, 
                                                        country, 
                                                        title, 
                                                        input_or_output, 
                                                        sector,
                                                        world)
                
    style_name = 'Style for '+sector+' '+title+' '+nominal_values_or_shares
    
    if remake_style: #if remake of style is forced, delete style if it exists
        if style_name in p4c.get_visual_style_names():
            p4c.delete_visual_style(style_name)
    
    if style_name not in p4c.get_visual_style_names(): #if style doesn't exist, build it
        
        print('Creating style')    
    
        node_size_max = 300
        edge_size_max = 60
        
        node_size_property_to_map = input_or_output+' value'
        
        if nominal_values_or_shares == 'shares':
            print('share changes')
            edge_color_property_to_map = 'share of '+input_or_output+' hat'
            if country is None:
                node_color_property_to_map = 'share '+input_or_output+' traded hat'
            else:
                node_color_property_to_map = 'share of '+input_or_output+' hat'
                
        if nominal_values_or_shares == 'values':
            print('nominal value changes')
            edge_color_property_to_map = 'hat'
            if country is None:
                node_color_property_to_map = input_or_output+' hat'
            else:
                node_color_property_to_map = 'share of '+input_or_output+' hat'
        
        if country is None:
            nodes = nodes_total
            edges = edges_total
        else:
            nodes = nodes_country
            edges = edges_country
            
        n = nodes_p(nodes,sector,node_size_max,node_color_property_to_map,node_size_property_to_map)
        e = edges_p(edges,sector,edge_size_max,input_or_output,edge_color_property_to_map)
            
        defaults = {'NODE_SHAPE': "circle",
                    'NODE_FILL_COLOR': '#DCDCDC',
                    'EDGE_TARGET_ARROW_SHAPE':e.edge_target_arrow_shape,
                    'EDGE_SOURCE_ARROW_SHAPE':e.edge_source_arrow_shape,
                    'EDGE_STACKING_DENSITY':0.7,
                    # 'NODE_LABEL_POSITION': 'W,E,c,0.00,0.00'
                    'NODE_LABEL_POSITION': 'W,c,100.00,0.00'
                    }
        n_labels = p4c.map_visual_property('node label', 'id', 'p')
        n_position_y = p4c.map_visual_property('NODE_Y_LOCATION', 'latitude', 'p')
        n_position_x = p4c.map_visual_property('NODE_X_LOCATION', 'longitude', 'p')
        n_position_z = p4c.map_visual_property('NODE_Z_LOCATION', sector+', '+node_size_property_to_map, 'p')
        n_size = p4c.map_visual_property('NODE_SIZE', sector+', '+node_size_property_to_map, 'c',
                                                table_column_values = n.size_table_values,
                                                visual_prop_values = n.size_visual_values )
        n_label_size = p4c.map_visual_property('NODE_LABEL_FONT_SIZE', sector+', '+node_size_property_to_map, 'c', 
                                               table_column_values = n.label_size_table_values,
                                               visual_prop_values = n.label_size_visual_values)
    
        n_color = p4c.map_visual_property('NODE_FILL_COLOR', sector+', '+node_color_property_to_map, 'c', 
                                                table_column_values = n.color_table_values,
                                                visual_prop_values = n.color_visual_values )
        n_tooltip = p4c.map_visual_property('NODE_TOOLTIP', sector+', '+node_color_property_to_map, 'p')
                                            
        e_stroke_color = p4c.map_visual_property('EDGE_STROKE_UNSELECTED_PAINT', sector+', '+edge_color_property_to_map, 'c', 
                                                table_column_values=e.color_table_values,
                                                visual_prop_values=e.color_visual_values )
        e_arrow_source_color = p4c.map_visual_property('EDGE_SOURCE_ARROW_UNSELECTED_PAINT', sector+', '+edge_color_property_to_map, 'c', 
                                                table_column_values=e.color_table_values,
                                                visual_prop_values=e.color_visual_values )
        e_arrow_target_color = p4c.map_visual_property('EDGE_TARGET_ARROW_UNSELECTED_PAINT', sector+', '+edge_color_property_to_map, 'c', 
                                                table_column_values=e.color_table_values,
                                                visual_prop_values=e.color_visual_values )
        e_width = p4c.map_visual_property('EDGE_WIDTH', sector+', value', 'c', 
                                               table_column_values = e.size_table_values,
                                               visual_prop_values = e.size_visual_values )
        e_arrow_target_width = p4c.map_visual_property('EDGE_TARGET_ARROW_SIZE', sector+', value', 'c', 
                                               table_column_values = e.size_table_values,
                                               visual_prop_values = e.size_visual_values )
        e_arrow_source_width = p4c.map_visual_property('EDGE_SOURCE_ARROW_SIZE', sector+', value', 'c', 
                                               table_column_values = e.size_table_values,
                                               visual_prop_values = e.size_visual_values)
        e_tooltip = p4c.map_visual_property('EDGE_TOOLTIP', sector+', '+edge_color_property_to_map, 'p')
        
        if country is None:
            e_transparency = p4c.map_visual_property('EDGE_TRANSPARENCY', sector+', value', 'c', 
                                                   table_column_values = e.transparency_table_values,
                                                   visual_prop_values = e.transparency_visual_values)
        else:
            e_transparency = p4c.map_visual_property('EDGE_TRANSPARENCY', sector+', value', 'c', 
                                                   table_column_values = e.transparency_table_values,
                                                   visual_prop_values = [255, 255])
        
        p4c.create_visual_style(style_name, defaults = defaults, mappings = [n_labels,
                                                                             n_position_y,
                                                                             n_position_x,
                                                                             n_position_z,
                                                                             n_size,
                                                                             n_label_size,
                                                                             n_color,
                                                                             n_tooltip,
                                                                             e_width,
                                                                             e_stroke_color,
                                                                             e_arrow_target_color,
                                                                             e_arrow_source_color,
                                                                             e_arrow_target_width,
                                                                             e_arrow_source_width,
                                                                             e_transparency,
                                                                             e_tooltip
                                                                             ])
        
    p4c.set_visual_style(style_name = style_name) #apply style
    
    w = 3700
    h = 2435*w/4378
    if 'map' not in [anot['name'] for anot in p4c.get_annotation_list()]: #add map as annotation if not already there
        p4c.add_annotation_image(url='https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/World_map_blank_without_borders.svg/4378px-World_map_blank_without_borders.svg.png'
                                  , x_pos=-1755.38, y_pos=-1028.95, opacity=100, brightness=0, contrast=None,
                                  border_thickness=None, border_color=None, border_opacity=0, height=h, width=w,
                                  name='map', canvas='background', z_order=None, network=None)
    
    p4c.toggle_graphics_details()
    # p4c.hide_all_panels()
    # p4c.dock_panel('control')
    # if not fit_view:
        # p4c.set_network_center_bypass(331,-142, bypass=True)
        # p4c.set_network_zoom_bypass(10, bypass=True)
    # if fit_view:
    p4c.fit_content()
    europe = ['SVN','POL','IRL','NLD','EST','DNK','GRC','SWE','AUT','NOR','PRT','ITA',
              'CHE','SVK','ISL','ESP','BEL','LTU','CZE','FRA','LVA','HRV','FIN','DEU','ROU','BGR']
    rest_of_europe = [c for c in europe if c != country]
    if country is not None:
        p4c.set_node_property_bypass(country, 'diamond', 'NODE_SHAPE')
        p4c.set_node_property_bypass(country, 1e9, 'NODE_Z_LOCATION')
        p4c.set_node_property_bypass(country, node_size_max/2, 'NODE_SIZE')
        p4c.set_node_font_size_bypass(country, node_size_max/6)
        if country in europe:
            p4c.set_node_property_bypass(country, 190, 'NODE_TRANSPARENCY')
    p4c.set_node_property_bypass(rest_of_europe, 140, 'NODE_TRANSPARENCY')        
    if write_title:
        p4c.add_annotation_text(text=title, x_pos=-1000, y_pos=800, font_size=100, font_family=None, font_style=None)
        
        
#%% export as image
def export_as_image(path,height,width):
    p4c.hide_all_panels()
    p4c.fit_content()
    title = p4c.get_network_name()
    print(path+title)
    p4c.export_image(filename=path+title, type='PNG', units='pixels', height=height, width=width)

#%% choose color palette
# from numpy import arange
# x = arange(25).reshape(5, 5)
# cmap = sns.diverging_palette(10,120 , s=70,l=50, as_cmap=True)
# ax = sns.heatmap(x, cmap=cmap)


# temp = edges_unpivoted[(edges_unpivoted['source'] == 'ROW') & (edges_unpivoted['target'] == 'CHN')]
# temp.set_index('sector')['hat weighted']/edges_unpivoted['hat weighted'].mean()

