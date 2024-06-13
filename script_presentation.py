#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:56:29 2022

@author: slepot
"""
#%% import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from adjustText import adjust_text
from tqdm import tqdm
# from labellines import labelLines
# import treatment_funcs as t
import lib.data_funcs as d
import lib.treatment_funcs as t
import os

#%% Set seaborn parameters
sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')


#%% path stuffs

save_path = 'presentation_material/uniform_elasticities/paper/'
data_path = 'data/'
results_path = 'results/'

try:
    os.mkdir(save_path)
except:
    pass


save_all = True
save_format = 'eps'

#%% Intro plots (data pres)

#%% GHG distrib over time by sector of use - Climate watch data

save = False

print('Plotting GHG distribution by categories')

years = [y for y in range(1995,2019)]

emissions_baseline = [pd.read_csv('data/yearly_CSV_agg_treated/datas'+str(year)+'/prod_CO2_WLD_with_agri_agri_ind_proc_fug_'+str(year)+'.csv').value.sum() for year in years]

ghg = pd.read_csv('data/climate_watch/ghg_EDGAR_ip.csv').set_index(['sector','year'])

datas = [ 'Electricity/Heat',
        'Manufacturing/Construction','Transportation','Agriculture','Forest fires',
         'Fugitive Emissions', 'Industrial Processes','Other Fuel Combustion',
        'Waste']

labels = [ 'Energy (fuel)',
        'Industries (fuel)','Transports (fuel)','Agriculture (direct emissions) \nand related land-use change',
        'Forestry - Land change',  
         'Fugitive Emissions', 'Industrial Processes','Other Fuel Combustion',
        'Waste (direct emissions)']

to_plot = [ghg.loc[data].value for data in datas]

fig, ax = plt.subplots(figsize=(16,8),constrained_layout=True)

palette2 = [sns.color_palette()[i] for i in [2,1,3,0]]
palette2[0] = (0.4333333333333333, 0.5588235294117647, 0.40784313725490196)
palette = [*sns.color_palette("dark:salmon_r",n_colors=6)[::-1][:5] , *palette2]
palette[3] , palette[5] = palette[5] , palette[3]
palette[4] , palette[7] = palette[7] , palette[4]
palette[6] , palette[7] = palette[7] , palette[6]

stacks = ax.stackplot(years, 
             to_plot,
             labels = labels,
                colors = palette,zorder = -10,
                linewidth=0.5
             )

hatches = ['','','','','','','','//','//']
for stack, hatch in zip(stacks, hatches):
    stack.set_hatch(hatch)

plt.grid(alpha=0.4)

ax.plot(years,np.array(emissions_baseline)/1e3,ls='--',lw=6,color = sns.color_palette('Set2')[5],label = 'Data')

ax.plot(years, (-ghg.loc['Forestland']).value, ls = '--', lw=3, label = 'Forest carbon sinks \nNatural net zero target', color= 'g')

handles, labels = plt.gca().get_legend_handles_labels()
order = [9,0,1,2,3,4,5,6,7,8,10]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order]
          ,loc=(1,0),fontsize = 25) 
ax.margins(x=0)

ax.tick_params(axis='both', labelsize=20 )
# ax.set_title('Main greenhouse gases emissions (in Gt of CO2 equivalent)',fontsize=25,pad=15)

plt.tight_layout()


# if save or save_all:
#     plt.savefig(save_path+'data_em_by_sector.pdf',format='pdf')

plt.show()

#%% GHG distrib over time by gas - Climate watch data

print('Plotting GHG distribution by gas')

save = False

carbon_prices = pd.read_csv('data/emissions_priced.csv')

ghg = pd.read_csv('data/climate_watch/ghg_EDGAR_gas.csv').set_index('gas')

datas = ['CO2_data', 'CO2', 'CH4_data', 'CH4', 'N2O_data', 'N2O',   'F-Gas']

to_plot = [ghg.loc[data].value for data in datas]

fig, ax = plt.subplots(figsize=(12,8),constrained_layout=True)

palette = [sns.color_palette()[i] for i in [7,7,5,5,3,3,9]]

stacks = ax.stackplot(years, 
             *to_plot,
             # labels = labels,
             colors = palette, 
             zorder = -10,
             lw = 0.5
             )
hatches = [None,'/',None,'/',None,'/',None]
for stack, hatch in zip(stacks, hatches):
    stack.set_hatch(hatch)
    
plt.grid(alpha=0.4)

# phantom plots for legend. it's dirty !
legend_labels = ['Carbon dioxyde','Methane','Nitrous oxyde','F-Gas','Datas','Unaccounted for']
legend_colors = [sns.color_palette()[i] for i in [7,5,3,9,0,0]]
legend_hatches = [None]*5+['//']
stacks_legend = ax.stackplot([],[[]*24]*6,colors = legend_colors,labels = legend_labels)
for stack_legend, hatch in zip(stacks_legend, legend_hatches):
    stack_legend.set_hatch(hatch)
    
# ax.plot(years, carbon_prices.value, ls = '--', lw=5, label = 'Emissions falling under some\ntype of carbon pricing scheme', color= 'r')    
    
ax.legend(loc='lower left'
           ,fontsize=17)


ax.margins(x=0)

ax.tick_params(axis='both', labelsize=20 )
# ax.set_title('Main greenhouse gases emissions (in Gt of CO2 equivalent)',fontsize=25,pad=15)


if save or save_all:
    plt.savefig(save_path+'data_em_by_gas.'+save_format,format=save_format)
plt.show()
#%% Pie charts distributions LOAD BASELINE

print('Plotting pie charts of the distribution of emissions by sector/country')

save = False

y  = 2018
year = str(y)

sector_map = pd.read_csv('data/industry_labels_after_agg_expl.csv',sep=';').set_index('ind_code')
sector_list = sector_map.industry.to_list()

countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country')
country_list = countries.country.to_list()

S = len(sector_list)
C = len(country_list)

b = d.baseline(y,data_path) # LOAD BASELINE #!!!!
b = b.make_np_arrays()

sector_list_noD = []
for sector in sector_map.index.to_list():
    sector_list_noD.append(sector[1:])

years = [y for y in range(1995,2019)]

emissions_baseline = [pd.read_csv('data/yearly_CSV_agg_treated/datas'+str(year)+'/prod_CO2_WLD_with_agri_agri_ind_proc_fug_'+str(year)+'.csv').value.sum() for year in years]

fig, ax = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
# ,constrained_layout=True
nbr_labels = 10
fontsize = 10
fontsize_title = 20
lab_distance = 1.04
center = (0,0)
radius = 1

data = b.co2_prod.groupby(level=0).sum().value
labels = [country if data.loc[country]>data.sort_values(ascending=False).iloc[nbr_labels] else '' for country in country_list]
colors_emissions = sns.diverging_palette(145, 50, s=80,l=40)
colors_output =sns.diverging_palette(220, 20, s=80,l=40)

colors_emissions = sns.diverging_palette(200, 130, s=80,l=40)
colors_output =sns.diverging_palette(60, 20, s=80,l=40)

ax[0,0].pie(x = data,
        labels = labels,
        explode=np.ones(len(countries.country))-0.98,
        colors=colors_emissions,
        autopct=None,
        pctdistance=0.6,
        shadow=False,
        labeldistance=lab_distance,
        startangle=0,
        radius=radius,
        counterclock=False,
        wedgeprops={'linewidth': 0.5},
        textprops={'fontsize':fontsize},
        center=center,
        frame=False,
        rotatelabels=False,
        normalize=True)

ax[0,0].set_title('Emissions distribution',fontsize=fontsize_title,y=0.8,x=1.25)

data = b.output.groupby(level=0).sum().value
labels = [country if data.loc[country]>data.sort_values(ascending=False).iloc[nbr_labels] else '' for country in country_list]

ax[1,0].pie(x = data,
        labels = labels,
        explode=np.ones(len(countries.country))-0.98,
        colors=colors_output,
        autopct=None,
        pctdistance=0.6,
        shadow=False,
        labeldistance=lab_distance,
        startangle=0,
        radius=radius,
        counterclock=False,
        wedgeprops={'linewidth': 0.5},
        textprops={'fontsize':fontsize},
        center=center,
        frame=False,
        rotatelabels=False,
        normalize=True)


ax[1,0].set_title('Gross output distribution',fontsize=fontsize_title,y=0.8,x=1.25)

data = b.co2_prod.groupby(level=1).sum().value
labels = [sector_list[i] if data.loc[sector]>data.sort_values(ascending=False).iloc[nbr_labels] else '' for i,sector in enumerate(sector_list_noD)]

ax[0,1].pie(x = data,
        labels = labels,
        explode=np.ones(len(sector_map.industry))-0.98,
        colors=colors_emissions,
        autopct=None,
        pctdistance=0.6,
        shadow=False,
        labeldistance=lab_distance,
        startangle=0,
        radius=radius,
        counterclock=False,
        wedgeprops={'linewidth': 0.5},
        textprops={'fontsize':fontsize},
        center=center,
        frame=False,
        rotatelabels=False,
        normalize=True)


# ax[0,1].set_title('Emission output distribution',fontsize=fontsize_title)

data = b.output.groupby(level=1).sum().value
labels = [sector_list[i] if data.loc[sector]>data.sort_values(ascending=False).iloc[nbr_labels] else '' for i,sector in enumerate(sector_list_noD)]

ax[1,1].pie(x = data,
        labels = labels,
        explode=np.ones(len(sector_map.industry))-0.98,
        colors=colors_output,
        autopct=None,
        pctdistance=0.6,
        shadow=False,
        labeldistance=lab_distance,
        startangle=0,
        radius=radius,
        counterclock=False,
        wedgeprops={'linewidth': 0.5},
        textprops={'fontsize':fontsize},
        center=center,
        frame=False,
        rotatelabels=False,
        normalize=True)


# ax[1,1].set_title('Gross output distribution',fontsize=fontsize_title)



if save or save_all:
    plt.savefig(save_path+'emissions_by_sector_country_pies.'+save_format,format=save_format)
plt.show()

#%% only emissions

fig, ax = plt.subplots(1,2,figsize=(12,8),constrained_layout=True)
# ,constrained_layout=True
nbr_labels = 10
fontsize = 10
fontsize_title = 20
lab_distance = 1.04
center = (0,0)
radius = 1

data = b.co2_prod.groupby(level=0).sum().value
labels = [country if data.loc[country]>data.sort_values(ascending=False).iloc[nbr_labels] else '' for country in country_list]
colors_emissions = sns.diverging_palette(145, 50, s=80,l=40)
colors_output =sns.diverging_palette(220, 20, s=80,l=40)

colors_emissions = sns.diverging_palette(200, 130, s=80,l=40)
colors_output =sns.diverging_palette(60, 20, s=80,l=40)

ax[0].pie(x = data,
        labels = labels,
        explode=np.ones(len(countries.country))-0.98,
        colors=colors_emissions,
        autopct=None,
        pctdistance=0.6,
        shadow=False,
        labeldistance=lab_distance,
        startangle=0,
        radius=radius,
        counterclock=False,
        wedgeprops={'linewidth': 0.5},
        textprops={'fontsize':fontsize},
        center=center,
        frame=False,
        rotatelabels=False,
        normalize=True)

# ax[0].set_title('Emissions distribution',fontsize=fontsize_title,y=0.8,x=1.25)

data = b.output.groupby(level=0).sum().value
labels = [country if data.loc[country]>data.sort_values(ascending=False).iloc[nbr_labels] else '' for country in country_list]

data = b.co2_prod.groupby(level=1).sum().value
labels = [sector_list[i] if data.loc[sector]>data.sort_values(ascending=False).iloc[nbr_labels] else '' for i,sector in enumerate(sector_list_noD)]

ax[1].pie(x = data,
        labels = labels,
        explode=np.ones(len(sector_map.industry))-0.98,
        colors=colors_emissions,
        autopct=None,
        pctdistance=0.6,
        shadow=False,
        labeldistance=lab_distance,
        startangle=0,
        radius=radius,
        counterclock=False,
        wedgeprops={'linewidth': 0.5},
        textprops={'fontsize':fontsize},
        center=center,
        frame=False,
        rotatelabels=False,
        normalize=True)


# ax[0,1].set_title('Emission output distribution',fontsize=fontsize_title)

data = b.output.groupby(level=1).sum().value
labels = [sector_list[i] if data.loc[sector]>data.sort_values(ascending=False).iloc[nbr_labels] else '' for i,sector in enumerate(sector_list_noD)]

# ax[1,1].set_title('Gross output distribution',fontsize=fontsize_title)
save=False
if save or save_all:
    plt.savefig(save_path+'emissions_only_by_sector_country_pies.'+save_format,format=save_format)
plt.show()

#%% Existing carbon pricing

save = False

print('Plotting existing carbon tax / ETS system')

carbon_prices_all = pd.read_csv('data/weighted_average_prices_rest0.csv')
carbon_prices_all = carbon_prices_all.replace('combined','Combined').set_index(['instrument','year'])
carbon_prices = pd.read_csv('data/weighted_average_prices.csv')
carbon_prices = carbon_prices.replace('combined','Mean').set_index(['instrument','year'])

fig, ax = plt.subplots(figsize=(12,8))

for data in carbon_prices.index.get_level_values(0).drop_duplicates():
    ax.plot(years[-len(carbon_prices.loc[data].price):],carbon_prices.loc[data].price, lw=3, label = data)

ax.legend()
ax.set_ylabel('Dollars per ton of carbon',fontsize = 20)
plt.title('Average carbon price on emissions that fall under a pricing scheme',fontsize=20)

plt.show()

if save or save_all:
    plt.savefig(save_path+'carbon_pricing.'+save_format,format=save_format)
    

#%% Agriculture emissions

print('Plotting Agriculture emissions details')

save = False

emissions_agri = pd.read_csv('data/agriculture_emissions_subsectors.csv').sort_values('category')
palette = []*len(emissions_agri)
palette[:4] = sns.dark_palette("grey", n_colors=5)[1:]#[::-1]
palette[4:7] = sns.dark_palette("green", n_colors=5)[1:][::-1]
palette[8:] = sns.dark_palette("red", n_colors=6)[1:][::-1]

fig, ax = plt.subplots(figsize=(10,10))

ax.pie(x = emissions_agri.total_CO2eq,
        labels = emissions_agri.Item,
        explode=np.ones(len(emissions_agri.Item))-0.98,
        colors=palette,
        autopct=None,
        pctdistance=0.6,
        shadow=False,
        labeldistance=lab_distance,
        startangle=0,
        radius=radius,
        counterclock=False,
        wedgeprops={'linewidth': 0.5},
        textprops={'fontsize':20},
        center=center,
        frame=False,
        rotatelabels=False,
        normalize=True)



if save or save_all:
    plt.savefig(save_path+'agriculture_emissions.'+save_format,format=save_format)
plt.show()
#%% LOAD SOLUTIONS DIFFERENT CARBON TAXES, CHOOSE OPTIONS 
#!!!!

carb_cost_list = np.linspace(0,1e-3,1001)
# carb_cost_list = [1e-4]
# eta_path = ['elasticities_agg1.csv']
# sigma_path = ['elasticities_agg1.csv']
eta_path = ['uniform_elasticities_4.csv']
sigma_path = ['uniform_elasticities_4.csv']
taxed_countries_list = [None]
taxing_countries_list = [None]
taxed_sectors_list = [None]
specific_taxing_list = [None]
fair_tax_list = [False]
y  = 2018
years = [y]
dir_num = 1

cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                      taxed_sectors_list,specific_taxing_list,fair_tax_list)



sols, baselines, relevant_runs, found_cases, not_found_cases = t.sol.load_sols(cases,
                                                      years,
                                                      dir_num,
                                                      results_path,
                                                      data_path,
                                                      baselines = None,
                                                      compute_sols = True,
                                                      # compute_hats= True,
                                                      return_not_found_cases=True,
                                                      drop_duplicate_runs=True,
                                                      keep='last')

b = baselines[y]
t_index = np.argmin(np.abs(np.array([sol.params.carb_cost for sol in sols])-1e-4))
sol = sols[t_index]

#%% Plot macro effects

save = False

print('Plotting welfare and GDP cost corresponding to a carbon tax')

carb_taxes = np.array([sol.params.carb_cost*1e6  for sol in sols])
norm_emissions = np.array([sol.run.emissions/baselines[y].co2_prod.value.sum() for sol in sols])
norm_gdp = np.array([sol.va.value.sum()/baselines[y].va.value.sum() for sol in sols])
norm_real_income = np.array([sol.run.utility for sol in sols])

fig, ax = plt.subplots(2,2,figsize=(12,8))

color = 'g'

# Upper left - Emissions
ax[0,0].plot(carb_taxes,norm_emissions,lw=4,color=color)
ax[0,0].legend(['Global emissions'])
ax[0,0].set_xlabel('')
ax[0,0].tick_params(axis='x', which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax[0,0].set_xlim(0,1000)

y_100 = np.array(norm_emissions)[np.argmin(np.abs(carb_taxes-100))]
# y_0 = runs_low_carb_cost.iloc[0].emissions

ax[0,0].vlines(x=100,
            ymin=0,
            ymax=y_100,
            lw=3,
            ls = '--',
            color = color)

ax[0,0].hlines(y=y_100,
            xmin=0,
            xmax=100,
            lw=3,
            ls = '--',
            color = color)

ax[0,0].margins(y=0)

ax[0,0].annotate(str(y_100.round(3)),
             xy=(100,y_100),
             xytext=(0,0),
             textcoords='offset points',color=color)

ax[0,0].set_ylim(norm_emissions.min(),norm_emissions.max()+0.05)

# Upper right - GDP
color = 'b'

ax[0,1].plot(carb_taxes,norm_gdp,lw=4)
ax[0,1].set_xlabel('')
ax[0,1].tick_params(axis='x', which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
ax[0,1].set_xlim(0,1000)
ax[0,1].legend(['GDP'])

y_100 = norm_gdp[np.argmin(np.abs(carb_taxes-100))]

ax[0,1].vlines(x=100,
            ymin=norm_gdp.min(),
            ymax=y_100,
            lw=3,
            ls = '--',
            color = color)

ax[0,1].hlines(y=y_100,
            xmin=0,
            xmax=100,
            lw=3,
            ls = '--',
            color = color)

ax[0,1].margins(y=0)

ax[0,1].annotate(str(y_100.round(3)),
              xy=(100,y_100),
              xytext=(0,0),
              textcoords='offset points',color=color)

ax[0,1].set_ylim(norm_gdp.min(),norm_gdp.max()+0.005)

# Bottom left - Welfare
color = 'r'

ax[1,0].plot(carb_taxes,norm_real_income,lw=4,color='r')
ax[1,0].legend(['Real income'])
ax[1,0].set_xlabel('Carbon tax ($/ton of CO2eq.)')
ax[1,0].set_xlim(0,1000)
# ax[1,0].set_ylim(min(utility),1.001)

y_100 = norm_real_income[np.argmin(np.abs(carb_taxes-100))]
# y_0 = runs_low_carb_cost.iloc[0].emissions

ax[1,0].vlines(x=100,
            ymin=norm_real_income.min(),
            ymax=y_100,
            lw=3,
            ls = '--',
            color = color)

ax[1,0].hlines(y=y_100,
            xmin=0,
            xmax=100,
            lw=3,
            ls = '--',
            color = color)

ax[1,0].margins(y=0)

ax[1,0].set_ylim(norm_real_income.min(),1.005)

ax[1,0].annotate(str(y_100.round(3)),
              xy=(100,y_100),
              xytext=(0,0),
              textcoords='offset points',color=color)

# Bottom right summary
ax[1,1].plot(carb_taxes,norm_gdp,lw=4)
ax[1,1].plot(carb_taxes,norm_real_income,lw=4,color='r')
ax[1,1].plot(carb_taxes,norm_emissions,lw=4,color='g')
ax[1,1].legend(['GDP','Real income','Emissions'])
ax[1,1].set_xlabel('Carbon tax ($/ton of CO2eq.)')
ax[1,1].set_xlim(0,1000)

plt.tight_layout()


if save or save_all:
    plt.savefig(save_path+'macro_effects.'+save_format,format=save_format)
plt.show()
#%% Effect on output share traded COMPUTE SHARE TRADED

print('Plotting share of output traded')

iot_traded_unit = b.iot.copy()
iot_traded_unit['value'] = 1
iot_traded_unit.loc[iot_traded_unit.query("row_country == col_country").index, 'value'] = 0
cons_traded_unit = b.cons.copy()
cons_traded_unit['value'] = 1
cons_traded_unit.loc[cons_traded_unit.query("row_country == col_country").index, 'value'] = 0

carb_taxes = np.array([sol.params.carb_cost*1e6  for sol in sols])
share_traded = np.array([((sol.cons.value.to_numpy() * cons_traded_unit.value.to_numpy()).sum() + (sol.iot.value.to_numpy()  * iot_traded_unit.value.to_numpy()).sum())*100 /\
                         (sol.cons.value.to_numpy().sum()+sol.iot.value.to_numpy().sum()) for sol in sols])
    
    
fig, ax1 = plt.subplots(figsize=(12,8))
color = 'tab:blue'

ax1.set_xlabel('Carbon tax ($/ ton of CO2)',size = 20)
ax1.set_xlim(0,1000)
ax1.tick_params(axis='x', labelsize = 20)

ax1.set_ylabel('Share of output traded (%)', size = 20)
ax1.plot(carb_taxes,share_traded, color=color,lw=5)
ax1.tick_params(axis='y', labelsize = 20)
ax1.set_ylim(10,20)

plt.tight_layout()



if save or save_all:
    plt.savefig(save_path+'share_traded.'+save_format,format=save_format)
plt.show()

#%% Scatter plot of gross ouput change with kernel density by coarse industry

print('Plotting scatter plot of output changes for every country x sector according to produciton intensity with kernel density estimates for categories of sectors')

save = False

sector_list = b.sector_list

p_hat_sol = sol.res.price_hat.to_numpy().reshape(C,S)
E_hat_sol = sol.res.output_hat.to_numpy().reshape(C,S)
q_hat_sol = E_hat_sol / p_hat_sol
q_hat_sol_percent = (q_hat_sol-1)*100

sector_map = pd.read_csv(data_path+'industry_labels_after_agg_expl_wgroup.csv').set_index('ind_code')

data = pd.DataFrame(data = q_hat_sol_percent.ravel(),
                    index = pd.MultiIndex.from_product([country_list,sector_map.index.get_level_values(level=0).to_list()],
                                                       names=['country','sector']),
                    columns=['value'])
data = data.reset_index().merge(sector_map.reset_index(),how='left',left_on='sector',right_on='ind_code').set_index(['country','sector']).drop('ind_code',axis=1)
data['co2_intensity'] = b.co2_intensity.value.values
data['output'] = b.output.value.values
data=data.sort_values('group_code')
data = data.loc[data.group_label.isin(['Agro-food','Raw materials','Energy'])]

sector_list_full = []
for sector in sector_list:
    sector_list_full.append(sector_map.loc['D'+sector].industry)

group_labels_sorted = data.group_label.drop_duplicates().to_list()

data_no_z =data.copy()
data_no_z = data_no_z[data_no_z['co2_intensity'] != 0]
# data_no_z = data_no_z[data_no_z['co2_intensity'] < 1e4]
# data_no_z['co2_intensity'] = np.log(data_no_z['co2_intensity'])
data_no_z = data_no_z[['co2_intensity','value','group_label','group_code','output']]

data_no_z_1 = data_no_z[data_no_z['co2_intensity'] < 100].copy()
data_no_z_2 = data_no_z[data_no_z['co2_intensity'] >= 100].copy()

fig, ax = plt.subplots(figsize=(12,8),constrained_layout=True)
# # sns.move_legend(plot, "lower left", bbox_to_anchor=(.55, .45), title='Species')

# plot.fig.get_axes()[0].legend(loc='lower left')
# # plt.legend(loc='lower left')

palette = [sns.color_palette('bright')[i] for i in [2,4,0,3,1,7]]
palette[0] = sns.color_palette()[2]
palette[1] = sns.color_palette("hls", 8)[-2]
for data_no_z_i in [data_no_z_1,data_no_z_2] :
# for data_no_z_i in [data_no_z] :
    plot2 = sns.kdeplot(data=data_no_z_i,
                x='co2_intensity',
                y="value",
                hue = 'group_label',
                fill = True,
                alpha = 0.25,
                log_scale=(True, False),
                # height=10,
                # ratio=5,
                # bw_adjust=0.5,
                weights = 'output',
                legend=False,
                levels = 2,
                palette = palette,
                common_norm = True,
                # shade=True,
                thresh = 0.2,
                # fill = False,
                # alpha=0.6,
                # hue_order = data.group_label.drop_duplicates().to_list()[::-1],
                ax = ax
                )
for data_no_z_i in [data_no_z_1,data_no_z_2] :
    for i,group in enumerate(data_no_z_i.group_code.drop_duplicates().to_list()):
        ax.scatter(data_no_z_i[data_no_z_i['group_code'] == group].co2_intensity,
                   data_no_z_i[data_no_z_i['group_code'] == group].value,
                   color=palette[i],
                   s=(data_no_z_i[data_no_z_i['group_code'] == group].output)/1e3,zorder=1-i)

ax.set_ylabel('Production changes (%)',
                fontsize=20
                )
ax.set_xscale('log')
ax.set_ylim(-80,+37.5)
# ax.set_ylim(-80, +80)

# ax.set_xlim(0.5,20000)
ax.set_xlim(data_no_z.co2_intensity.min(),3e4)
ax.margins(x=0)
ax.tick_params(axis='both', which='major', labelsize=20)


ax.set_xlabel('Carbon intensity of production (Tons / Mio.$)',
                fontsize=20)

handles = [mpatches.Patch(color=palette[ind], label=group_labels_sorted[ind]) for ind,group in enumerate(group_labels_sorted)]
ax.legend(handles=handles,fontsize=20, loc = 'lower left')

ax.xaxis.set_major_formatter(ScalarFormatter())

ax.hlines(0,xmin=b.co2_intensity.value.min(),xmax=1e5,colors='black',ls='--',lw=1)

# sec = '20'
# sector = sector_map.loc['D' + sec].industry
# sector_index = sector_list.index(sec)

# country = 'RUS'
# country_index = country_list.index(country)

# ax.annotate(country + ' - ' + sector,
#             xy=(b.co2_intensity.loc[country, sec].value, q_hat_sol_percent[country_index, sector_index]),
#             xycoords='data',
#             xytext=(-250, 0),
#             textcoords='offset points',
#             va='center',
#             arrowprops=dict(arrowstyle="->",
#                             connectionstyle="arc3", color='black'),
#             bbox=dict(boxstyle="round", fc="w"), zorder=10
#             )

# sec = '28'
# sector = sector_map.loc['D' + sec].industry
# sector_index = sector_list.index(sec)

# country = 'CHN'
# country_index = country_list.index(country)

# ax.annotate(country + ' - ' + sector,
#             xy=(b.co2_intensity.loc[country, sec].value, q_hat_sol_percent[country_index, sector_index]),
#             xycoords='data',
#             xytext=(-50, -80),
#             textcoords='offset points',
#             va='center',
#             arrowprops=dict(arrowstyle="->",
#                             connectionstyle="arc3", color='black'),
#             bbox=dict(boxstyle="round", fc="w"), zorder=10
#             )

# sec = '35'
# sector = sector_map.loc['D' + sec].industry
# sector_index = sector_list.index(sec)

# country = 'NOR'
# country_index = country_list.index(country)

# ax.annotate(country + ' - ' + sector,
#             xy=(b.co2_intensity.loc[country, sec].value, q_hat_sol_percent[country_index, sector_index]),
#             xycoords='data',
#             xytext=(20, 80),
#             textcoords='offset points',
#             va='center',
#             arrowprops=dict(arrowstyle="->",
#                             connectionstyle="arc3", color='black'),
#             bbox=dict(boxstyle="round", fc="w"), zorder=10
#             )

# sec = '50'
# sector = sector_map.loc['D' + sec].industry
# sector_index = sector_list.index(sec)

# country = 'DEU'
# country_index = country_list.index(country)

# ax.annotate(country + ' - ' + sector,
#             xy=(b.co2_intensity.loc[country, sec].value, q_hat_sol_percent[country_index, sector_index]),
#             xycoords='data',
#             xytext=(80, 15),
#             textcoords='offset points',
#             va='center',
#             arrowprops=dict(arrowstyle="->",
#                             connectionstyle="arc3", color='black'),
#             bbox=dict(boxstyle="round", fc="w"), zorder=10
#             )

# sec = '01T02'
# sector = sector_map.loc['D' + sec].industry
# sector_index = sector_list.index(sec)

# country = 'BRA'
# country_index = country_list.index(country)

# ax.annotate(country + ' - ' + sector,
#             xy=(b.co2_intensity.loc[country, sec].value, q_hat_sol_percent[country_index, sector_index]),
#             xycoords='data',
#             xytext=(-250, -5),
#             textcoords='offset points',
#             va='center',
#             arrowprops=dict(arrowstyle="->",
#                             connectionstyle="arc3", color='black'),
#             bbox=dict(boxstyle="round", fc="w"), zorder=10
#             )

# sec = '01T02'
# sector = sector_map.loc['D' + sec].industry
# sector_index = sector_list.index(sec)

# country = 'CHE'
# country_index = country_list.index(country)

# ax.annotate(country + ' - ' + sector,
#             xy=(b.co2_intensity.loc[country, sec].value, q_hat_sol_percent[country_index, sector_index]),
#             xycoords='data',
#             xytext=(100, -35),
#             textcoords='offset points',
#             va='center',
#             arrowprops=dict(arrowstyle="->",
#                             connectionstyle="arc3", color='black'),
#             bbox=dict(boxstyle="round", fc="w"), zorder=10
#             )

# plt.tight_layout()

sec = '24'
sector = sector_map.loc['D' + sec].industry
sector_index = sector_list.index(sec)

country = 'CHL'
country_index = country_list.index(country)

ax.annotate(country + ' - ' + sector,
            xy=(b.co2_intensity.loc[country, sec].value, q_hat_sol_percent[country_index, sector_index]),
            xycoords='data',
            xytext=(-250, 0),
            textcoords='offset points',
            va='center',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color='black'),
            bbox=dict(boxstyle="round", fc="w"), zorder=10
            )
sec = '24'
sector = sector_map.loc['D' + sec].industry
sector_index = sector_list.index(sec)

country = 'PER'
country_index = country_list.index(country)

ax.annotate(country + ' - ' + sector,
            xy=(b.co2_intensity.loc[country, sec].value, q_hat_sol_percent[country_index, sector_index]),
            xycoords='data',
            xytext=(-250, 0),
            textcoords='offset points',
            va='center',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color='black'),
            bbox=dict(boxstyle="round", fc="w"), zorder=10
            )
sec = '19'
sector = sector_map.loc['D' + sec].industry
sector_index = sector_list.index(sec)

country = 'IRL'
country_index = country_list.index(country)

ax.annotate(country + ' - ' + sector,
            xy=(b.co2_intensity.loc[country, sec].value, q_hat_sol_percent[country_index, sector_index]),
            xycoords='data',
            xytext=(-250, 0),
            textcoords='offset points',
            va='center',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color='black'),
            bbox=dict(boxstyle="round", fc="w"), zorder=10
            )

sec = '03'
sector = sector_map.loc['D' + sec].industry
sector_index = sector_list.index(sec)

country = 'IDN'
country_index = country_list.index(country)

ax.annotate(country + ' - ' + sector,
            xy=(b.co2_intensity.loc[country, sec].value, q_hat_sol_percent[country_index, sector_index]),
            xycoords='data',
            xytext=(-200, -100),
            textcoords='offset points',
            va='center',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color='black'),
            bbox=dict(boxstyle="round", fc="w"), zorder=10
            )

sec = '05T06'
sector = sector_map.loc['D' + sec].industry
sector_index = sector_list.index(sec)

country = 'SAU'
country_index = country_list.index(country)

ax.annotate(country + ' - ' + sector,
            xy=(b.co2_intensity.loc[country, sec].value, q_hat_sol_percent[country_index, sector_index]),
            xycoords='data',
            xytext=(150, 0),
            textcoords='offset points',
            va='center',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color='black'),
            bbox=dict(boxstyle="round", fc="w"), zorder=10
            )

# if save or save_all:
#     plt.savefig(save_path+'micro_effect.pdf',format='pdf')
plt.show()
#%% Production reallocation sector-wise, % changes

print('Plotting production reallocation in percentages')

sector_map = pd.read_csv(data_path+'industry_labels_after_agg_expl_wgroup.csv')
sector_map['sector'] = sector_map['ind_code'].str.replace('D','')
data = 100*(sol.output.groupby('sector').sum()-b.output.groupby('sector').sum())/b.output.groupby('sector').sum()
data = data.reset_index().merge(sector_map,
                                how='left',
                                left_on='sector',
                                right_on='sector').set_index(['sector']).drop('ind_code',axis=1)
data = data.sort_values('value').reset_index()

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

ax.bar(data.sector
            ,data.value
            ,label='Change of nominal output (%)',
            )

ax.set_xticklabels(['']
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)

# ax.set_yscale('log')
ax.tick_params(axis='x', which='major', labelsize = 20, pad=-9)
ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)
ax.set_ylabel('Percent change of nominal output', fontsize = 20)


leg = ax.legend(fontsize=20,loc='lower right')
ax.grid(axis='x')

ax.bar_label(ax.containers[0],
             labels=data.industry,
             rotation=90,
              label_type = 'edge',
              padding=2,
              zorder=10)

ax.set_ylim(-32,12)

# plt.tight_layout()



if save or save_all:
    plt.savefig(save_path+'cross_sector_effects.'+save_format,format=save_format)

plt.show()
#%% Production reallocation country-wise, % changes

countries = pd.read_csv('data/countries_after_agg.csv',sep=';').sort_values('country')
data = 100*(sol.output.groupby('country').sum()-b.output.groupby('country').sum())/b.output.groupby('country').sum()
data = data.sort_values('value').reset_index()

fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

ax.bar(data.country
            ,data.value
            ,label='Change of nominal output (%)',
            )

ax.set_xticklabels(['']
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)

# ax.set_yscale('log')
ax.tick_params(axis='x', which='major', labelsize = 20, pad=-9)
ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)
ax.set_ylabel('Percent change of nominal output', fontsize = 20)


leg = ax.legend(fontsize=20,loc='lower right')
ax.grid(axis='x')

ax.bar_label(ax.containers[0],
             labels=data.country,
             rotation=90,
              label_type = 'edge',
              padding=2,
              zorder=10)

ax.set_ylim(-16,8)

# plt.tight_layout()


if save or save_all:
    plt.savefig(save_path+'cross_country_effects.'+save_format,format=save_format)
plt.show()
# %% Reallocation of labor, percentage changes - FIG 9

print('Plotting workforce reallocation in percentages')

country_map = pd.read_csv(data_path+'countries_after_agg.csv', sep=';').set_index('country')
income_rank = pd.read_csv(data_path+'World bank/country_income_rank.csv',sep=';',index_col=0)
country_map = country_map.join(income_rank)
labor = pd.read_csv('data/World bank/labor_force/labor.csv')
labor.set_index('country', inplace=True)
labor.sort_index(inplace=True)
labor_year = labor[year]

# Calculate labor force by country x sector in counterfactual and realized world
labor_sect = sol.va.join(pd.DataFrame(sol.va.groupby('col_country')['value'].sum()/labor_year, columns = ['wage']))
labor_sect = labor_sect['value']/labor_sect['wage']

labor_sect_baseline = b.va.join(pd.DataFrame(b.va.groupby('col_country')['value'].sum()/labor_year, columns = ['wage']))
labor_sect_baseline = labor_sect_baseline['value']/labor_sect_baseline['wage']

labor_sect_diff = labor_sect-labor_sect_baseline

data = country_map.copy()
data = data.join(pd.DataFrame(labor_year))
data['realloc'] = labor_sect_diff[labor_sect_diff>0].groupby('col_country').sum().rename_axis('country')
data['realloc_percent'] = data['realloc']*100/data['2018']

data.sort_values('realloc_percent', inplace = True)

fig, ax = plt.subplots(figsize=(16, 10), constrained_layout=True)

palette = [sns.color_palette()[i] for i in [3,0,2]]
colors = [palette[ind] for ind in data.income_code]

ax1=ax.twinx()

ax.bar(data.index.get_level_values(0)
       , data.realloc_percent
       # ,bottom = country_dist_df.realloc_neg
       # , label='Net reallocation (%)'
       , color=colors
       )

ax.set_xticklabels(['']
                   , rotation=75
                   # , ha='right'
                   # , rotation_mode='anchor'
                   , fontsize=19)
# ax.set_yscale('log')
# ax.tick_params(axis='x', which='major', labelsize = 18, pad=-9)
ax.tick_params(axis='y', labelsize=20)
ax.margins(x=0.01)
ax.set_ylabel('% of national labor force',
              fontsize=25)

handles = []
# for ind in country_df.income_code.drop_duplicates().to_list():
handles = [mpatches.Patch(color=palette[ind], 
                          label=data[data.income_code == ind].income_label.drop_duplicates().to_list()[0]
                          ) 
           for ind in data.income_code.drop_duplicates().to_list()]
legend = ax1.legend(handles=handles,
          fontsize=20,
          # title='Greensourcing possibility',
           loc='upper left'
          )
ax1.grid(visible=False)
ax1.set_yticks([])

# leg = ax.legend(fontsize=20, loc='upper left')
# leg.legendHandles[0].set_color('grey')
# leg.legendHandles[1].set_color('grey')

ax.grid(axis='x')
# ax.set_ylim(-20,5)
max_lim = data['realloc_percent'].max()
ax.set_ylim(0, max_lim + 1)

ax.bar_label(ax.containers[0],
             labels=data.index.get_level_values(0),
             rotation=90,
             label_type='edge',
             padding=3, zorder=10, fontsize=15)


if save or save_all:
    plt.savefig(save_path+'within_country_disruption.'+save_format,format=save_format)
plt.show()
#%% Inequalities relative to gdp per capita

print('Plotting inequalities')

# Construct GDP per capita and welfare change
gdp = sol.va.groupby(level=0).sum()
labor = pd.read_csv('data/World bank/labor_force/labor.csv')
labor.set_index('country', inplace=True)
labor.sort_index(inplace=True)
labor_year = labor[year]
gdp = gdp.join(labor_year).rename(columns={year:'labor'})
gdp['per_capita'] = (gdp.value / gdp.labor)*1e3

gdp['utility_percent_change'] = (sol.utility.values-1)*100

# Format regions for kernel density
country_map = pd.read_csv('data/country_continent.csv',sep=';').set_index('country')
gdp = gdp.join(country_map)

income_rank = pd.read_csv('data/World bank/country_income_rank.csv',sep=';',index_col=0)
gdp = gdp.join(income_rank)

gdp.loc['TWN','Continent'] = 'Asia'
gdp.loc['ROW','Continent'] = 'Africa'
gdp.loc['AUS','Continent'] = 'Asia'
gdp.loc['NZL','Continent'] = 'Asia'
gdp.loc['CRI','Continent'] = 'South America'
gdp.loc['RUS','Continent'] = 'Asia'
gdp.loc['SAU','Continent'] = 'Africa'
gdp.loc['CAN','labor'] = gdp.loc['CAN','labor']*6
gdp.loc['MEX','labor'] = gdp.loc['MEX','labor']*2

# palette = sns.color_palette()[0:5][::-1]
palette = [sns.color_palette()[i] for i in [0,2,3]]
income_colors = {
    'Low-income' : palette[2],
    'Middle-income' : palette[0],
    'High-income' : palette[1],
                    }
colors = [income_colors[gdp.loc[country,'income_label']] for country in b.country_list]

fig, ax = plt.subplots(figsize=(12,8),constrained_layout = True)

ax.scatter(gdp.per_capita,gdp.utility_percent_change,marker='x',lw=2,s=50,c = colors)     # For kernel density
# ax.scatter(gdp.per_capita,gdp.utility_percent_change,marker='x',lw=2,s=50)

ax.set_xlabel('GDP per workforce (Thousands $)', fontsize = 20)
ax.set_ylabel('Real income change (%)', fontsize = 20)

sns.kdeplot(data=gdp,
                x='per_capita',
                y="utility_percent_change",
                hue = 'income_label',
                fill = True,
                alpha = 0.25,
                # height=10,
                # ratio=5,
                # bw_adjust=0.7,
                weights = 'labor',
                legend=False,
                levels = 2,
                palette = palette,
                # common_norm = False,
                shade=True,
                thresh = 0.2,
                # dropna=True,
                # fill = False,
                # alpha=0.6,
                # hue_order = data.group_label.drop_duplicates().to_list()[::-1],
                ax = ax
                )

# sns.move_legend(ax, "lower right")

ax.set_xlim(0,175)
ax.set_ylim(-8,3)

handles = []
# for ind in country_df.income_code.drop_duplicates().to_list():
handles = [mpatches.Patch(color=income_colors[gdp[gdp.income_code == ind].income_label.drop_duplicates().to_list()[0]], 
                          label=gdp[gdp.income_code == ind].income_label.drop_duplicates().to_list()[0]
                          ) 
           for ind in [0,1,2]]
legend = ax.legend(handles=handles,
          fontsize=20,
          # title='Greensourcing possibility',
           loc='lower right'
          )

texts = [plt.text(gdp.per_capita.loc[country], gdp.utility_percent_change.loc[country], country,size=15, c = colors[i]) for i,country in enumerate(country_list)]     # For kernel density

adjust_text(texts, precision=0.001,
        expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
        force_text=(0.01, 0.25), force_points=(0.01, 0.25),
        arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
                        ))
save=True
if save or save_all:
    plt.savefig(save_path+'welfare_changes.pdf',format='pdf')
plt.show()


#%% Cross sector effects trade COMPUTE SHARE TRADED BY SECTOR

iot_traded_unit = b.iot.copy()
iot_traded_unit['value'] = 1
iot_traded_unit.loc[iot_traded_unit.query("row_country == col_country").index, 'value'] = 0
cons_traded_unit = b.cons.copy()
cons_traded_unit['value'] = 1
cons_traded_unit.loc[cons_traded_unit.query("row_country == col_country").index, 'value'] = 0

sector_traded = (sol.cons.value * cons_traded_unit.value).groupby('row_sector').sum()+(sol.iot.value * iot_traded_unit.value).groupby('row_sector').sum()
share_traded_sector = sector_traded/(sol.cons.value.groupby('row_sector').sum()+sol.iot.value.groupby('row_sector').sum())

sector_traded_baseline = (b.cons.value * cons_traded_unit.value).groupby('row_sector').sum()+(b.iot.value * iot_traded_unit.value).groupby('row_sector').sum()
share_traded_sector_baseline = sector_traded_baseline/(b.cons.value.groupby('row_sector').sum()+b.iot.value.groupby('row_sector').sum())

sector_map = pd.read_csv(data_path+'industry_labels_after_agg_expl_wgroup.csv')
sector_map['sector'] = sector_map['ind_code'].str.replace('D','')
sector_map.set_index('sector',inplace=True)

data = pd.DataFrame((share_traded_sector-share_traded_sector_baseline)*100/share_traded_sector_baseline)
data = data.sort_values('value')
data.rename_axis('sector',inplace=True)
data = data.join(sector_map)
data.reset_index(inplace=True)


fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

ax.bar(data.sector
            ,data.value
            ,label='Change of share of output traded (%)',
            )

ax.set_xticklabels(['']
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)

# ax.set_yscale('log')
ax.tick_params(axis='x', which='major', labelsize = 20, pad=-9)
ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)
ax.set_ylabel('% of initial share', fontsize = 20)


leg = ax.legend(fontsize=20,loc='lower right')
ax.grid(axis='x')

ax.bar_label(ax.containers[0],
             labels=data.industry,
             rotation=90,
              label_type = 'edge',
              padding=2,
              zorder=10)

# ax.set_ylim(-60,180)
ax.set_ylim(-15,35)

# plt.tight_layout()


if save or save_all:
    plt.savefig(save_path+'cross_sector_effects_trade.'+save_format,format=save_format)
plt.show()
#%% Cross country effects trade COMPUTE SHARE TRADED BY COUNTRY

iot_traded_unit = b.iot.copy()
iot_traded_unit['value'] = 1
iot_traded_unit.loc[iot_traded_unit.query("row_country == col_country").index, 'value'] = 0
cons_traded_unit = b.cons.copy()
cons_traded_unit['value'] = 1
cons_traded_unit.loc[cons_traded_unit.query("row_country == col_country").index, 'value'] = 0

country_traded = (sol.cons.value * cons_traded_unit.value).groupby('row_country').sum()+(sol.iot.value * iot_traded_unit.value).groupby('row_country').sum()
share_traded_country = country_traded/(sol.cons.value.groupby('row_country').sum()+sol.iot.value.groupby('row_country').sum())

country_traded_baseline = (b.cons.value * cons_traded_unit.value).groupby('row_country').sum()+(b.iot.value * iot_traded_unit.value).groupby('row_country').sum()
share_traded_country_baseline = country_traded_baseline/(b.cons.value.groupby('row_country').sum()+b.iot.value.groupby('row_country').sum())

data = pd.DataFrame((share_traded_country-share_traded_country_baseline)*100/share_traded_country_baseline)
data = data.sort_values('value')
data.rename_axis('country',inplace=True)
# data = data.join(country_map)
data.reset_index(inplace=True)


fig, ax = plt.subplots(figsize=(18,10),constrained_layout = True)

ax.bar(data.country
            ,data.value
            ,label='Change of share of output traded (%)',
            )

ax.set_xticklabels(['']
                    , rotation=45
                    , ha='right'
                    , rotation_mode='anchor'
                    ,fontsize=19)

# ax.set_yscale('log')
ax.tick_params(axis='x', which='major', labelsize = 20, pad=-9)
ax.tick_params(axis='y', labelsize = 20)
ax.margins(x=0.01)
ax.set_ylabel('% of initial import share', fontsize = 20)


leg = ax.legend(fontsize=20,loc='lower right')
ax.grid(axis='x')

ax.bar_label(ax.containers[0],
             labels=data.country,
             rotation=90,
              label_type = 'edge',
              padding=2,
              zorder=10)

ax.set_ylim(-6,25)

# plt.tight_layout()


if save or save_all:
    plt.savefig(save_path+'cross_country_effects_trade.'+save_format,format=save_format)
plt.show()