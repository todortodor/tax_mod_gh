#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 16:15:32 2024

@author: slepot
"""
main_path = './'
import sys
sys.path.append(main_path+'lib/')
import solver_funcs as s
import data_funcs as d
from tqdm import tqdm
import numpy as np
# from deco import *
import pandas as pd
import time
from time import perf_counter
from scipy import optimize

dir_num = 102
data_path = main_path+'data/'
results_path = 'results/'

numeraire_type = 'wage'
numeraire_country = 'WLD'

carb_cost_list = [0]
eta_path = ["cp_estimate_allyears.csv"]
sigma_path = ["cp_estimate_allyears.csv"]
taxed_countries_list = [None]
taxing_countries_list = [None]
taxed_sectors_list = [None]
specific_taxing_list = [None]
tax_scheme_list = ['consumer']

fair_tax_list = [False]
pol_pay_tax_list = [False]
tau_factor_list = [1]
autarky = False

cases = d.build_cases(eta_path,sigma_path,carb_cost_list,taxed_countries_list,taxing_countries_list,
                      taxed_sectors_list,specific_taxing_list,fair_tax_list,pol_pay_tax_list,tax_scheme_list,tau_factor_list,
                      same_elasticities=True)
exclude_direct_emissions = False

params = d.params(data_path, **cases[0])

years = [2018]

y=2018         
year=str(y)

baseline = d.baseline(year, data_path, exclude_direct_emissions=exclude_direct_emissions)

# baseline.num_scale(numeraire_type, numeraire_country, inplace = True)
baseline.num_type = 'no_scaling'
baseline.num = 1

baseline.make_np_arrays(inplace = True)

baseline.compute_shares_and_gammas(inplace = True)

baseline.co2_intensity = baseline.co2_intensity/1e6
baseline.co2_intensity_np = baseline.co2_intensity_np/1e6

#%%

import solver_funcs as s

helper = s.helper_for_min()

lb = []
ub = []
# lb.append(np.ones(64)*10)
lb.append(np.zeros(64))
ub.append(np.ones(64)*1000)
bounds = (np.concatenate(lb),np.concatenate(ub))

# def minimize_with_carbon_price():
opt_cab_price = optimize.least_squares(fun = s.to_minimize_from_carb_price,    
                        x0 = np.ones(64)*100, 
                        args = (params, baseline, helper), 
                        bounds = bounds,
                        max_nfev=1e8,
                        xtol=1e-10, 
                        verbose = 2)

#%%
fair_carb_price = pd.read_csv('fair_cab_price.csv').set_index('col_country')*1e6
helper.carb_price = fair_carb_price.values.squeeze()

import solver_funcs as s
while True:
    opt_cab_price = optimize.least_squares(fun = s.to_minimize_from_carb_price,    
                            x0 = helper.carb_price, 
                            # x0 = opt_cab_price.x, 
                            args = (params, baseline, helper), 
                            bounds = bounds,
                            max_nfev=1e8,
                            xtol=1e-10, 
                            verbose = 2)
    
#%%

# import solver_funcs as s
# while True:
#     opt_cab_price_scal = optimize.minimize(fun = s.scalar_to_minimize_from_carb_price,    
#                             # x0 = helper.carb_price, 
#                             x0 = opt_cab_price.x, 
#                             args = (params, baseline, helper), 
#                             bounds = [(0,1000)]*64,
#                             # max_nfev=1e8,
#                             # xtol=1e-10, 
#                             # verbose = 2,
#                             options={'disp':True}
#                             )

#%%

df = pd.DataFrame(index=baseline.country_list)
df.rename_axis('col_country',inplace=True)
df['value'] = opt_cab_price.x/1e6
df.to_csv('fair_cab_price_3.csv')

#%%

import numpy as np
import matplotlib.pyplot as plt

# Sample 2D array (you can replace this with your own data)
data = opt_cab_price.jac

fig,ax = plt.subplots(figsize = (14,14))

# Create a heatmap
# ax.imshow(data, cmap='viridis', interpolation='nearest')
ax.imshow(data, cmap='viridis')
# plt.colorbar()  # Add a colorbar for reference

# Optionally, add labels to the axes if needed
plt.xticks(range(len(data[0])), range(len(data[0])))
plt.yticks(range(len(data)), range(len(data)))

plt.title("2D Array Heatmap")
plt.show()

#%%
from scipy.optimize import minimize
import solver_funcs as s

# helper = s.helper_for_min()

# # Define your objective function
# def objective_function(x, *args):
#     v, y = s.to_minimize_under_constraint_from_carb_price(x, *args)  # Calculate v and y using your function f with additional arguments
#     # return np.sum(v**2)
#     return v

# # Define the constraint function
# def constraint_function(x, *args):
#     v, y = s.to_minimize_under_constraint_from_carb_price(x, *args)  # Calculate v and y using your function f with additional arguments
#     return y - 39399.01705150698  # Constraint: y must be equal to y_0

# # Define the bounds for the variables
# lb = []
# ub = []
# # lb.append(np.ones(64)*10)
# lb.append(np.zeros(64))
# ub.append(np.ones(64)*500)
# bounds = (np.concatenate(lb),np.concatenate(ub))

bounds = [(0.0,500.0)]*64

# Initial guess for x (optional)
# initial_guess = np.ones(64)*45  # You can provide your own initial guess
initial_guess = helper.carb_price # You can provide your own initial guess

# Constraint: y must be equal to y_0
# constraint = {'type': 'eq', 'fun': s.constraint_from_carb_price}
constraint = {'type': 'eq',
       'fun': s.constraint_from_carb_price,
       'args': (params, baseline, helper)       
       }

# Additional arguments for f
# args =   # Specify your additional arguments here

# Use the minimize function to find the optimal x
result = minimize(s.to_minimize_under_constraint_from_carb_price, 
                  initial_guess, 
                  args=(params, baseline, helper), 
                  bounds=bounds, 
                  constraints=constraint,
                  options={'disp':2})

# The optimized x vector will be in result.x
optimized_x = result.x
# optimized_v, optimized_y = s.to_minimize_under_constraint_from_carb_price(optimized_x, *args)

# print("Optimal x:", optimized_x)
# print("Optimal v:", optimized_v)
# print("Optimal y:", optimized_y)



#%%
baseline_scaled = baseline.copy()
baseline_scaled = d.baseline(2018, data_path)

#%%
import treatment_funcs as t
import seaborn as sns
import matplotlib.pyplot as plt


# baseline_scaled.num_scale('wage', 'WLD', inplace = True)

# baseline_scaled.co2_intensity = baseline_scaled.co2_intensity*1e6
# baseline_scaled.co2_intensity_np = baseline_scaled.co2_intensity_np*1e6

va = baseline.va.groupby('col_country').sum()
labor = baseline.labor.set_index('country').rename_axis('col_country')['2018'].to_frame()
labor.columns = ['value']

gdp_per = va/labor

imf_run = pd.read_csv('results/2018_101/runs.csv').iloc[0]
imf_sol = t.sol(imf_run,results_path,data_path)
imf_sol.compute_solution(baseline_scaled,inplace=True)

uniform_tax_runs = pd.read_csv('results/2018_50/runs.csv')
eq_uniform_tax_run = uniform_tax_runs.loc[np.argmin(np.abs(uniform_tax_runs.emissions-imf_run.emissions))]

eq_uniform_tax_sol = t.sol(eq_uniform_tax_run,results_path,data_path)
eq_uniform_tax_sol.compute_solution(baseline_scaled,inplace=True)

fig,ax = plt.subplots(figsize=(16,12),dpi=288)

# ax.scatter(gdp_per.value,eq_uniform_tax_sol.utility*100-100,label='Uniform tax')
ax.scatter(gdp_per.value,
           imf_sol.params.carb_cost_df.groupby('col_country')['value'].mean()*1e6,
           label='IMF tax')
ax.scatter(gdp_per.value,opt_cab_price.x,label='Custom tax')

ax.set_yscale('symlog',linthresh=10)

texts = [ax.text(gdp_per.loc[country,'value'], 
        opt_cab_price.x[i], 
        country,
        size=14,
        # color=colors[g],
        rotation = 0,ha='center',va='top') 
        for i,country in enumerate(gdp_per.index)]

# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))

ax.set_ylabel('Real income change (%)')
ax.set_xlabel('GDP per capita (Mio. US$)')

plt.legend()
plt.show()

#%%
import treatment_funcs as t
import seaborn as sns
import matplotlib.pyplot as plt

# baseline_scaled = baseline.copy()
# baseline_scaled = d.baseline(2018, data_path)
# baseline_scaled.num_scale('wage', 'WLD', inplace = True)

# baseline_scaled.co2_intensity = baseline_scaled.co2_intensity*1e6
# baseline_scaled.co2_intensity_np = baseline_scaled.co2_intensity_np*1e6

va = baseline.va.groupby('col_country').sum()
labor = baseline.labor.set_index('country').rename_axis('col_country')['2018'].to_frame()
labor.columns = ['value']

gdp_per = va/labor

imf_run = pd.read_csv('results/2018_101/runs.csv').iloc[0]
imf_sol = t.sol(imf_run,results_path,data_path)
imf_sol.compute_solution(baseline_scaled,inplace=True)

uniform_tax_runs = pd.read_csv('results/2018_50/runs.csv')
eq_uniform_tax_run = uniform_tax_runs.loc[np.argmin(np.abs(uniform_tax_runs.emissions-imf_run.emissions))]

eq_uniform_tax_sol = t.sol(eq_uniform_tax_run,results_path,data_path)
eq_uniform_tax_sol.compute_solution(baseline_scaled,inplace=True)

fig,ax = plt.subplots(figsize=(16,12),dpi=288)

ax.scatter(helper.carb_price,eq_uniform_tax_sol.utility*100-100,label='Uniform tax')
ax.scatter(helper.carb_price,imf_sol.utility*100-100,label='IMF tax')
ax.scatter(helper.carb_price,helper.utility_countries*100-100,label='Custom tax')

texts = [ax.text(helper.carb_price[i], 
        imf_sol.utility.loc[country,'hat']*100-100, 
        country,
        size=14,
        # color=colors[g],
        rotation = 0,ha='center',va='top') 
        for i,country in enumerate(gdp_per.index)]

# ax.set_xscale('symlog',linthresh=1)
ax.set_xscale('symlog',linthresh=10)

# ax.axhline(y=eq_uniform_tax_run.utility*100-100,color=sns.color_palette()[0],label='Global real income change\nUniform tax')
# ax.axhline(y=imf_run.utility*100-100,color=sns.color_palette()[1],label='Global real income change\nIMF tax')
# ax.axhline(y=imf_run.utility*100-100,color=sns.color_palette()[1],label='Global real income change\nIMF tax')

# adjust_text(texts, precision=0.001,
#         expand_text=(1.01, 1.05), expand_points=(1.01, 1.05),
#         force_text=(0.01, 0.25), force_points=(0.01, 0.25),
#         arrowprops=dict(arrowstyle='-', color='k'#, alpha=.5
#                         ))

ax.set_ylabel('Real income change (%)')
ax.set_xlabel('Carbon price of the custom tax ($ per tCO2eq)')

plt.legend()
plt.show()