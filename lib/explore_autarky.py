#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:19:35 2023

@author: slepot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline, BSpline

df = pd.read_csv('results/2018_80/runs.csv',index_col=0)

df_with_tax = df.loc[np.isclose(df.carb_cost,1e-4)]
df_without_tax = df.loc[np.isclose(df.carb_cost,0)]

def smooth_out(x,y):
    xnew = np.linspace(x.min(), x.max(), 300) 
    spl = make_interp_spline(x, y, k=3)  # type: BSpline
    power_smooth = spl(xnew)
    return xnew, power_smooth

fig,ax = plt.subplots(figsize = (12,8),dpi=288)

ax2 = ax.twinx()

# ax.plot(df_without_tax['tau_factor'],df_without_tax['emissions'],label='Emissions without tax')
ax.plot(*smooth_out(df_without_tax['tau_factor'],df_without_tax['emissions']),label='Emissions without tax')
ax.plot(*smooth_out(df_with_tax['tau_factor'],df_with_tax['emissions']),label='Emissions with tax')

# ax.plot(df_with_tax['tau_factor'],df_with_tax['emissions'],label='Emissions with tax')

ax2.plot(*smooth_out(df_without_tax['tau_factor'],df_without_tax['utility']*100-100),label='Welfare without tax',ls='--')
ax2.plot(*smooth_out(df_with_tax['tau_factor'],df_with_tax['utility']*100-100),label='Welfare with tax',ls='--')

ax.legend(loc=[-0.3,0.5])
ax2.legend(loc=[1.1,0.5])

ax.set_xlabel('Proportional change in trade costs')
ax.set_ylabel('Emissions')
ax2.set_ylabel('Welfare change (%)')

# ax.hlines(xmin=1,xmax=5,y=emissions_sol)
# ax2.hlines(xmin=1,xmax=5,y=utility*100-100,
#            color=sns.color_palette()[1])

plt.show()

#%%
n = 9
colors = sns.diverging_palette(145, 20, center="dark", n=n)

def smooth_out(x,y):
    xnew = np.linspace(x.min(), x.max(), 300) 
    spl = make_interp_spline(x, y, k=3)  # type: BSpline
    power_smooth = spl(xnew)
    return xnew, power_smooth

fig,ax = plt.subplots(2,1,figsize = (12,12),dpi=288)
# ax2 = ax.twinx()

for i,tau in enumerate(np.linspace(1,5,n)):
    
    df = pd.read_csv('results/2018_90/runs.csv',index_col=0)
    df = df.loc[df.tau_factor == tau]
    
    
    
    # ax.plot(df_without_tax['tau_factor'],df_without_tax['emissions'],label='Emissions without tax')
    ax[0].plot(*smooth_out(df['carb_cost']*1e6,df['emissions']),label=tau,c=colors[i])
    # ax.plot(*smooth_out(df_with_tax['tau_factor'],df_with_tax['emissions']),label='Emissions with tax')
    
    # ax.plot(df_with_tax['tau_factor'],df_with_tax['emissions'],label='Emissions with tax')
    
    ax[1].plot(*smooth_out(df['carb_cost']*1e6,df['utility']*100-100),label='Welfare',ls='--',c=colors[i])
    # ax2.plot(*smooth_out(df_with_tax['tau_factor'],df_with_tax['utility']*100-100),label='Welfare with tax',ls='--')

ax[0].legend(loc=[-0.3,0.1],title='Proportional change\nin trade costs')
# ax2.legend(loc=[1.1,0.5])

ax[1].set_xlabel('Carbon price')
ax[0].set_ylabel('Emissions')
ax[1].set_ylabel('Welfare change\nRelative to baseline (%)')
# ax2.set_ylabel('Welfare change (%)')
ax[0].grid()
ax[1].grid()
# ax.hlines(xmin=1,xmax=5,y=emissions_sol)
# ax2.hlines(xmin=1,xmax=5,y=utility*100-100,
#            color=sns.color_palette()[1])

plt.show()

#%%

fig,ax = plt.subplots(figsize = (12,8),dpi=288)

ax2 = ax.twinx()

ax.plot(df_without_tax['tau_factor'],(df_with_tax['emissions'].values*100/df_without_tax['emissions'].values-100),
        label = 'Emissions reduction due to the tax')

ax2.plot(df_without_tax['tau_factor'],(df_with_tax['utility'].values*100/df_without_tax['utility'].values-100),
        label = 'Welfare loss due to the tax',
        color=sns.color_palette()[1])

ax.legend()
ax2.legend(loc='center right')

ax.set_xlabel('Proportional change in trade costs')
ax.set_ylabel('Emissions')
ax2.set_ylabel('Welfare change (%)')

plt.show()