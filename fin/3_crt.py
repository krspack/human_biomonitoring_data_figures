#!/usr/bin/env python
# coding: utf-8

# ## Kreatinin - vzorky nad/pod limitem, rozdíl F × M  

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import pickle
from pandas.api.types import CategoricalDtype
import seaborn as sns
import numpy as np
import math
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols

df = pd.read_pickle('df_no_smokers.pkl')

with open('pah_columns.pkl', 'rb') as f:  
    pah_columns = pickle.load(f)

with open('parent_to_metabolite.pkl', 'rb') as f:  
    parent_to_metabolite = pickle.load(f)

with open('metabolites_colors.pkl', 'rb') as f:  
    metabolites_colors = pickle.load(f)

with open('parents_colors.pkl', 'rb') as f:  
    parents_colors = pickle.load(f)

def get_parent(compound):
    for parent, metabolites in parent_to_metabolite.items():
        for m in metabolites:
            if compound == m:
                return parent
    return ''


# In[2]:


# vzorky nad a pod limitem byly odstraneny v ramci cisteni dat, vi 00_clean_data.ipynb

# kreatinin F x M
df['crt_g_l_log10'] = np.log10(df['crt'] / 1000000)   # prevedeni na jednotku g/l, log

# plot
fig = plt.figure(figsize=[5, 6])
ax = plt.gca()
colors = {'M':'lightblue', 'F': 'lightpink'}
sns.boxplot(x='sex', y='crt_g_l_log10', hue = 'sex', data=df, whis=(5, 95), palette = colors)
plt.title('Concentration of CRT x sex', pad=30)

# n
counts = df.groupby('sex', observed = False)['crt_g_l_log10'].size()

# t-test
f_values = df[df['sex'] == 'F']['crt_g_l_log10']
m_values = df[df['sex'] == 'M']['crt_g_l_log10']
t_stat, p_value = ttest_ind(f_values, m_values)

# x-axis
xticks = ax.get_xticks()
ax.set_xticks(xticks)
ax.set_xticklabels(f'{item}\n n = {counts.get(item)}' for item in counts.index)
ax.set_xlabel(f'p-value:  {p_value}')

# y-axis in linear units
ax.set_ylabel('CRT in g/l, log10-transformed values')
yticks = ax.get_yticks()
ax.set_yticks(yticks)
yticks_linear = [np.exp(y) for y in yticks]  
ax.set_yticklabels([str(round(value, 2)) for value in yticks_linear])

plt.show()


