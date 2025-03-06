#!/usr/bin/env python
# coding: utf-8

# ## Kreatin - vzorky nad/pod limitem, rozdíl F × M  

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import numpy as np
import math
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols

# načtení dat, volba typů proměnných
df = pd.read_csv('data/data_copy.csv', sep=";", encoding="UTF-8", decimal=',')
data_types = pd.read_csv('dtypes_csv2.csv', sep = ',', encoding = 'UTF-8')   # data_types = muj výtvor, lze menit
dtypes_dict = data_types.to_dict(orient = "records")[0]
for col, dtype in dtypes_dict.items():
    df[col] = df[col].astype(dtype)

# sledovane latky - vybrane sloupce
pah_columns = ['oneohnap', 'twoohnap', 'diohnap', 'twohfluo', 'threehfluo', 'ninehfluo', 'onehphe', 
     'twohphe', 'threehphe', 'fourhphe', 'ninehphe', 'ohpyr', 'ohbap']
log_columns = [x+'_log' for x in pah_columns]

imp_columns = [x+'_imp' for x in pah_columns]
implog_columns = [x+'_implog' for x in pah_columns]

impcrt_columns = [x+'_impcrt' for x in pah_columns]
impcrtlog_columns = [x for x in df.columns if 'impcrtlog' in x]

# CRT: vyhodit chybejici data a vzorky nad a pod limitem (mene nez 10 radek)
df.dropna(subset=['crt'], axis=0, inplace=True)   # maže 5 řádek
df = df.assign(crt_g_l = df['crt']/1000000, crt_limity = '')
df['crt_limity'] = df['crt_g_l'].apply(lambda x: 'pod limitem' if x < 0.05 else ('ok' if 0.05 <= x <= 5 else 'nad limitem'))
# df = df[df['crt_limity'] == 'ok']  # maže 6 řádek

# kuřáci pryč
df = df[df['smoking'] != True] # maže 443 řádek

seznam_latek_podle_rodice = {
    'PYR': ['ohpyr'], 
    'FLU': ['twohfluo', 'threehfluo', 'ninehfluo'],
    'PHE': ['onehphe', 'twohphe', 'threehphe', 'fourhphe', 'ninehphe'],
    'NAP': ['oneohnap', 'twoohnap', 'diohnap'], 
    'BAP': ['ohbap']
}

metabolites_colors = {
    "oneohnap": "#1f7cb4",       # 1-hydroxynaphthalene
    "twoohnap": "#A5CEE2",      # 2-hydroxynaphthalene        # opraven kod
    "diohnap": "#5BDBC3",       # 1-,2-dihydroxynaphthalene   # chybi, doplneno
    "twohfluo": "#ffa700",      # 2-hydroxyfluorene
    "threehfluo": "#fbf356",    # 3-hydroxyfluorene
    "ninehfluo": "#FCBF64",     # 9-hydroxyfluorene           # chybi, doplneno 
    "onehphe": "#6d8b3c",       # 1-hydroxyphenanthrene
    "twohphe": "#bce96c",       # 2-hydroxyphenanthrene
    "threehphe": "#6CC92A",     # 3-hydroxyphenanthrene       # chybi, doplneno
    "fourhphe": "#33a32a",      # 4-hydroxyphenanthrene
    "ninehphe": "#b5e9b4",      # 9-hydroxyphenanthrene
    "ohpyr": "#cab2d6",         # 1-hydroxypyrene  
    "ohbap": "#f07075"          # 3-hydroxybenzo(a)pyrene
}

parent_colors = {
    'PYR': '#e9843f', 
    'FLU': '#f9c013',
    'PHE': '#abcf93', 
    'NAP': '#659fd3',
    'BAP': '#E21A3F'   # opraven kod
}

def get_parent(compound):
    for parent, metabolites in parent_to_metabolite.items():
        for m in metabolites:
            if compound == m:
                return parent
    return ''


# In[2]:


# vzorky nad a pod limitem pro normalni CRT:

df['crt_g_l'] = df['crt']/1000000
df['crt_limity'] = ''
df['crt_limity'] = df['crt_g_l'].apply(lambda x: 'pod limitem' if x < 0.05 else ('ok' if 0.05 <= x <= 5 else 'nad limitem'))

print("Počet vzorků pod limitem 0.05 g/l nad limitem 5 g/l:")
print("====================================================")
print(df.groupby('crt_limity').size())


# kreatin F x M
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


