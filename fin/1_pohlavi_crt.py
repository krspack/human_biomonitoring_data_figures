#!/usr/bin/env python
# coding: utf-8

# ## Sex + PAHs (CRT-standardized values)

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import pickle
from pandas.api.types import CategoricalDtype
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind

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


# pohlaví × koncentrace pro všechny PAHy 

fig, axes = plt.subplots(4, 4, figsize=(20, 18))
fig.suptitle('Metabolites vs. sex. Smokers excluded', fontsize=16, y = 0.99)

for idx, substance in enumerate(pah_columns):

    row = idx // 4
    col = idx % 4

    # n
    grouped = df.groupby('sex', observed = False)[[substance + '_impcrt', substance + '_impcrtlog10']]
    grouped_dropna = grouped.apply(lambda x: x.dropna()).reset_index()
    counts = grouped_dropna.groupby('sex', observed = False).size()

    # t-test
    f_values = grouped_dropna[grouped_dropna['sex'] == 'F'][substance+'_impcrtlog10']
    m_values = grouped_dropna[grouped_dropna['sex'] == 'M'][substance+'_impcrtlog10']
    t_stat, p_value = ttest_ind(f_values, m_values)
    p_value = np.format_float_scientific(p_value, precision=2)
    
    # Boxplot
    y = df[substance + '_impcrt'].dropna() 
    sns.boxplot(x = df['sex'], y = y, color = metabolites_colors.get(substance), ax=axes[row, col], whis = (5, 95))

    # x-axis
    xticks = axes[row, col].get_xticks()
    axes[row, col].set_xticks(xticks)
    axes[row, col].set_xticklabels([f'{item}\n n = {counts.get(item)}' for item in counts.index])
    axes[row, col].set_xlabel(f'p-value:  {p_value}')
    
    # y-axis:
    axes[row, col].set_yscale('log') 
    axes[row, col].yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    axes[row, col].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.3f}" if x < 1 else f"{int(x)}"))
    axes[row, col].set_ylabel("CRT-standardized, imputed, log10-transfrmed values")

    # title
    axes[row, col].set_title(f"{substance}")

# Adjust layout
plt.tight_layout()
plt.show()


# In[ ]:




