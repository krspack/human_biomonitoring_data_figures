#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind

# načtení dat, volba typů proměnných
df = pd.read_csv('data/data_copy.csv', sep=";", encoding="UTF-8", decimal=',')
data_types = pd.read_csv('dtypes_csv2.csv', sep = ',', encoding = 'UTF-8')   # data_types = muj výtvor, lze menit
dtypes_dict = data_types.to_dict(orient = "records")[0]
for col, dtype in dtypes_dict.items():
    df[col] = df[col].astype(dtype)

# sledovane latky - vybrane sloupce
pah_columns = ['oneohnap', 'twoohnap', 'diohnap', 'twohfluo', 'threehfluo', 'ninehfluo', 'onehphe', 
     'twohphe', 'threehphe', 'fourhphe', 'ninehphe', 'ohpyr', 'ohbap']    
pah_12_columns = ['oneohnap', 'twoohnap', 'twohfluo', 'threehfluo', 'ninehfluo', 'onehphe', 
     'twohphe', 'threehphe', 'fourhphe', 'ninehphe', 'ohpyr', 'ohbap']   # bez diohnap
log_columns = [x+'_log' for x in pah_columns]

imp_columns = [x+'_imp' for x in pah_columns]
implog_columns = [x+'_implog' for x in pah_columns]

impcrt_columns = [x+'_impcrt' for x in pah_columns]
impcrtlog_columns = [x for x in df.columns if 'impcrtlog' in x]

# CRT: vyhodit chybejici data a vzorky nad a pod limitem
df.dropna(subset=['crt'], axis=0, inplace=True)   # maže 5 řádek
df = df.assign(crt_g_l = df['crt']/1000000, crt_limity = '')
df['crt_limity'] = df['crt_g_l'].apply(lambda x: 'pod limitem' if x < 0.05 else ('ok' if 0.05 <= x <= 5 else 'nad limitem'))
df = df[df['crt_limity'] == 'ok']  # maže 6 řádek

# kuřáci pryč
df = df[df['smoking'] != True] # # maže 443 řádek

# doplnit do tabulky log10
for substance in pah_columns:
    df[substance+'_impcrtlog10'] = np.log10(df[substance+'_impcrt']) 
    df[substance+'_impcrtlog10'] = pd.to_numeric(df[substance+'_impcrtlog10'], errors='coerce')

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


# pohlaví × koncentrace pro všechny PAHy 

fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle('Metabolites vs. sex. Smokers excluded', fontsize=16)

for idx, substance in enumerate(pah_columns[:12]):

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

