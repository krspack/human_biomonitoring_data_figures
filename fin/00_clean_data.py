#!/usr/bin/env python
# coding: utf-8

# ## Load and clean data, compute additional variables for later use
# 
# Vystup tohoto kodu (vycistena a dopocitana data pro analyzy) je ulozen zde:
# 
# * df_incl_smokers.pkl = data s kuraky 
# * df_no_smokers.pkl = bez kuraku
# * df_fillna_medians.pkl = bez kuraku, chybejici hodnoty u metabolitu jsou nahrazeny medianem daneho metabolitu. Bez diohnap a ninehfluo
# 
# * pah_columns.pkl
# * parent_to_metabolite.pkl = slovnik parent: metabolites
# * metabolites_colors.pkl
# * parents_colors.pkl
#   

# In[1]:


import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
import pandas as pd
import pickle
from pandas.api.types import CategoricalDtype
import seaborn as sns
import numpy as np
import scipy.stats as stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statannotations.Annotator import Annotator

# načtení dat, volba typů proměnných
df = pd.read_csv('data/data_copy.csv', sep=";", encoding="UTF-8", decimal=',')
data_types = pd.read_csv('dtypes_csv2.csv', sep = ',', encoding = 'UTF-8')   # data_types = muj výtvor, lze menit
dtypes_dict = data_types.to_dict(orient = "records")[0]
for col, dtype in dtypes_dict.items():
    df[col] = df[col].astype(dtype)

# CRT: vyhodit chybejici data a vzorky nad a pod limitem
df.dropna(subset=['crt'], axis=0, inplace=True)   # maže 5 řádek
df = df.assign(crt_g_l = df['crt']/1000000, crt_limity = '')
df['crt_limity'] = df['crt_g_l'].apply(lambda x: 'pod limitem' if x < 0.05 else ('ok' if 0.05 <= x <= 5 else 'nad limitem'))
df = df[df['crt_limity'] == 'ok']  # maže 6 řádek

# sledovane latky - vybrane sloupce
pah_columns = ['oneohnap', 'twoohnap', 'diohnap', 'twohfluo', 'threehfluo', 'ninehfluo', 'onehphe', 
     'twohphe', 'threehphe', 'fourhphe', 'ninehphe', 'ohpyr', 'ohbap']

with open('pah_columns.pkl', 'wb') as file:
    pickle.dump(pah_columns, file)
    
pah_11_columns = ['oneohnap', 'twoohnap', 'twohfluo', 'threehfluo', 'onehphe', 
     'twohphe', 'threehphe', 'fourhphe', 'ninehphe', 'ohpyr', 'ohbap']  # without diohnap and ninehfluo

with open('pah_11_columns.pkl', 'wb') as file:
    pickle.dump(pah_11_columns, file)
    
log_columns = [x+'_log' for x in pah_columns]
log_11_columns = [x+'_log' for x in pah_11_columns]

imp_columns = [x+'_imp' for x in pah_columns]
imp_11_columns = [x+'_imp' for x in pah_11_columns]
implog_columns = [x+'_implog' for x in pah_columns]
implog_11_columns = [x+'_implog' for x in pah_11_columns]

impcrt_columns = [x+'_impcrt' for x in pah_columns]
impcrt_11_columns = [x+'_impcrt' for x in pah_11_columns]
impcrtlog_columns = [x for x in df.columns if 'impcrtlog' in x]
impcrtlog_11_columns = [x for x in df.columns if ('impcrtlog' in x and x not in ['diohnap_impcrtlog', 'ninehfluo_impcrtlog'])]

# dopocitat imputaci pro ohbap
df['ohbap_imp'] = np.where(
    df['ohbap'] == -1,
    df['ohbap_lod']/np.sqrt(2),
    np.where(
        df['ohbap'].isin([-2, -3]),
        df['ohbap_loq']/np.sqrt(2),
        df['ohbap']
    )
)
df['ohbap_impcrt'] = df['ohbap_imp'] / df['crt_g_l']

# pokud je nekde chybne namerena nula, nahradit ji, jako by to byla -1, tj. pod limitem detekce
for column in pah_columns:
    df[column] = np.where(df[column] == 0, -1, df[column])
    df[f'{column}_imp'] = np.where(df[f'{column}_imp'] == 0, df[column+'_lod']/np.sqrt(2), df[f'{column}_imp'])
    df[f'{column}_impcrt'] = np.where(df[f'{column}_impcrt'] == 0, df[column+'_imp'] / df['crt_g_l'], df[f'{column}_impcrt'])

# doplnit do tabulky _implog10 a _impcrtlog10
new_columns = {f'{substance}_implog10': np.log10(df[f'{substance}_imp']) for substance in pah_columns}  # separatne vyrobit nove sloupce
df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)  # pripojit je
for column in new_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

new_columns = {f'{substance}_impcrtlog10': np.log10(df[f'{substance}_impcrt']) for substance in pah_columns}  
df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)  # pripojit je
for column in new_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# doplnit sloupec heating_season
heating_season = {
"CH": [10,11,12,1,2,3,4],
"CZ": [10,11,12,1,2,3,4],
"DE": [10,11,12,1,2,3,4],
"DK": [9,10,11,12,1,2,3,4],
"FR": [10,11,12,1,2,3,4],
"HR": [11,12,1,2,3],
"IS": [9,10,11,12,1,2,3,4,5],
"LU": [10,11,12,1,2,3,4],
"PL": [9,10,11,12,1,2,3,4],
"PT": [11,12,1,2,3]
}
df['heating_season'] = df.apply(
    lambda row: int(row['samplingmonth']) in heating_season[row['country']], axis=1)

df_no_parents = df.copy()  # pro pozdejsi pouziti


# In[2]:


parent_to_metabolite = {
    'PYR': ['ohpyr'], 
    'FLU': ['twohfluo', 'threehfluo', 'ninehfluo'],
    'PHE': ['onehphe', 'twohphe', 'threehphe', 'fourhphe', 'ninehphe'],
    'NAP': ['oneohnap', 'twoohnap', 'diohnap'], 
    'BAP': ['ohbap']
}
with open('parent_to_metabolite.pkl', 'wb') as file:
    pickle.dump(parent_to_metabolite, file)

parent_to_metabolite_11= {
    'PYR': ['ohpyr'], 
    'FLU': ['twohfluo', 'threehfluo'],
    'PHE': ['onehphe', 'twohphe', 'threehphe', 'fourhphe', 'ninehphe'],
    'NAP': ['oneohnap', 'twoohnap'], 
    'BAP': ['ohbap']
}
with open('parent_to_metabolite_11.pkl', 'wb') as file:
    pickle.dump(parent_to_metabolite_11, file)

metabolites_colors = {
    "oneohnap": "#1f7cb4",      # 1-hydroxynaphthalene
    "twoohnap": "#A5CEE2",      # 2-hydroxynaphthalene        
    "diohnap": "#5BDBC3",       # 1-,2-dihydroxynaphthalene   
    "twohfluo": "#ffa700",      # 2-hydroxyfluorene
    "threehfluo": "#fbf356",    # 3-hydroxyfluorene
    "ninehfluo": "#FCBF64",     # 9-hydroxyfluorene           
    "onehphe": "#6d8b3c",       # 1-hydroxyphenanthrene
    "twohphe": "#bce96c",       # 2-hydroxyphenanthrene
    "threehphe": "#6CC92A",     # 3-hydroxyphenanthrene       
    "fourhphe": "#33a32a",      # 4-hydroxyphenanthrene
    "ninehphe": "#b5e9b4",      # 9-hydroxyphenanthrene
    "ohpyr": "#cab2d6",         # 1-hydroxypyrene  
    "ohbap": "#f07075"          # 3-hydroxybenzo(a)pyrene
}
with open('metabolites_colors.pkl', 'wb') as file:
    pickle.dump(metabolites_colors, file)

parents_colors = {
    'PYR': '#e9843f', 
    'FLU': '#f9c013',
    'PHE': '#abcf93', 
    'NAP': '#659fd3',
    'BAP': '#E21A3F'   
}
with open('parents_colors.pkl', 'wb') as file:
    pickle.dump(parents_colors, file)

def get_parent(compound):
    for parent, metabolites in parent_to_metabolite.items():
        for m in metabolites:
            if compound == m:
                return parent
    return ''

# Spocitat soucty koncentraci pro parent compounds
for parent, metabolites in parent_to_metabolite.items():

    for metab in metabolites:
        df[metab] = pd.to_numeric(df[metab], errors='coerce')
    df[f'{parent}'] = df[metabolites].sum(axis=1)
       
    # _imp
    metabolites_imp = [f'{metab}_imp' for metab in metabolites]
    for metab in metabolites_imp:
        df[metab] = pd.to_numeric(df[metab], errors='coerce')
    df[f'{parent}_imp'] = df[metabolites_imp].sum(axis=1)

    # _impcrt
    metabolites_impcrt = [f'{metab}_impcrt' for metab in metabolites]
    for metab in metabolites_impcrt:
        df[metab] = pd.to_numeric(df[metab], errors='coerce')
    df[f'{parent}_impcrt'] = df[metabolites_impcrt].sum(axis=1)
    
    # log10 columns
    df[f'{parent}_implog10'] = np.log10(df[f'{parent}_imp'].replace(0, np.nan))
    df[f'{parent}_impcrtlog10'] = np.log10(df[f'{parent}_impcrt'].replace(0, np.nan))


# In[3]:


# uložit tabulku i s kuřáky
df.to_pickle('df_incl_smokers.pkl')

# uložit bez kuřáků
df = df[df['smoking'] != True] # maže 443 řádek
df.to_pickle('df_no_smokers.pkl')


# In[4]:


# kontroly
# hodnota impcrt by mela byt radove stejna jako df[imp]/df[crt_g_l]
try:
    # vyradit ze srovnani chybejici hodnoty
    df_assertion = df[df[imp_columns[0]].notna()].copy()

    # dve np-arrays maji mit priblizne stejne hodnoty. Porovnavam zaohrouhlena cisla.
    assert np.allclose(
        np.round(df_assertion[impcrt_columns[0]], 0), 
        np.round(df_assertion[imp_columns[0]] / df_assertion['crt_g_l'], 0)
    ), "Hodnota impcrt nesedi."
except AssertionError:
    # u kterych radek assertion nesedi:
    mismatch = df_assertion[
        ~np.isclose(
            np.round(df_assertion[impcrt_columns[0]], 0), 
            np.round(df_assertion[imp_columns[0]] / df_assertion['crt_g_l'], 0)
        )
    ][[impcrt_columns[0], imp_columns[0], 'crt_g_l']]
    # vytisknout mismatch
    raise ValueError(f"Problemove radky:\n{mismatch}")


# In[5]:


# verze tabulky pro profily statu: pred sectenim parent compounds se chybejici hodnoty nahradi medianem daneho sloupce 
# bez diohnap a ninehfluo

# chybejici hodnoty nahradit medianem
for column in pah_11_columns:
    df_no_parents[column] = df_no_parents[column].fillna(df_no_parents[column].median())
    df_no_parents[f'{column}_imp'] = df_no_parents[f'{column}_imp'].fillna(df_no_parents[f'{column}_imp'].median())
    df_no_parents[f'{column}_implog10'] = np.log10(df_no_parents[f'{column}_imp'])
    
    df_no_parents[f'{column}_impcrt'] = df_no_parents[f'{column}_impcrt'].fillna(df_no_parents[f'{column}_impcrt'].median())
    df_no_parents[f'{column}_impcrtlog10'] = np.log10(df_no_parents[f'{column}_impcrt'])
 
# Spocitat soucty koncentraci pro parent compounds
for parent, metabolites in parent_to_metabolite_11.items():
    
    # _imp
    metabolites_imp = [f'{metab}_imp' for metab in metabolites]
    for metab in metabolites_imp:
        df_no_parents[metab] = pd.to_numeric(df_no_parents[metab])
    df_no_parents[f'{parent}_imp'] = df_no_parents[metabolites_imp].sum(axis=1)

    # _impcrt
    metabolites_impcrt = [f'{metab}_impcrt' for metab in metabolites]
    for metab in metabolites_impcrt:
        df_no_parents[metab] = pd.to_numeric(df_no_parents[metab])
    df_no_parents[f'{parent}_impcrt'] = df_no_parents[metabolites_impcrt].sum(axis=1)
    
    # log10 columns
    df_no_parents[f'{parent}_implog10'] = np.log10(df[f'{parent}_imp'].replace(0, np.nan))
    df_no_parents[f'{parent}_impcrtlog10'] = np.log10(df[f'{parent}_impcrt'].replace(0, np.nan))
    df_no_parents = df_no_parents.copy()  # defragmentace
    
# ulozit pro pouziti v country profiles
df_fillna_medians = df_no_parents.copy()
df_fillna_medians.to_pickle('df_fillna_medians.pkl')


# In[6]:


# kontroly: zadne chybejici hodnoty u 11 metabolitu v tabulce fillna_medians
for col in pah_11_columns:
    assert df_fillna_medians[col].isna().sum() == 0, f"Chyba: Ve sloupci '{col}' chybi hodnoty."
    assert df_fillna_medians[f'{col}_imp'].isna().sum() == 0, f"Chyba: Ve sloupci '{col}_imp' chybi hodnoty."
    assert df_fillna_medians[f'{col}_impcrt'].isna().sum() == 0, f"Chyba: Ve sloupci '{col}_impcrt' chybi hodnoty."

# kontrola: zadne chybejici hodnoty u parents
for parent in parent_to_metabolite.keys():
    assert df_fillna_medians[parent+'_imp'].isna().sum() == 0, f"Chyba: Ve sloupci '{parent}_imp' chybi hodnoty."
    assert df_fillna_medians[parent+'_impcrt'].isna().sum() == 0, f"Chyba: Ve sloupci '{parent}_impcrt' chybi hodnoty."


# kontrola: parent sloupce jsou v tabulce
parent_imp_columns = [f'{p}_imp' for p in parent_to_metabolite.keys()]
parent_implog10_columns = [f'{p}_implog10' for p in parent_to_metabolite.keys()]
parent_impcrt_columns = [f'{p}_impcrt' for p in parent_to_metabolite.keys()]
parent_impcrtlog10_columns = [f'{p}_impcrtlog10' for p in parent_to_metabolite.keys()]
assert set(parent_impcrt_columns).issubset(df_fillna_medians.columns), "Chyba: Parent imp columns nejsou mezi sloupci tabulky."
assert set(parent_implog10_columns).issubset(df_fillna_medians.columns), "Chyba: Parent implog10 columns nejsou mezi sloupci tabulky."
assert set(parent_impcrt_columns).issubset(df_fillna_medians.columns), "Chyba: Parent impcrt columns nejsou mezi sloupci tabulky."
assert set(parent_impcrtlog10_columns).issubset(df_fillna_medians.columns), "Chyba: Parent impcrtlog10 columns nejsou mezi sloupci tabulky."

# kontrola: imp a impcrt maji jen kladne hodnoty
for c in parent_imp_columns:
    assert (df_fillna_medians[c] > 0).all(), f'chyba: nulove nebo zaporne hodnoty ve sloupci {c}'
    
for c in parent_impcrt_columns:
    assert (df_fillna_medians[c] > 0).all(), f'chyba: nulove nebo zaporne hodnoty ve sloupci {c}'


# In[ ]:




