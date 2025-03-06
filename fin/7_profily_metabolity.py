#!/usr/bin/env python
# coding: utf-8

# ## Profily - metabolity

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import numpy as np
from scipy.stats import f
from hotelling.stats import hotelling_t2

# načtení dat, volba typů proměnných
df = pd.read_csv('data/data_copy.csv', sep=";", encoding="UTF-8", decimal=',')
data_types = pd.read_csv('dtypes_csv2.csv', sep = ',', encoding = 'UTF-8')   # data_types = muj výtvor, lze menit
dtypes_dict = data_types.to_dict(orient = "records")[0]
for col, dtype in dtypes_dict.items():
    df[col] = df[col].astype(dtype)

# sledovane latky - vybrane sloupce
pah_columns = ['oneohnap', 'twoohnap', 'diohnap', 'twohfluo', 'threehfluo', 'ninehfluo', 'onehphe', 
     'twohphe', 'threehphe', 'fourhphe', 'ninehphe', 'ohpyr', 'ohbap']
pah_11_columns = ['oneohnap', 'twoohnap', 'twohfluo', 'threehfluo', 'onehphe', 
     'twohphe', 'threehphe', 'fourhphe', 'ninehphe', 'ohpyr', 'ohbap']  # without diohnap and ninehfluo
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


# CRT: vyhodit chybejici data a vzorky nad a pod limitem (mene nez 10 radek)
df.dropna(subset=['crt'], axis=0, inplace=True)   # maže 5 řádek
df = df.assign(crt_g_l = df['crt']/1000000, crt_limity = '')
df['crt_limity'] = df['crt_g_l'].apply(lambda x: 'pod limitem' if x < 0.05 else ('ok' if 0.05 <= x <= 5 else 'nad limitem'))
df = df[df['crt_limity'] == 'ok']  # maže 6 řádek

# kuřáci pryč
df = df[df['smoking'] != True] # maže 443 řádek

parent_11_compounds_dict = {
    'PYR': ['ohpyr'], 
    'FLU': ['twohfluo', 'threehfluo'],
    'PHE': ['onehphe', 'twohphe', 'threehphe', 'fourhphe', 'ninehphe'],
    'NAP': ['oneohnap', 'twoohnap'], 
    'BAP': ['ohbap']
}

metabolites_colors = {
    "oneohnap": "#1f7cb4",       # 1-hydroxynaphthalene
    "twoohnap": "#A5CEE2",      # 2-hydroxynaphthalene        # opraven kod
    "diohnap": "#5BDBC3",       # 1-,2-dihydroxynaphthalene   # chybi, doplneno ////////////
    "twohfluo": "#ffa700",      # 2-hydroxyfluorene
    "threehfluo": "#fbf356",    # 3-hydroxyfluorene
    "ninehfluo": "#FCBF64",     # 9-hydroxyfluorene           # chybi, doplneno  --------------
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

# doplnit do tabulky log10
for substance in pah_columns:
    df[substance+'_impcrtlog10'] = np.log10(df[substance+'_impcrt']) 
    df[substance+'_impcrtlog10'] = pd.to_numeric(df[substance+'_impcrtlog10'], errors='coerce')


# In[2]:


# uprava tabulky pro profily zemi. Doplneni chybejicich hodnot.
df_copy = df.copy()

missing_substance = 'ohbap'

# dopocitat imputaci pro ohbap pro ucely profilu
df_copy[f'{missing_substance}_imp'] = np.where(
    df_copy[f'{missing_substance}'] == -1,
    df_copy[f'{missing_substance}_lod']/np.sqrt(2),
    np.where(
        df_copy[f'{missing_substance}'].isin([-2, -3]),
        df_copy[f'{missing_substance}_loq']/np.sqrt(2),
        df_copy[f'{missing_substance}']
    )
)

# Dopocitat pro obhap dalsi sloupce
df_copy[f'{missing_substance}_impcrt'] = df_copy[f'{missing_substance}_imp'] / df_copy['crt_g_l']
df_copy[f'{missing_substance}_impcrtlog'] = np.log(df_copy[f'{missing_substance}_impcrt'])
df_copy[f'{missing_substance}_impcrtlog10'] = np.log10(df_copy[f'{missing_substance}_impcrt'])

# Nahradit chybejici hodnoty vsech latek medianem, nahradit nulu limitem detekce
def fillna_with_medians(dataframe: pd.DataFrame):
    for column in impcrt_11_columns:
        dataframe[column] = dataframe[column].fillna(dataframe[column].median())
        substance = column.replace('_impcrt', '')
        dataframe[column] = np.where(dataframe[column] == 0, dataframe[substance+'_lod'], dataframe[column])
    return dataframe

df_copy = fillna_with_medians(df_copy)


# In[3]:


# profily zemi

df_medians = df_copy.groupby(['country'], observed=False)[impcrt_11_columns].median()
df_medians.sort_index(axis = 'index', inplace = True)
impcrt_palette = {key+"_impcrt": value for key, value in metabolites_colors.items()}

def plot_profiles(dataframe, title, ylabel):
    fig = plt.figure(figsize = [16, 6])
    ax = plt.gca()
    
    countries = sorted(dataframe.index)
    bottom_values = np.zeros(len(countries)) 
    for i, substance in enumerate(impcrt_11_columns):
        values = dataframe.loc[countries, substance]       
        ax.bar(countries, values, bottom=bottom_values, label=substance.replace('_impcrt', ''), color=impcrt_palette[substance])
        bottom_values += values
    
    # Set x-ticks and labels
    ax.set_xticks(countries)
    
    # Add labels and title
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Countries')
    ax.set_title(title)
    
    # Add legend
    ax.legend(title="Compounds:", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Show plot
    plt.tight_layout()
    plt.show()
    return
print(plot_profiles(df_medians,
                    'Country profiles. Median values, smokers excluded, missing observations imputed with median value. Diohnap and ninehfluo are missing. Ohbap: mostly imputed values. Smokers excluded',
                   'CRT-standardized, imputed values, linear scale'))


# In[4]:


# Profily zemí II, procenta z lineárních hodnot, chybejici hodnoty nahrazeny medianem

df_medians_percentage = df_medians.div(other = df_medians.sum(axis=1), axis=0, fill_value = None) * 100
# grouped_imputed_percentage.to_csv('percentage.csv', encoding = 'UTF-8', sep = ';', decimal = ',')
print(plot_profiles(df_medians_percentage, 
                    'Country profiles - percentage of total. Missing observations imputed with median value. Diohnap and ninehfluo missing. Ohbap: mostly imputed values. Smokers excluded',
                   'CRT-standardized, imputed values, percentage of total for each country'))


# In[5]:


# Hotelliguv test

# vybrat sloupce impcrt
df_hotelling = df_copy.copy()
df_hotelling = df_hotelling[['country', *impcrt_11_columns]]

# prevest hodnoty na procenta
df_hotelling[impcrt_11_columns] = df_hotelling[impcrt_11_columns].div(other = df_hotelling[impcrt_11_columns].sum(axis=1), axis=0) * 100

# vydelit vsechny hodnoty hodnotou prvniho metabolitu
for column in impcrt_11_columns:
    df_hotelling[column] = df_hotelling[column] / df_hotelling[impcrt_11_columns[0]]
    
# zlogaritmovat vsechny hodnoty pro normalni rozdeleni
for column in impcrt_11_columns:
    df_hotelling[column] = np.log(df_hotelling[column])

# vyhodit z dat oneohnap, kterym se delilo
columns_without_oneohnap = impcrt_11_columns[1:]

# Hotellinguv test 
countries = sorted(list(df.country.unique()))
p_values_df = pd.DataFrame(index=countries, columns=countries, dtype=float)

# vyrazeni statu, co maji cele shodne sloupce dat (chybejici hodnoty nahrazene medianem). Test je nebere.
def safe_hotelling_t2(data_1, data_2):
    try:
        return hotelling_t2(data_1, data_2) 
    except np.linalg.LinAlgError:
        return None, None, np.nan  # vrati np.nan jako p-value u problematickych dvojic

# prochazeni statu a Hotellinguv test pro kazdou dvojici statu:
all_results = {}
for country_1 in countries:
    for country_2 in countries:     
        if country_1 != country_2:
            # Extract data for the two countries
            data_1 = df_hotelling[df_hotelling['country'] == country_1][columns_without_oneohnap].values
            data_2 = df_hotelling[df_hotelling['country'] == country_2][columns_without_oneohnap].values
            # Perform Hotelling T² test and unpack results
            result = safe_hotelling_t2(data_1, data_2)
        else:
            result = None, None, np.nan, None

        p_value = result[2]
        all_results[(country_1, country_2)] = result
        p_values_df.loc[country_1, country_2] = p_value

print(p_values_df)


# In[6]:


# profily zemi pro topnou a netopnou sezonu

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

df_copy['heating_season'] = df_copy.apply(
    lambda row: int(row['samplingmonth']) in heating_season[row['country']], axis=1)

df_heating = df_copy.groupby(['country', 'heating_season'], observed=False)[impcrt_11_columns].median()
df_heating.index = df_heating.index.map(lambda x: f"{x[0]} {str(x[1])}")

print('Mediánová koncentrace látky v topné vs. netopné sezóně, absolutní hodnota')
print('===========================================================================')
print(df_heating)
print(' ')

def plot_profiles(dataframe, title, ylabel):
    fig = plt.figure(figsize = [16, 6])
    ax = plt.gca()
    
    country_heating = sorted(dataframe.index)
    bottom_values = np.zeros(len(country_heating)) 
    
    for i, compound in enumerate(impcrt_11_columns):  
        values = dataframe.loc[country_heating, compound]
        ax.bar(country_heating, values, bottom=bottom_values, label=compound.replace('_impcrt', ''), color=impcrt_palette[compound])
        bottom_values += values
    
    # osy, popisky, ticks
    ax.set_xticks(dataframe.index)
    for pos in [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5]:
        ax.axvline(x=pos, color='gray', linestyle='--', linewidth=1)
        
    ax.set_xlabel('Countries and heating season. Missing observations imputed with median value. Diohnap and ninehfluo missing. BAP: mostly imputed values')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    ax.legend(title="Cmpounds:", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Show plot
    plt.tight_layout()
    plt.show()
    return
print(plot_profiles(df_heating,
                    'Country profiles, heating and non-heating season compared. Smokers excluded',
                   'Median concentrations of imputed, CRT-starndardized values, linear scale' ))


# In[7]:


# Hotelliguv test pro rozliseni topne x netopne sezony u kazdeho statu

# vybrat sloupce impcrt
df_hotelling = df_copy.copy()
df_hotelling = df_hotelling[~df_hotelling['country'].isin(['DE', 'HR', 'PL'])]
df_hotelling['heating_season'] = df_hotelling.apply(
    lambda row: int(row['samplingmonth']) in heating_season[row['country']], axis=1)

df_hotelling = df_hotelling[['country', 'heating_season', *impcrt_11_columns]]

# prevest hodnoty na procenta
row_sums = df_hotelling[impcrt_11_columns].sum(axis=1)
df_hotelling[impcrt_11_columns] = df_hotelling[impcrt_11_columns].div(row_sums, axis=0) * 100

# vydelit vsechny hodnoty hodnotou prvniho metabolitu
for column in impcrt_11_columns:
    df_hotelling[column] = df_hotelling[column] / df_hotelling[impcrt_11_columns[-1]]
    
# zlogaritmovat vsechny hodnoty pro normalni rozdeleni
for column in impcrt_11_columns:
    df_hotelling[column] = np.log(df_hotelling[column])

# vyhodit z dat posledni sloupec, kterym se delilo
columns_without_last = impcrt_11_columns[:-1]

# Hotellinguv test
countries = sorted(list(df_hotelling.country.unique()))
p_values_dict = {}

# test nebere dvojice, co maji cele shodne sloupce dat (chybejici hodnoty nahrazene medianem). Nahradit error hodnotou np.nan.
def safe_hotelling_t2(data_1, data_2):
    try:
        return hotelling_t2(data_1, data_2) 
    except np.linalg.LinAlgError:
        return None, None, np.nan  # vrati np.nan jako p-value u problematickych dvojic

# prochazeni statu a Hotellinguv test porovnavajici topnou a netopnou sezonu pro kazdy stat
all_results = {}
for country_1 in countries:     
    # Extract data for the two conditions
    data_1 = df_hotelling[
        (df_hotelling['country'] == country_1) & (df_hotelling['heating_season'] == True)
    ][columns_without_last].values
    data_2 = df_hotelling[
        (df_hotelling['country'] == country_1) & (df_hotelling['heating_season'] == False)
    ][columns_without_last].values

     # Check for empty slices or insufficient rows
    if data_1.size == 0 or data_2.size == 0:
        print(f"Skipping {country_1}: Empty data slice.")
        p_values_dict[country_1] = np.nan
        continue

    result = safe_hotelling_t2(data_1, data_2)

    p_value = result[2]
    all_results[country_1] = result
    p_values_dict[country_1] = p_value

print('Hotelling test results / p-values:')
print('==================================')
print('Heating vs. non-heating season difference in countries with samples from both seasons.')
for k, v in p_values_dict.items():
    print(f'{k}: {v}')


# In[8]:


# Profily zemí II, procenta z lineárních hodnot

row_sums = df_heating.sum(axis=1)
df_heating_percentage = (df_heating.div(row_sums, axis=0)) * 100

print('Mediánová koncentrace látky v topné vs. netopné sezóně, procentuální hodnota')
print('=============================================================================')
print(df_heating_percentage)
print('')

print(plot_profiles(df_heating_percentage,
                    'Country profiles - percentage of total, heating and non-heating season compared. Smokers excluded',
                   'Percentage, median concentrations of imputed, CRT-starndardized values' ))


# In[ ]:




