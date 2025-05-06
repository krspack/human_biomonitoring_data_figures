#!/usr/bin/env python
# coding: utf-8

# ## Profily - parent compounds (CRT-standardized values)

# In[1]:


from itertools import combinations
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import numpy as np
from scipy.stats import f
from hotelling.stats import hotelling_t2
from statannotations.Annotator import Annotator

df = pd.read_pickle('df_fillna_medians.pkl') 

with open('pah_11_columns.pkl', 'rb') as f:  
    pah_columns = pickle.load(f)

with open('parent_to_metabolite_11.pkl', 'rb') as f:  
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

parent_impcrt_columns = [f'{p}_impcrt' for p in parent_to_metabolite.keys()]


# In[2]:


# profily zemi

df_medians = df.groupby(['country'], observed=False)[parent_impcrt_columns].median()
df['country'] = df['country'].astype(str)  # aby bylo mozne ho seradit abecedne pro ucely grafu
df_medians.sort_index(inplace = True)

impcrt_palette = {key+"_impcrt": value for key, value in parents_colors.items()}

def plot_profiles(dataframe, title, ylabel):
    fig = plt.figure(figsize = [16, 6])
    ax = plt.gca()
    
    countries = sorted(dataframe.index)
    
    bottom_values = np.zeros(len(countries)) 
    for i, parent_compound in enumerate(parent_impcrt_columns):
        values = dataframe.loc[countries, parent_compound]
        ax.bar(countries, values, bottom=bottom_values, label=parent_compound.replace('_impcrt', ''), color=impcrt_palette[parent_compound])
        bottom_values += values
    
    # Set x-ticks and labels
    ax.set_xticks(countries)
    
    # Add labels and title
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Countries. Missing observations imputed with median value. Diohnap and ninehfluo missing. BAP: mostly imputed values.')
    ax.set_title(title)
    
    # Add legend
    ax.legend(title="Parent compounds:", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Show plot
    plt.tight_layout()
    plt.show()
    return
print(plot_profiles(df_medians,
                    'Country profiles of parent compounds. Smokers excluded',
                   'Median concentrations of CRT-standardized values, linear scale' ))


# In[3]:


# Profily zemí II, procenta z lineárních hodnot, chybejici hodnoty nahrazeny medianem

row_sums = df_medians.sum(axis=1)
df_medians_percentage = (df_medians.div(row_sums, axis=0)) * 100

print(plot_profiles(df_medians_percentage, 
                    'Country profiles of parent compounds, percentage. Smokers excluded',
                   'Median values, percentage of total for each country'))


# In[4]:


# Hotelliguv test

# vybrat sloupce impcrt
df_hotelling = df.copy()
df_hotelling = df_hotelling[['country', *parent_impcrt_columns]]

# prevest hodnoty na procenta
row_sums = df_hotelling[parent_impcrt_columns].sum(axis=1)
df_hotelling[parent_impcrt_columns] = df_hotelling[parent_impcrt_columns].div(row_sums, axis=0) * 100

# vydelit vsechny hodnoty hodnotou prvniho metabolitu
for column in parent_impcrt_columns:
    df_hotelling[column] = df_hotelling[column] / df_hotelling[parent_impcrt_columns[0]]
    
# zlogaritmovat vsechny hodnoty pro normalni rozdeleni
for column in parent_impcrt_columns:
    df_hotelling[column] = np.log(df_hotelling[column])

# vyhodit z dat sloupec, kterym se delilo
columns_without_first = parent_impcrt_columns[1:]

# Hotellinguv test 
countries = sorted(list(df.country.unique()))
p_values_df = pd.DataFrame(index=countries, columns=countries, dtype=float)

# test nebere staty, co maji cele shodne sloupce dat (chybejici hodnoty nahrazene medianem). Nahradit error hodnotou np.nan.
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
            data_1 = df_hotelling[df_hotelling['country'] == country_1][columns_without_first].values
            data_2 = df_hotelling[df_hotelling['country'] == country_2][columns_without_first].values
            # Perform Hotelling T² test and unpack results
            result = safe_hotelling_t2(data_1, data_2)
        else:
            result = None, None, np.nan, None

        p_value = result[2]
        all_results[(country_1, country_2)] = result
        p_values_df.loc[country_1, country_2] = p_value

print(p_values_df)


# In[6]:


# # profily zemi pro topnou a netopnou sezonu, test a graf

df['heating_season'] = pd.Categorical(df['heating_season'],categories=[False, True])

df_heating = df.groupby(['country', 'heating_season'], observed=False)[parent_impcrt_columns].median()
df_heating.index = df_heating.index.map(lambda x: x[0] + " " + str(x[1]), na_action = 'ignore')  # vstupem funkce map() je funkce. Zde lambda ze dvou sloupcu (stat a season) vyrobi jeden

print('Parent compounds v topné vs. netopné sezóně, absolutní hodnoty')
print('===============================================================')
print(df_heating)
print('')

# vybrat sloupce impcrt
df_hotelling = df.copy()
df_hotelling = df_hotelling[~df_hotelling['country'].isin(['DE', 'HR', 'PL'])]  # vyhodit se srovnani staty, ktere nemerily v obou sezonach
df_hotelling = df_hotelling[['country', 'heating_season', *parent_impcrt_columns]]

# prevest hodnoty na procenta
row_sums = df_hotelling[parent_impcrt_columns].sum(axis=1)
df_hotelling[parent_impcrt_columns] = df_hotelling[parent_impcrt_columns].div(row_sums, axis=0) * 100

# vydelit vsechny hodnoty hodnotou posledniho metabolitu
for column in parent_impcrt_columns:
    df_hotelling[column] = df_hotelling[column] / df_hotelling[parent_impcrt_columns[-1]]
    
# zlogaritmovat vsechny hodnoty pro normalni rozdeleni
for column in parent_impcrt_columns:
    df_hotelling[column] = np.log(df_hotelling[column])

# vyhodit z dat prvni sloupec, kterym se delilo
columns_without_last = parent_impcrt_columns[:-1]

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
for country in countries:     
    # Extract data for the two conditions
    data_1 = df_hotelling[
        (df_hotelling['country'] == country) & (df_hotelling['heating_season'] == True)
    ][columns_without_last].values
    data_2 = df_hotelling[
        (df_hotelling['country'] == country) & (df_hotelling['heating_season'] == False)
    ][columns_without_last].values

     # Check for empty slices or insufficient rows
    if data_1.size == 0 or data_2.size == 0:
        print(f"Skipping {country}: Empty data slice.")
        p_values_dict[country] = np.nan
        continue

    result = safe_hotelling_t2(data_1, data_2)
    p_value = result[2]
    all_results[country] = result
    p_values_dict[country] = p_value

def plot_profiles(dataframe, title, ylabel):
    fig = plt.figure(figsize = [16, 6])
    ax = plt.gca()
    
    country_heating = sorted(dataframe.index)
    bottom_values = np.zeros(len(country_heating)) 
    
    for i, parent_compound in enumerate(parent_impcrt_columns):
        values = dataframe.loc[country_heating, parent_compound]       
        ax.bar(x = country_heating, height = values, bottom=bottom_values, label=parent_compound.replace('_impcrt', ''), color=impcrt_palette[parent_compound])
        bottom_values += values
    
    # osy, popisky, ticks
    ax.set_xticks(dataframe.index)
    for pos in [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5]:
        ax.axvline(x=pos, color='gray', linestyle='--', linewidth=1)
        
    ax.set_xlabel('Countries and heating season. Missing observations imputed with median value. Diohnap and ninehfluo missing. BAP: mostly imputed values')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    ax.legend(title="Parent compounds:", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show plot
    plt.tight_layout()
    plt.show()
    return ''
print(plot_profiles(df_heating,
                    'Country profiles of parent compounds, heating and non-heating season compared. Smokers excluded',
                   'Median concentrations of CRT-standardized values, linear scale' ))
print('P-values:')
print('=========')
for k, v in p_values_dict.items():
    print(f'{k}: {v}')


# In[7]:


# Profily zemí II, procenta z lineárních hodnot

row_sums = df_heating.sum(axis=1)
df_heating_percentage = (df_heating.div(row_sums, axis=0)) * 100

print('Parent compounds v topné vs. netopné sezóně, procentualni hodnoty')
print('===============================================================')
print(df_heating_percentage)
print('')


print(plot_profiles(df_heating_percentage,
                    'Country profiles of parent compounds - percentage, heating and non-heating season compared. Smokers excluded',
                   'Percentage of total, imputed, CRT-starndardized values (country medians)' ))


# In[8]:


# profily zemi pro degurba

df['degurba'] = df['degurba'].cat.rename_categories({1.0: 'city', 2.0: 'town/suburb', 3.0: 'rural'})
df['degurba'] = df['degurba'].cat.set_categories(['city', 'town/suburb', 'rural'], ordered = True)
df_degurba = df.groupby(['country', 'degurba'], observed=False)[parent_impcrt_columns].median()
df_degurba.index = df_degurba.index.map(lambda x: f"{x[0]} {str(x[1])}")

print('Parent compounds v jednotlivých typech zástavby, absolutní hodnoty')
print('==================================================================')
print(df_degurba)
print('')

def plot_profiles(dataframe, title, ylabel):
    fig = plt.figure(figsize = [16, 6])
    ax = plt.gca()
    
    country_degurba = dataframe.index
    bottom_values = np.zeros(len(country_degurba)) 
    
    for i, parent_compound in enumerate(parent_impcrt_columns):
        values = dataframe.loc[country_degurba, parent_compound]       
        ax.bar(country_degurba, values, bottom=bottom_values, label=parent_compound.replace('_impcrt', ''), color=impcrt_palette[parent_compound])
        bottom_values += values
    
    # osy, popisky, ticks
    ax.set_xticks(dataframe.index)    
    ax.set_xticklabels(dataframe.index, rotation = 90)
    for pos in [2.5, 5.5, 8.5, 11.5, 14.5, 17.5, 20.5, 23.5, 26.5]:
        ax.axvline(x=pos, color='gray', linestyle='--', linewidth=1)
        
    ax.set_xlabel('Countries and degree of urbanization. Missing observations imputed with median value. Diohnap and ninehfluo missing. BAP: mostly imputed values')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    ax.legend(title="Parent compounds:", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Show plot
    plt.tight_layout()
    plt.show()
    return
print(plot_profiles(df_degurba,
                    'Country profiles of parent compounds, degrees of urbanization compared. Smokers excluded',
                   'Median concentrations of CRT-standardized values, linear scale' ))


# In[9]:


# Hotelliguv test pro rozliseni degurba u kazdeho statu

# vybrat sloupce impcrt
df_hotelling = df.copy()
df_hotelling = df_hotelling[~df_hotelling['country'].isin(['DE', 'PL', 'CH'])]  # vyhodit se srovnani staty, ktere nemerily ve vsech typech zastavby
df_hotelling = df_hotelling[['country', 'degurba', *parent_impcrt_columns]]

# prevest hodnoty na procenta
row_sums = df_hotelling[parent_impcrt_columns].sum(axis=1)
assert len(row_sums) == df_hotelling.shape[0]
df_hotelling[parent_impcrt_columns] = df_hotelling[parent_impcrt_columns].div(row_sums, axis=0) * 100

# vydelit vsechny hodnoty hodnotou jednoho sloupce
for column in parent_impcrt_columns:
    df_hotelling[column] = df_hotelling[column] / df_hotelling[parent_impcrt_columns[0]]
    
# zlogaritmovat vsechny hodnoty pro normalni rozdeleni
for column in parent_impcrt_columns:
    df_hotelling[column] = np.log(df_hotelling[column])

# vyhodit z dat ten sloupec, kterym se delilo
columns_without_one = parent_impcrt_columns[1:]

# Hotellinguv test
countries = sorted(list(df_hotelling.country.unique()))

# test nebere dvojice, co maji cele shodne sloupce dat (chybejici hodnoty nahrazene medianem). Nahradit error hodnotou np.nan.
def safe_hotelling_t2(group_1, group_2):
    if (len(group_1) > 0 and len(group_2) > 0):
        try:
            return hotelling_t2(group_1, group_2) 
        except np.linalg.LinAlgError:
            return None, None, np.nan  # vrati np.nan jako p-value u problematickych dvojic
    return None, None, np.nan

# prochazeni statu 
all_results = {}
for country in countries:     
    # vytvorit skupiny, co se budou mezi sebou porovnavat
    data_city = df_hotelling[
        (df_hotelling['country'] == country) & (df_hotelling['degurba'] == 'city')
    ][columns_without_one].values
    data_town = df_hotelling[
        (df_hotelling['country'] == country) & (df_hotelling['degurba'] == 'town/suburb')
    ][columns_without_one].values
    data_rural = df_hotelling[
        (df_hotelling['country'] == country) & (df_hotelling['degurba'] == 'rural')
    ][columns_without_one].values

    p_value_city_town = safe_hotelling_t2(data_city, data_town)[2]
    p_value_city_rural = safe_hotelling_t2(data_city, data_rural)[2]
    p_value_town_rural = safe_hotelling_t2(data_town, data_rural)[2]
    
    all_results[country] = [{'p_value_city_town': p_value_city_town}, {'p_value_city_rural': p_value_city_rural}, {'p_value_town_rural': p_value_town_rural}]

print('Hotelling test results / p-values:')
print('==================================')
print('Differences between degrees of urbanization per country (where data for comparison is available): ')
for k, v in all_results.items():
    print('')
    print(k)
    for _ in v:
        print(_)
        


# In[10]:


# totez v procentech
row_sums = df_degurba.sum(axis = 1)
df_degurba_percentage = (df_degurba.div(row_sums, axis=0)) * 100

print('Parent compounds v jednotlivých typech zástavby, procentualni hodnoty')
print('=====================================================================')
print(df_degurba_percentage)
print('')

print(plot_profiles(df_degurba_percentage,
                    'Country profiles of parent compounds, degrees of urbanization compared. Smokers excluded',
                   'Median concentrations of CRT-standardized values, percentage' ))


# In[ ]:




