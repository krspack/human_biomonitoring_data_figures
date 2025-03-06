#!/usr/bin/env python
# coding: utf-8

# ## Profily - parent compounds

# In[1]:


from itertools import combinations
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import numpy as np
from scipy.stats import f
from hotelling.stats import hotelling_t2
from statannotations.Annotator import Annotator

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
df = df.assign( crt_g_l = df['crt']/1000000, crt_limity = '' )
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
    for parent, metabolites in parent_11_compounds_dict.items():
        for m in metabolites:
            if compound == m:
                return parent
    return ''


# In[2]:


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


# In[3]:


# uprava tabulky pro profily zemi

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

# Dopocitat pro obhap impcrt
df_copy[f'{missing_substance}_impcrt'] = df_copy[f'{missing_substance}_imp'] / df_copy['crt_g_l']

# Nahradit chybejici hodnoty vsech latek (impcrt) medianem, nahradit nulu limitem detekce
def fillna_with_medians(dataframe: pd.DataFrame):
    for column in impcrt_11_columns:
        dataframe[column] = dataframe[column].fillna(dataframe[column].median())
        substance = column.replace('_impcrt', '')
        dataframe[column] = np.where(dataframe[column] == 0, dataframe[substance+'_lod'], dataframe[column])
    return dataframe
df_copy = fillna_with_medians(df_copy)

# Dopocitat pro obhap dalsi sloupce
df_copy[f'{missing_substance}_impcrtlog'] = np.log(df_copy[f'{missing_substance}_impcrt'])
df_copy[f'{missing_substance}_impcrtlog10'] = np.log10(df_copy[f'{missing_substance}_impcrt'])

# Spocitat soucty koncentraci pro parent compounds
parent_impcrt_columns = []
for parent, metabolites in parent_11_compounds_dict.items():
    metabolites_impcrt = [f'{metab}_impcrt' for metab in metabolites]
    for metab in metabolites_impcrt:
        df_copy[metab] = pd.to_numeric(df_copy[metab])
    df_copy[f'{parent}_impcrt'] = df_copy[metabolites_impcrt].sum(axis=1)
    parent_impcrt_columns.append(f'{parent}_impcrt')
    


# In[4]:


# kontrola: zadne chybejici hodnoty u metabolitu
for col in impcrt_11_columns:
    assert df_copy[col].isna().sum() == 0, f"Chyba: Ve sloupci '{col}' chybi hodnoty."

# kontrola: zadne chybejici hodnoty u parents
for parent in parent_11_compounds_dict.keys():
    assert df_copy[parent+'_impcrt'].isna().sum() == 0, f"Chyba: Ve sloupci '{col}' chybi hodnoty."

# kontrola: parent sloupce jsou v tabulce
assert set(parent_impcrt_columns).issubset(df_copy.columns), "Chyba: Parent impcrt columns nejsou mezi sloupci tabulky."

# kontrola: impcrt maji jen kladne hodnoty
for c in parent_impcrt_columns:
    assert (df_copy[c] > 0).all(), f'chyba: nulove nebo zaporne hodnoty ve sloupci {c}'


# In[5]:


# profily zemi

df_medians = df_copy.groupby(['country'], observed=False)[parent_impcrt_columns].median()
df_copy['country'] = df_copy['country'].astype(str)  # aby bylo mozne ho seradit abecedne pro ucely grafu
df_medians.sort_index(inplace = True)

impcrt_palette = {key+"_impcrt": value for key, value in parent_colors.items()}

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


# In[6]:


# Profily zemí II, procenta z lineárních hodnot, chybejici hodnoty nahrazeny medianem

row_sums = df_medians.sum(axis=1)
df_medians_percentage = (df_medians.div(row_sums, axis=0)) * 100

print(plot_profiles(df_medians_percentage, 
                    'Country profiles of parent compounds, percentage. Smokers excluded',
                   'Median values, percentage of total for each country'))


# In[7]:


# Hotelliguv test

# vybrat sloupce impcrt
df_hotelling = df_copy.copy()
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


# In[8]:


# # profily zemi pro topnou a netopnou sezonu, test a graf

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
df_copy['heating_season'] = pd.Categorical(df_copy['heating_season'],categories=[False, True])

df_heating = df_copy.groupby(['country', 'heating_season'], observed=False)[parent_impcrt_columns].median()
df_heating.index = df_heating.index.map(lambda x: x[0] + " " + str(x[1]), na_action = 'ignore')  # vstupem funkce map() je funkce. Zde lambda ze dvou sloupcu (stat a season) vyrobi jeden

print('Parent compounds v topné vs. netopné sezóně, absolutní hodnoty')
print('===============================================================')
print(df_heating)
print('')

# vybrat sloupce impcrt
df_hotelling = df_copy.copy()
df_hotelling = df_hotelling[~df_hotelling['country'].isin(['DE', 'HR', 'PL'])]  # vyhodit se srovnani staty, ktere nemerily v obou sezonach
df_hotelling['heating_season'] = df_hotelling.apply(
    lambda row: int(row['samplingmonth']) in heating_season[row['country']], axis=1)

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


# In[9]:


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


# In[10]:


# profily zemi pro degurba

df_copy['degurba'] = df_copy['degurba'].cat.rename_categories({1.0: 'city', 2.0: 'town/suburb', 3.0: 'rural'})
df_copy['degurba'] = df_copy['degurba'].cat.set_categories(['city', 'town/suburb', 'rural'], ordered = True)
df_degurba = df_copy.groupby(['country', 'degurba'], observed=False)[parent_impcrt_columns].median()
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


# In[11]:


# Hotelliguv test pro rozliseni degurba u kazdeho statu

# vybrat sloupce impcrt
df_hotelling = df_copy.copy()
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
        


# In[12]:


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




