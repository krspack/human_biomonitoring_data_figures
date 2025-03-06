#!/usr/bin/env python
# coding: utf-8

# ## Údaje o respondentech a sběru vzorků

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import numpy as np

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
df = df[df['crt_limity'] == 'ok']  # maže 6 řádek


# In[2]:


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

parent_to_metabolite = {
    'PYR': ['ohpyr'], 
    'FLU': ['twohfluo', 'threehfluo', 'ninehfluo'],
    'PHE': ['onehphe', 'twohphe', 'threehphe', 'fourhphe', 'ninehphe'],
    'NAP': ['oneohnap', 'twoohnap', 'diohnap'], 
    'BAP': ['ohbap']
}

def get_parent(compound):
    for parent, metabolites in parent_to_metabolite.items():
        for m in metabolites:
            if compound == m:
                return parent
    return ''


# In[3]:


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


# In[4]:


# velikost datasetu, chybejici hodnoty:
print("Počet respondentů: ", len(df))
print('==================')
print("Počet proměnných: ", len(df.columns))
print('==================')
print('Chybějící hodnoty:')
print('==================')
for column in df:
    print(column, df[column].isna().sum())


# In[5]:


# počet vzorků vč. LOD, LOQ, pohled po jednotlivých látkách: 
df_copy = df.copy()
limits_columns = []  # pro pouziti v dalsich grafech

fig, ax = plt.subplots(3, 5, figsize=(18, 8))
fig.suptitle('Počet a typ vzorků I: přehled látek', fontsize=16, y=1.02)
color_palette = {'-3.0': 'yellow', '-2.0': 'orange', '-1.0': 'red', 'ok': 'lightblue'}
for index, substance in enumerate(pah_columns):

    df_copy[f'{substance}_limity'] = ''
    df_copy[f'{substance}_limity'] = df_copy[f'{substance}'].dropna(axis = 0).apply(lambda x: 'ok' if x >= 0 else str(x))
    limits_columns.append(f'{substance}_limity')
    
    samples_count = df_copy.groupby(['country', f'{substance}_limity'], observed = False)['id_hbm4eu_subject'].size().reset_index(name='pocet_vzorku')
    samples_count_pivoted = samples_count.pivot(index = 'country', columns = f'{substance}_limity', values = 'pocet_vzorku')
    
    row = index // 5
    col = index % 5

    samples_count_pivoted.plot(kind='bar', stacked=True, ax=ax[row, col], legend = True if [row, col] == [0, 0] else False, color=[color_palette.get(str(col), 'grey') for col in samples_count_pivoted.columns])
    ax[row, col].set_title(get_parent(substance)+'_'+substance)
    ax[row, col].set_xticklabels(ax[row, col].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()


# In[6]:


# totéž co výše, ale pohled po jednotlivých státech: počty vzorků po jednotlivých státech, vč. LOD a LOQ
fig, ax = plt.subplots(2, 5, figsize=(18, 8)) 
fig.suptitle('Počet a typ vzorků II: přehled států', fontsize=16, y=1.02)
color_palette = {'-3.0': 'yellow', '-2.0': 'orange', '-1.0': 'red', 'ok': 'lightblue'}

for index, country in enumerate(sorted(df['country'].unique())):
    row = index // 5
    col = index % 5

    df_country = df[df['country'] == country].copy()

    # vytvorit pomocne sloupky latka_limity pro informaci, zda je koncentrace merena, nebo pod loq nebo lod
    for index, substance in enumerate(pah_columns):
        df_country[f'{substance}_limity'] = ''
        df_country[f'{substance}_limity'] = df_country[f'{substance}'].apply(lambda x: 'ok' if x >= 0 else str(x))   # vcetne prazdnych poli
    
    # preskupit tabulku, pak agregovat, pak pivotovat, protoze plot vyzaduje pivotovanou tabulku jako vstup
    melted_df = df_country.melt(value_vars=limits_columns, var_name='substance_measured', value_name='measurement_quality')
    
    samples_count = melted_df.groupby(['substance_measured', 'measurement_quality'])['measurement_quality'].count().reset_index(name='pocet_vzorku')

    samples_count_pivoted = samples_count.pivot(index='substance_measured', columns='measurement_quality', values='pocet_vzorku')
    samples_count_pivoted['nan'] = 0   # nezobrazovat Nan

    # zobrazit na ose X parent compounds
    new_index = [item.split('_')[0] for item in samples_count_pivoted.index]
    new_index = [get_parent(item)+'_'+item for item in new_index]
    samples_count_pivoted.index = new_index
    samples_count_pivoted.sort_index(inplace = True)

    # graf
    color=[color_palette.get(col, 'grey') for col in samples_count_pivoted.columns]
    samples_count_pivoted.plot(kind='bar', stacked=True, ax=ax[row, col], color = color, legend = True if [row, col] == [0, 0] else False)
    ax[row, col].set_title(country)
    ax[row, col].tick_params(axis='x', rotation=90)

    # pridat linky a barvy podle parent compound
    for pos in [0.5, 3.5, 6.5, 11.5]:
        ax[row, col].axvline(x=pos, color='gray', linestyle='--', linewidth=1)
    for label in ax[row, col].get_xticklabels():
        text = label.get_text().split('_')[0]  # Example extraction logic
        color = parent_colors.get(text, 'black')  # Default to black if no color specified
        label.set_color(color)
    
plt.tight_layout()
plt.show()


# In[7]:


# poměr počtu respondentů, kteří mají data o PAHs standradizovaná na creatinin (crt) a na hustotu moči (sg)
df['CRT_SG'] = df.apply (lambda row: 'both_crt_sg' if (pd.notna(row.crt) and pd.notna(row.sg)) else (
        'crt' if (pd.notna(row.crt) and pd.isna(row.sg)) else (
        'sg' if (pd.notna(row.sg) and pd.isna(row.crt)) else 'neither_crt_nor_sg')), axis=1)
print('Počty respondentů s daty standardizovanými na kreatinin (crt) a na hustotu moči (sg):')
print('=====================================================================================')
print(df['CRT_SG'].value_counts())
print('')

# Kontrola: sloupec sg znamena, ze respondent ma zmerenou hustotu moci. Bez teto promenne neni mozne mit 
# vyplnena data u promennych *_sg. Alogicky u crt. Kontrola, ze tyto sloupce nemaji vyplnenych vic hodnot, nez ma sloupec sg/crt:
sg_columns = [column for column in df.columns if 'sg' in column]
sg_columns_lengths = {column: df[column].count() for column in sg_columns}
for value in sg_columns_lengths.values():
    assert value <= sg_columns_lengths.get('sg'), 'sg error'

crt_columns = [column for column in df.columns if 'crt' in column]
crt_columns_lengths = {column: df[column].count() for column in crt_columns}
for value in crt_columns_lengths.values():
    assert value <= crt_columns_lengths.get('crt'), 'crt error'

# vyřezení z dataestu 5 hodnot, co nemají vyplnený CRT (ani SG)
df.dropna(subset=['crt'], axis=0, inplace=True)  


# In[8]:


# 1. pocet vzorku v jednotlivych letech

grouped = df.groupby('samplingyear').size()

fig = plt.figure(figsize=(10, 3))
ax = plt.gca()
x = df['samplingyear']
years = sorted(df['samplingyear'].unique())
# ax.hist(x, bins = years, color = 'lightblue', edgecolor='black', align='mid')
ax.hist(df['samplingyear'], bins=[y-0.5 for y in range(2014, 2023)], color='lightblue', edgecolor='black')
ax.set(title='Number of Samples per Year', xlabel='samplingyear', ylabel='Count')

# n
grouped = df.groupby('samplingyear').size()
xticks = years
ax.set_xticks(years)
ax.set_xticklabels([f"{int(year)}\n(n = {grouped.get(year)})" for year in years])

plt.show()


# In[9]:


# 2.stát × rok vzorkování
roky_staty = df.groupby(['samplingyear', 'country'],observed = False).size().reset_index(name='count')
heatmap_data = roky_staty.pivot(index='country', columns='samplingyear', values='count')
heatmap_data = heatmap_data.fillna(0)

plt.figure(figsize=(7, 5))  
sns.heatmap(heatmap_data, annot=True, fmt="g", cmap='GnBu', linewidths=0.5)
plt.title('Number of Samples per Year accross Countries')
plt.ylabel('samplingyear')
plt.xlabel('country')
plt.show()



# In[10]:


# 3. sezóna
season_count = df.groupby('samplingseason')['samplingseason'].count().reset_index(name = 'count')
season_count['legend'] = ['spring', 'summer', 'autumn', 'winter']

fig = plt.figure(figsize=(5, 2))
ax = plt.gca()
x = season_count['samplingseason']
y = season_count['count']
ax.bar(x, y, color = 'lightblue')
ax.set(title='Number of Samples per Season', ylabel='Count', xlabel = 'samplingseason', xticks = range(1, 5))
plt.show()

print(season_count)
print('')


# In[11]:


# 4. stát × měsíc
season_countries = df.groupby(['samplingmonth', 'country'], observed = False).size().reset_index(name = 'count')
season_pivoted = season_countries.pivot(columns = 'samplingmonth', index = 'country', values = 'count')
season_pivoted = season_pivoted.fillna(0)

plt.figure(figsize=(8, 4)) 
sns.heatmap(season_pivoted, annot=True, fmt="g", cmap='GnBu', linewidths=0.5)
plt.title('Number of Samples accross Countries and Months of the Year')
plt.ylabel('Country')
plt.xlabel('Month')
plt.show()


# In[12]:


# 5. matrice
matrix_count = df.groupby('matrix', observed = False)['matrix'].size().reset_index(name = 'count')
matrix_count['matrix_explained'] = ['first morning urine', 'urine-spot', 'urine-24h']

fig = plt.figure(figsize = [3, 2])
ax = plt.gca()
x = matrix_count['matrix']
y = matrix_count['count']
ax.bar(x, y, color = 'lightblue')
ax.set(title='Number of Urine Samples per Matrix', ylabel='Count')
plt.show()

print(matrix_count)
print('')


# In[13]:


# 6. stát × matrice
state_matrix = df.groupby(['matrix', 'country'], observed = False).size().reset_index(name = 'count')
state_matrix_pivoted = state_matrix.pivot(columns = 'matrix', index = 'country', values = 'count')
state_matrix_pivoted = state_matrix_pivoted.fillna(0)

plt.figure(figsize = [5, 5])
sns.heatmap(state_matrix_pivoted, annot = True, fmt = 'g', linewidths = 0.7, cmap = 'BuGn')
plt.title('Number of Samples per Matrix accross Countries')
plt.ylabel('Country')
plt.xlabel('Matrix')
plt.show()
print('UM = First morning urine, US = Urine-spot, UD = Urine-24h')


# In[14]:


# 7. pohlaví
sex_groupby = df.groupby('sex', observed = False)['sex'].count().reset_index(name = 'count')

plt.figure(figsize = [3, 2])
palette = {'F': 'lightpink', 'M': 'lightblue'}


ax = plt.gca()
x = sex_groupby['sex']
y = sex_groupby['count']
ax.bar(x, y, color = palette.values())
ax.set(title = 'Number of Samples of both Sexes', ylabel = "Count", xlabel = "Sex")

# n
ax.set_xticks(ax.get_xticks())
xt = [sex_groupby.loc[item]['sex']+'\n n = '+ str(sex_groupby.loc[item]['count']) for item in sex_groupby.index]
ax.set_xticklabels(xt)

plt.show()


# In[39]:


# pohlaví x země
country_sex_count = df.groupby(['country', 'sex'], observed = False).size().reset_index(name='count')

fig, ax = plt.subplots(figsize=[16, 3])
palette = {'F': 'lightpink', 'M': 'lightblue'}
sns.barplot(x='country', y='count', hue='sex', data=country_sex_count, ax=ax, palette = palette)
ax.set(title='Number of Samples of both Sexes accross Countries ', ylabel='Number of Samples', xlabel='Countries')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Sex')

# n
pivoted = country_sex_count.pivot(columns = 'sex', index = 'country', values = 'count')
xticks = ax.get_xticks()
ax.set_xticks(xticks)
xticklabels = [f"{country}\n(n={pivoted.loc[country, 'F']}, {pivoted.loc[country, 'M']})" for country in pivoted.index]
ax.set_xticklabels(xticklabels)

plt.show()

country_sex_pivoted = country_sex_count.pivot(index
 = 'country', columns = 'sex', values = 'count')
row_sums = country_sex_pivoted.sum(axis = 1)
country_sex_percentage = (country_sex_pivoted.div(row_sums, axis = 0))*100
country_sex_percentage['difference'] = round(country_sex_percentage['F'] - country_sex_percentage['M'], 0)
print(country_sex_percentage)
mean_difference = round(country_sex_percentage['difference'].mean(), 0)
mean_difference_abs = round(country_sex_percentage['difference'].abs().mean(), 0)


print('women outnumber men on average by: ', int(mean_difference), '%')
print('mean absolute difference: ', int(mean_difference_abs), '%')











# In[16]:


# 8. věk
plt.figure(figsize = [8, 4])
ax = plt.gca()
x = df['ageyears']
ax.hist(x, bins = np.arange(min(x), max(x)+2) - 0.5, edgecolor = 'black', color = 'lightblue')
ax.set(title = 'Počet účastníků výzkumu podle věku', ylabel = 'počet', xlabel = "ageyears")
ax.set_xticks(np.arange(min(x), max(x) + 1))
ax.grid(alpha = 0.5, axis='y', linestyle='--')
plt.show()

print(df['ageyears'].describe())
print('')


# In[17]:


# 9. věk × stát
plt.figure(figsize=[10, 5])
ax = sns.boxplot(x='country', y='ageyears', data=df, whis=(5, 95), saturation = 0.60, color = 'lightblue', flierprops={"marker": "x"})
plt.title('Rozložení věku účastníků podle země')
plt.xlabel('country')
plt.ylabel('ageyears')
plt.xticks(rotation=45)   
ax.set_yticks(np.arange(20, 41, 1))
plt.show()

age_country = df.groupby('country', observed = False)['ageyears'].describe()
print(age_country)


# In[19]:


# Jak přesné je měření v jednotlivých státech? Přehled hodnot LOD, LOQ.

limits_cols = [col for col in df.columns if ('lod' in col or 'loq' in col)]
df_limits = df[['country', *limits_cols]]
print('LOQ and LOD for all substances accross countries')
print('================================================')
print(df_limits.groupby(['country'], observed = True).mean())


# In[ ]:




