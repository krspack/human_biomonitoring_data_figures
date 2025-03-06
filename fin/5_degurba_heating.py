#!/usr/bin/env python
# coding: utf-8

# ## Degree of urbanization and heating season 

# In[1]:


import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import numpy as np
import scipy.stats as stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.contingency_tables import Table
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

# kuřáci pryč
df = df[df['smoking'] != True] # maže 443 řádek


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

parent_impcrt_columns = []
parent_impcrtlog10_columns = []
for parent, metabolites in seznam_latek_podle_rodice.items():
    metabolites_impcrt = [f'{metab}_impcrt' for metab in metabolites]
    for metab in metabolites_impcrt:
        df_copy[metab] = pd.to_numeric(df_copy[metab])
    df_copy[f'{parent}_impcrt'] = df_copy[metabolites_impcrt].sum(axis=1)
    df_copy[f'{parent}_impcrt'] = np.where(df_copy[f'{parent}_impcrt'] == 0.0, np.nan, df_copy[f'{parent}_impcrt'])
    df_copy[f'{parent}_impcrtlog10'] = np.log10(df_copy[f'{parent}_impcrt'])
    
    parent_impcrt_columns.append(f'{parent}_impcrt')
    parent_impcrtlog10_columns.append(f'{parent}_impcrtlog10')
    


# In[3]:


# Degree of urbanization & heating season × koncentrace parent compounds pro vsechny staty dohromady

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
all_countries = sorted(list(heating_season.keys()))

fig, axes = plt.subplots(1, 5, figsize=(20, 7), sharey = False) 
fig.suptitle('Degree of urbanization & heating season × parent compounds, all countries together. Smokers excluded, BAP mostly imputed', fontsize=16, y=1.02)
print('')

parent_columns = seznam_latek_podle_rodice.keys()
for idx, parent_compound in enumerate(parent_columns):
    ax = axes[idx]

    filtered_df_parents = df_copy[['country', 'degurba', 'samplingmonth', parent_compound+'_impcrtlog10']].dropna()
    filtered_df_parents['degurba'] = filtered_df_parents['degurba'].cat.rename_categories({1.0: 'city', 2.0: 'town/suburb', 3.0: 'rural'})
    filtered_df_parents = filtered_df_parents[~filtered_df_parents['country'].isin(['CH', 'HR', 'DK', 'DE', 'PL'])]   # staty, ktere nemerily vsechny 3 degurba a obe heating seasons -------------------- iceland osetrit
    filtered_df_parents['heating_season'] = filtered_df_parents.apply(lambda row: int(row['samplingmonth']) in heating_season.get(row['country']), axis = 1)  # boolean expression

    # n
    counts = filtered_df_parents.groupby(['degurba', 'heating_season'], observed = False).size()
    if counts.values.sum() > 0:

        # plot color
        def modify_color(input_color):  # potrebne ke grafickemu rozliseni True a False
            rgb_color = list(to_rgb(input_color))
            rgb_color[0] -= 0.002   # random small number
            return to_hex(rgb_color)
    
        color = parent_colors[parent_compound]
        modified_color = modify_color(color)
        palette = {True: modified_color, False: color}
        
        # Boxplot
        sns.boxplot(
            x=filtered_df_parents['degurba'],
            y=filtered_df_parents[f'{parent_compound}_impcrtlog10'],
            hue=filtered_df_parents['heating_season'],
            hue_order=[False, True],
            whis = (5, 95),
            ax=ax,
            dodge = True,
            palette=palette
            )

        # graficke rozliseni True a False, legenda
        hatch_map = {False: '', True: '*'}  
        colors = set()
        for patch in ax.patches:
            colors.add(round(patch.get_facecolor()[0]*1000))
        for patch in ax.patches:
            patch.set_hatch(hatch_map[round(patch.get_facecolor()[0]*1000) == min(colors)])  # vzor pro True je odvozen od modifikovane barvy, ne od hodnoty 
    
        legend = ax.legend()
        for patch, label in zip(legend.get_patches(), [False, True]):     # prochazeni "obdelnicku" legendy
            patch.set_hatch(hatch_map[label])  # priradit vzor
        
        # t-test
        degurba_pvalue = {}
        for degree in list(filtered_df_parents['degurba'].unique()):
            heating_true_values = filtered_df_parents[
                    (filtered_df_parents['degurba'] == degree) & 
                    (filtered_df_parents['heating_season'] == True)
                ][parent_compound+'_impcrtlog10']
            heating_false_values = filtered_df_parents[
                    (filtered_df_parents['degurba'] == degree) & 
                    (filtered_df_parents['heating_season'] == False)
                ][parent_compound+'_impcrtlog10']
            if len(heating_true_values) > 0 and len(heating_false_values) > 0:
                result = stats.ttest_ind(heating_true_values, heating_false_values)
                p_value = round(result.pvalue, 4)
            else:
                p_value = None

            degurba_pvalue[degree] = p_value

        # x-axis
        xticks = [0, 1, 2]
        ax.set_xticks(xticks)
        ax.set_xticklabels([
                f"{item}\n n(F) = {counts.loc[(item, False)]}\n n(T) = {counts.loc[(item, True)]}\n p-val: {degurba_pvalue[item]}"
                for item in ['city', 'town/suburb', 'rural']
            ])
        ax.set_xlabel('')
        
        # y-axis in linear units
        y_ticks = ax.get_yticks()  # Get current y-tick positions 
        y_labels = [round(np.power(10, tick), 2) for tick in y_ticks]  # Convert and round to 2 decimals
        ax.set_yticks(y_ticks)  # Keep positions
        ax.set_yticklabels(y_labels)  # Replace labels with linear equivalents
        ax.set_ylabel("CRT-standardized, imputed, log10-transfrmed values")

        # title
        ax.set_title(f'{parent_compound}', pad = 10)


plt.tight_layout()
plt.show()


# In[4]:


# Degree of urbanization & heating season × koncentrace PAHů pro vsechny staty dohromady

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
all_countries = sorted(list(heating_season.keys()))

fig, axes = plt.subplots(3, 5, figsize=(20, 27), sharey = False) 
fig.suptitle('Degree of urbanization & heating season × PAH concentration for all countries together. Smokers excluded', fontsize=16, y=1.02)
fig.subplots_adjust(hspace=0.8)
print('')

for idx, substance in enumerate(pah_columns):
    row = idx // 5
    col = idx % 5

    ax = axes[row, col]

    filtered_df = df_copy[['country', 'degurba', 'samplingmonth', substance+'_impcrtlog10']].dropna()
    filtered_df['degurba'] = filtered_df['degurba'].cat.rename_categories({1.0: 'city', 2.0: 'town/suburb', 3.0: 'rural'})
    filtered_df = filtered_df[~filtered_df['country'].isin(['CH', 'HR', 'DK', 'DE', 'PL', 'IS'])]   # staty, ktere nemerily vsechny 3 degurba a obe heating seasons
    filtered_df['heating_season'] = filtered_df.apply(lambda row: int(row['samplingmonth']) in heating_season.get(row['country']), axis = 1)

    """
    # Chi Square Test, abychom vyloucili souvislost mezi degurba a heating season, napriklad ze by vetsina zimnich vzorku byla z mest
    kontingencni_tabulka = pd.crosstab(filtered_df['degurba'], filtered_df['heating_season'])
    chi2, p, dof, expected = stats.chi2_contingency(kontingencni_tabulka)
    print(f"Chi-Square Statistic: {chi2}, p-value: {p}")
    """

    # n
    counts = filtered_df.groupby(['degurba', 'heating_season'], observed = False).size()
    if counts.values.sum() > 0:

        # plot color
        def modify_color(input_color):  # potrebne ke grafickemu rozliseni True a False
            rgb_color = list(to_rgb(input_color))
            rgb_color[0] -= 0.002   # random small number
            return to_hex(rgb_color)
    
        color = metabolites_colors[substance]
        modified_color = modify_color(color)
        palette = {True: modified_color, False: color}
        
        # Boxplot
        sns.boxplot(
            x=filtered_df['degurba'],
            y=filtered_df[f'{substance}_impcrtlog10'],
            hue=filtered_df['heating_season'],
            hue_order=[False, True],
            whis = (5, 95),
            ax=ax,
            dodge = True,
            palette=palette
            )
        
        # graficke rozliseni True a False, legenda
        hatch_map = {False: '', True: '*'}  
        colors = set()
        for patch in ax.patches:
            colors.add(round(patch.get_facecolor()[0]*1000))
        for patch in ax.patches:
            patch.set_hatch(hatch_map[round(patch.get_facecolor()[0]*1000) == min(colors)])  # vzor pro True je odvozen od modifikovane barvy, ne od hodnoty 
    
        legend = ax.legend()
        for patch, label in zip(legend.get_patches(), [False, True]):     # prochazeni "obdelnicku" legendy
            patch.set_hatch(hatch_map[label])  # priradit vzor
        
        # t-test
        degurba_pvalue = {}
        for degree in list(filtered_df['degurba'].unique()):
            heating_true_values = filtered_df[
                    (filtered_df['degurba'] == degree) & 
                    (filtered_df['heating_season'] == True)
                ][substance+'_impcrtlog10']
            heating_false_values = filtered_df[
                    (filtered_df['degurba'] == degree) & 
                    (filtered_df['heating_season'] == False)
                ][substance+'_impcrtlog10']
            if len(heating_true_values) > 0 and len(heating_false_values) > 0:
                result = stats.ttest_ind(heating_true_values, heating_false_values)
                p_value = round(result.pvalue, 4)
            else:
                p_value = None

            degurba_pvalue[degree] = p_value

        # x-axis
        xticks = [0, 1, 2]
        ax.set_xticks(xticks)
        ax.set_xticklabels([
                f"{item}\n n(F) = {counts.loc[(item, False)]}\n n(T) = {counts.loc[(item, True)]}\n p-val: {degurba_pvalue[item]}"
                for item in ['city', 'town/suburb', 'rural']
            ])
        
        # y-axis in linear units
        y_ticks = ax.get_yticks()  # Get current y-tick positions 
        y_labels = [round(np.power(10, tick), 2) for tick in y_ticks]  # Convert and round to 2 decimals
        ax.set_yticks(y_ticks)  # Keep positions
        ax.set_yticklabels(y_labels)  # Replace labels with linear equivalents
        ax.set_ylabel("CRT-standardized, imputed, log10-transfrmed values")

        # title
        ax.set_title(f'{substance}', pad = 10)


plt.tight_layout()
plt.show()


# In[5]:


# degurba (degree of urbanization) × státy × topna sezona

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
all_countries = sorted(list(heating_season.keys()))
degurba_legend = {1.0: 'city', 2.0: 'town/suburb', 3.0: 'rural'}

for substance in pah_columns:
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # 3 rady podle degruba
    for order, degree in enumerate(df['degurba'].unique()):
        row = order - 1
        df_filtered = df[df['degurba'] == degree].copy()
        if df_filtered.shape[0] > 0:

            df_filtered['heating_season'] = df_filtered.apply(
                lambda row: row['samplingmonth'] in set(heating_season.get(row['country'], [])), axis=1
            )
    
            
            
            df_filtered = df_filtered[[substance+'_impcrtlog10', 'samplingmonth', 'heating_season', 'country', 'degurba']].dropna()
            df_filtered['degurba'] = df_filtered['degurba'].cat.rename_categories({1.0: 'city', 2.0: 'town/suburb', 3.0: 'rural'})
    
            # n
            grouped = df_filtered.groupby(['country', 'heating_season'], observed = False)[substance+'_impcrtlog10'].size().reset_index(name = 'count')
            n_list = grouped.to_dict('split')['data']
            n_list = sorted(n_list, key=lambda x: x[0])
            n_dict = {item[0]:[] for item in n_list}
            for li in n_list:
                n_dict[li[0]].append(li[2])
            n_dict = {'n': n_dict}
    
            if grouped['count'].sum() > 0:

                # t-test
                country_pvalue = {}
                for country in all_countries:
                    if country in list(df_filtered['country'].unique()):
                        heating_true_values = df_filtered[
                                (df_filtered['country'] == country) & 
                                (df_filtered['heating_season'] == True)
                            ][substance+'_impcrtlog10']
                        heating_false_values = df_filtered[
                                (df_filtered['country'] == country) & 
                                (df_filtered['heating_season'] == False)
                            ][substance+'_impcrtlog10']
                        if len(heating_true_values) > 0 and len(heating_false_values) > 0:
                            result = stats.ttest_ind(heating_true_values, heating_false_values)
                            p_value = round(result.pvalue, 4)
                        else:
                            p_value = None
                               
                    else:
                        p_value = None
                    country_pvalue[country] = p_value
    
                # plot color
                def modify_color(input_color):  # potrebne ke grafickemu rozliseni True a False
                    rgb_color = list(to_rgb(input_color))
                    rgb_color[0] -= 0.002   # random small number
                    return to_hex(rgb_color)
            
                color = metabolites_colors[substance]
                modified_color = modify_color(color)
                palette = {True: modified_color, False: color}
        
                # boxplot
                ax = axes[row]
                df_filtered['country'] = pd.Categorical(df_filtered['country'], categories=all_countries, ordered=True)
                sns.boxplot(
                    x=df_filtered['country'],
                    y=df_filtered[f'{substance}_impcrtlog10'],
                    hue=df_filtered['heating_season'],
                    hue_order=[False, True],
                    whis = (5, 95),
                    ax=ax,
                    dodge = True,
                    palette=palette
                )

                # graficke rozliseni True a False, legenda
                hatch_map = {False: '', True: '*'}  
                colors = set()
                for patch in ax.patches:
                    colors.add(round(patch.get_facecolor()[0]*1000))
                for patch in ax.patches:
                    patch.set_hatch(hatch_map[round(patch.get_facecolor()[0]*1000) == min(colors)])  # vzor pro True je odvozen od modifikovane barvy, ne od hodnoty 
            
                legend = ax.legend()
                for patch, label in zip(legend.get_patches(), [False, True]):     # prochazeni "obdelnicku" legendy
                    patch.set_hatch(hatch_map[label])  # priradit vzor
            
                # axes, titles
                ax.set_title(f'{substance} x {degurba_legend.get(degree, 'xxx')} x heating. Smokers excluded')
        
                ax.set_xlabel(n_dict)
                xticks = list(range(len(all_countries)))  
                ax.set_xticks(xticks)
                xtick_labels = [f'{item}\n p-value = {country_pvalue.get(item, "")}' for item in all_countries]
                ax.set_xticklabels(xtick_labels, rotation = 45)
        
                y_ticks = ax.get_yticks()  # Get current y-tick positions (log-scale)
                y_labels = [round(np.power(10, tick), 2) for tick in y_ticks]  # Convert and round to 2 decimals
                ax.set_yticks(y_ticks)  # Keep positions
                ax.set_yticklabels(y_labels)  # Replace labels with linear equivalents
                ax.set_ylabel("CRT-standardized, imputed, log10-transfrmed values")
        
    plt.tight_layout()
    plt.show()
    print('\n' * 4)

    


# In[ ]:




