#!/usr/bin/env python
# coding: utf-8

# ## Degree of urbanization

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


# dopocitat kocentrace parent compounds

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

df_copy[f'{missing_substance}_impcrt'] = df_copy[f'{missing_substance}_imp'] / df_copy['crt_g_l']
df_copy[f'{missing_substance}_impcrtlog'] = np.log(df_copy[f'{missing_substance}_impcrt'])
df_copy[f'{missing_substance}_impcrtlog10'] = np.log10(df_copy[f'{missing_substance}_impcrt'])

# chybejici hodnoty nenahrazuji medianem, protoze srovnavam v ramci jednoho statu

# Spocitat soucty koncentraci pro parent compounds + impcrtlog10
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


# Degree of urbanization × koncentrace parent compounds pro vsechny staty dohromady 
color_palette = {key+"_impcrtlog10": value for key, value in parent_colors.items()}
   
fig, axes = plt.subplots(1, 5, figsize=(18, 6), sharey = False) 
fig.suptitle('Degree of urbanization × parent compounds concentration. All countries together. Smokers excluded', fontsize=12, y=1.02)
print('')

for idx, parent_log10 in enumerate(parent_impcrtlog10_columns):
    col = idx % 5
    ax = axes[col]
    
    filtered_df = df_copy[['degurba', parent_log10]].dropna()
    filtered_df['degurba'] = filtered_df['degurba'].cat.rename_categories({1.0: 'city', 2.0: 'town/suburb', 3.0: 'rural'})

    counts = filtered_df.groupby('degurba', observed = False).size()
    
    if not pd.isna(filtered_df[parent_log10].sum()):
        
        # Boxplot for each parent compound            
        x = filtered_df['degurba']
        y = filtered_df[parent_log10]
        sns.boxplot(x=x, y=y, ax=ax, whis=[5, 95], color = color_palette[parent_log10])
    
        # ANOVA with log-transformed values
        anova_results = stats.f_oneway(
            filtered_df[filtered_df['degurba'] == 'city'][parent_log10],
            filtered_df[filtered_df['degurba'] == 'town/suburb'][parent_log10],
            filtered_df[filtered_df['degurba'] == 'rural'][parent_log10]
        )
        pvalue_anova = anova_results.pvalue

        # Tukey's HSD post-hoc test
        try:
            tukey = pairwise_tukeyhsd(endog=filtered_df[parent_log10], groups=filtered_df['degurba'], alpha=0.05)
            tukey_summary = tukey.summary()
            tukey_headers = tukey_summary.data[0]
            tukey_data = tukey_summary.data[1:]
            tukey_summary = pd.DataFrame(tukey_data, columns=tukey_headers)
        
            pairs = [(row["group1"], row["group2"]) for _, row in tukey_summary.iterrows()]
            p_values = [row["p-adj"] for _, row in tukey_summary.iterrows()]
        except ValueError:
            pairs = np.nan
            p_values = np.nan
           
    
        # Post-hoc test annotations
        if pairs is not np.nan and p_values is not np.nan:
            annotator = Annotator(ax, pairs=pairs, data=filtered_df, x="degurba", y=parent_log10)
            annotator.configure(text_format="star", loc="inside", verbose=1)
            annotator.set_pvalues(p_values)
            annotator.annotate()
        print('')

        # x-axis
        xticks = [0, 1, 2]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{item}\n n = {counts.get(item)}' for item in counts.index])
        ax.set_xlabel(f'ANOVA p-value: {pvalue_anova:.2e}')

        # y-axis in linear units
        y_ticks = ax.get_yticks()  # Get current y-tick positions 
        y_labels = [round(np.power(10, tick), 2) for tick in y_ticks]  # Convert and round to 2 decimals
        ax.set_yticks(y_ticks)  # Keep positions
        ax.set_yticklabels(y_labels)  # Replace labels with linear equivalents
        ax.set_ylabel("CRT-standardized, imputed, log10-transfrmed values")

        # title
        ax.set_title(parent_log10.replace('_impcrtlog10', ''), pad = 10)

plt.tight_layout()
plt.show()


# In[4]:


# Degree of urbanization × koncentrace PAHů pro vsechny staty dohromady
fig, axes = plt.subplots(3, 5, figsize=(16, 12), sharey = False) 
fig.suptitle('Degree of urbanization × PAH concentration for all countries together. Smokers excluded', fontsize=16, y=1.02)
fig.subplots_adjust(hspace=0.6)
print('')

for idx, substance in enumerate(pah_columns):
    row = idx // 5
    col = idx % 5
    ax = axes[row, col]

    filtered_df = df[['country', 'degurba', substance+'_impcrtlog10']].dropna()
    filtered_df['degurba'] = filtered_df['degurba'].cat.rename_categories({1.0: 'city', 2.0: 'town/suburb', 3.0: 'rural'})
    filtered_df = filtered_df[~filtered_df['country'].isin(['CH', 'DK', 'DE', 'PL'])]   # staty, ktere nemerily vsechny 3 degurba

    counts = filtered_df.groupby('degurba', observed = False).size()

    if counts.values.sum() > 0:
        print(f'{substance}: post-hoc test results')
        print('====================================')
        
        # Boxplot
        x = filtered_df['degurba']
        y = filtered_df[substance+'_impcrtlog10']
        sns.boxplot(x=x, y=y, color=metabolites_colors[substance], ax=ax, whis=[5, 95])
    
        # ANOVA with log-transformed values
        anova_results = stats.f_oneway(
            filtered_df[filtered_df['degurba'] == 'city'][substance+'_impcrtlog10'],
            filtered_df[filtered_df['degurba'] == 'town/suburb'][substance+'_impcrtlog10'],
            filtered_df[filtered_df['degurba'] == 'rural'][substance+'_impcrtlog10']
        )
        pvalue_anova = anova_results.pvalue
    
        # Tukey's HSD post-hoc test
        tukey = pairwise_tukeyhsd(endog=filtered_df[substance+'_impcrtlog10'], groups=filtered_df['degurba'], alpha=0.05)
        tukey_summary = tukey.summary()
    
        tukey_headers = tukey_summary.data[0]
        tukey_data = tukey_summary.data[1:]
        tukey_summary = pd.DataFrame(tukey_data, columns=tukey_headers)
    
        pairs = [(row["group1"], row["group2"]) for _, row in tukey_summary.iterrows()]
        p_values = [row["p-adj"] for _, row in tukey_summary.iterrows()]
    
        # Post-hoc test annotations
        annotator = Annotator(ax, pairs=pairs, data=filtered_df, x="degurba", y=substance+'_impcrtlog10')
        annotator.configure(text_format="star", loc="inside", verbose=2)
        annotator.set_pvalues(p_values)
        annotator.annotate()
        print('')

        # x-axis
        xticks = [0, 1, 2]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{item}\n n = {counts.get(item)}' for item in counts.index])
        ax.set_xlabel(f'ANOVA p-value: {pvalue_anova:.2e}')

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


# Degree of urbanization × koncentrace parent compounds zvlášť pro jednotlivé státy
countries_list = sorted(list(df.country.unique()))

color_palette = {key+"_impcrtlog10": value for key, value in parent_colors.items()}

for country in countries_list:
    df_one_country = df_copy[df.country == country].copy()
       
    fig, axes = plt.subplots(1, 5, figsize=(18, 6), sharey = False) 
    fig.suptitle(f'{country}, degree of urbanization × parent compounds concentration. Smokers excluded', fontsize=12, y=1.02)
    print('')

    for idx, parent_log10 in enumerate(parent_impcrtlog10_columns):
        col = idx % 5
        ax = axes[col]
        
        filtered_df = df_one_country[['degurba', parent_log10]].dropna()
        filtered_df['degurba'] = filtered_df['degurba'].cat.rename_categories({'1': 'city', '2': 'town/suburb', '3': 'rural'})

        counts = filtered_df.groupby('degurba', observed = False).size()
        
        if not pd.isna(filtered_df[parent_log10].sum()):
            
            # Boxplot for each parent compound            
            x = filtered_df['degurba']
            y = filtered_df[parent_log10]
            sns.boxplot(x=x, y=y, ax=ax, whis=[5, 95], color = color_palette[parent_log10])
        
            # ANOVA with log-transformed values
            anova_results = stats.f_oneway(
                filtered_df[filtered_df['degurba'] == 'city'][parent_log10],
                filtered_df[filtered_df['degurba'] == 'town/suburb'][parent_log10],
                filtered_df[filtered_df['degurba'] == 'rural'][parent_log10]
            )
            pvalue_anova = anova_results.pvalue
    
            # Tukey's HSD post-hoc test
            try:
                tukey = pairwise_tukeyhsd(endog=filtered_df[parent_log10], groups=filtered_df['degurba'], alpha=0.05)
                tukey_summary = tukey.summary()
                tukey_headers = tukey_summary.data[0]
                tukey_data = tukey_summary.data[1:]
                tukey_summary = pd.DataFrame(tukey_data, columns=tukey_headers)
            
                pairs = [(row["group1"], row["group2"]) for _, row in tukey_summary.iterrows()]
                p_values = [row["p-adj"] for _, row in tukey_summary.iterrows()]
            except ValueError:
                pairs = np.nan
                p_values = np.nan
                
        
            # Post-hoc test annotations
            if pairs is not np.nan and p_values is not np.nan:
                annotator = Annotator(ax, pairs=pairs, data=filtered_df, x="degurba", y=parent_log10)
                annotator.configure(text_format="star", loc="inside", verbose=1)
                annotator.set_pvalues(p_values)
                annotator.annotate()
            print('')
    
            # x-axis
            xticks = [0, 1, 2]
            ax.set_xticks(xticks)
            ax.set_xticklabels([f'{item}\n n = {counts.get(item)}' for item in counts.index])
            ax.set_xlabel(f'p-value: {pvalue_anova:.2e}')
    
            # y-axis in linear units
            y_ticks = ax.get_yticks()  # Get current y-tick positions 
            y_labels = [round(np.power(10, tick), 2) for tick in y_ticks]  # Convert and round to 2 decimals
            ax.set_yticks(y_ticks)  # Keep positions
            ax.set_yticklabels(y_labels)  # Replace labels with linear equivalents
            ax.set_ylabel("CRT-standardized, imputed, log10-transfrmed values")
    
            # title
            ax.set_title(parent_log10.replace('_impcrtlog10', ''), pad = 10)

    plt.tight_layout()
    plt.show()


# In[6]:


# Degree of urbanization × koncentrace metabolitů v jednotlivých státech

for country in sorted(list(df.country.unique())):
    df_one_country = df_copy[df.country == country].copy()
      
    fig, axes = plt.subplots(3, 5, figsize=(16, 12), sharey = False) 
    fig.suptitle(f'{country}, degree of urbanization × PAH concentration. Smokers excluded', fontsize=12, y=1.02)
    fig.subplots_adjust(hspace=0.6)
    print('')

    for idx, substance in enumerate(pah_columns):
 
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]
        
        filtered_df = df_one_country[['degurba', substance+'_impcrtlog10']].dropna()
        filtered_df['degurba'] = filtered_df['degurba'].cat.rename_categories({1.0: 'city', 2.0: 'town/suburb', 3.0: 'rural'})
        counts = filtered_df.groupby('degurba', observed = False).size()
    
        if counts.values.sum() > 0:
            
            # Boxplot for each substance            
            x = filtered_df['degurba']
            y = filtered_df[substance+'_impcrtlog10']
            sns.boxplot(x=x, y=y, color=metabolites_colors[substance], ax=ax, whis=[5, 95])
        
            # ANOVA with log-transformed values
            anova_results = stats.f_oneway(
                filtered_df[filtered_df['degurba'] == 'city'][substance+'_impcrtlog10'],
                filtered_df[filtered_df['degurba'] == 'town/suburb'][substance+'_impcrtlog10'],
                filtered_df[filtered_df['degurba'] == 'rural'][substance+'_impcrtlog10']
            )
            pvalue_anova = anova_results.pvalue
    
            # Tukey's HSD post-hoc test
            try:
                tukey = pairwise_tukeyhsd(endog=filtered_df[substance+'_impcrtlog10'], groups=filtered_df['degurba'], alpha=0.05)
                tukey_summary = tukey.summary()
                tukey_headers = tukey_summary.data[0]
                tukey_data = tukey_summary.data[1:]
                tukey_summary = pd.DataFrame(tukey_data, columns=tukey_headers)
            
                pairs = [(row["group1"], row["group2"]) for _, row in tukey_summary.iterrows()]
                p_values = [row["p-adj"] for _, row in tukey_summary.iterrows()]
            except ValueError:
                pairs = np.nan
                p_values = np.nan
                
        
            # Post-hoc test annotations
            if pairs is not np.nan and p_values is not np.nan:
                annotator = Annotator(ax, pairs=pairs, data=filtered_df, x="degurba", y=substance+'_impcrtlog10')
                annotator.configure(text_format="star", loc="inside", verbose=1)
                annotator.set_pvalues(p_values)
                annotator.annotate()
            print('')
    
            # x-axis
            xticks = [0, 1, 2]
            ax.set_xticks(xticks)
            ax.set_xticklabels([f'{item}\n n = {counts.get(item)}' for item in counts.index])
            ax.set_xlabel(f'ANOVA p-value: {pvalue_anova:.2e}')
    
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



# In[ ]:




