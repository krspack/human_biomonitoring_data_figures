#!/usr/bin/env python
# coding: utf-8

# ## Heating season (CRT-standardized values)

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

parent_impcrtlog10_columns = [f'{p}_impcrtlog10' for p in parent_to_metabolite.keys()]


# In[2]:


#  parent compounds × topna sezona, vsechny staty dohromady

all_countries = sorted(list(df['country'].unique()))

fig, axes = plt.subplots(1, 5, figsize=(18, 6))
fig.suptitle('Parent compounds x heating. All countries together. No smokers, CRT-standardized, imputed values')
for idx, parent_compound_log10 in enumerate(parent_impcrtlog10_columns):

    ax = axes[idx]
    
    df_filtered = df.copy()
    df_filtered = df_filtered[[parent_compound_log10, 'samplingmonth', 'heating_season', 'country']].dropna()
    
    # n
    n_dict = df_filtered.groupby('heating_season', observed=False)[parent_compound_log10].size().to_dict()
    
    # t-test
    heating_true_values = df_filtered[df_filtered['heating_season'] == True][parent_compound_log10]
    heating_false_values = df_filtered[df_filtered['heating_season'] == False][parent_compound_log10]
    if len(heating_true_values) > 0 and len(heating_false_values) > 0:
        result = stats.ttest_ind(heating_true_values, heating_false_values)
        p_value = round(result.pvalue, 4)
    else:
        p_value = None

    # plot color
    def modify_color(input_color):  # potrebne ke grafickemu rozliseni True a False
        rgb_color = list(to_rgb(input_color))
        rgb_color[0] -= 0.002   # random small number
        return to_hex(rgb_color)

    color = parents_colors[parent_compound_log10.replace('_impcrtlog10', '')]
    modified_color = modify_color(color)
    palette = {True: modified_color, False: color}

    # boxplot
    sns.boxplot(
        x=df_filtered['heating_season'],
        y=df_filtered[parent_compound_log10],
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
    ax.set_title(parent_compound_log10)
    
    ax.set_xlabel(f'p-value: {p_value}')
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    xtick_labels = n_dict.items()
    ax.set_xticklabels(xtick_labels)

    y_ticks = ax.get_yticks()  # Get current y-tick positions (log-scale)
    y_labels = [round(np.power(10, tick), 2) for tick in y_ticks]  # Convert and round to 2 decimals
    ax.set_yticks(y_ticks)  # Keep positions
    ax.set_yticklabels(y_labels)  # Replace labels with linear equivalents
    ax.set_ylabel("Log-transformed values (log10)")
        
plt.tight_layout()
plt.show()


# In[3]:


# metabolity × topna sezona, vsechny staty dohromady

fig, axes = plt.subplots(3, 5, figsize=(20, 20), sharey = False) 
fig.suptitle('Heating season × PAH concentration for all countries together. Smokers excluded. BAP mostly imputed.', fontsize=16, y=1.02)
fig.subplots_adjust(hspace=0.8)
print('')

for idx, substance in enumerate(pah_columns):
    row = idx // 5
    col = idx % 5

    ax = axes[row, col]
    filtered_df = df[['country', 'samplingmonth', 'heating_season', substance+'_impcrtlog10']].dropna()

    # n
    n_dict = filtered_df.groupby('heating_season', observed=False)[substance+'_impcrtlog10'].size().to_dict()
    if 0 not in n_dict.values():

        # plot color
        def modify_color(input_color):  # potrebne ke grafickemu rozliseni True a False
            rgb_color = list(to_rgb(input_color))
            rgb_color[0] -= 0.02   # random small number
            return to_hex(rgb_color)
    
        color = metabolites_colors[substance]
        modified_color = modify_color(color)
        palette = {True: modified_color, False: color}
        
        # Boxplot
        sns.boxplot(
            x=filtered_df['heating_season'],
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
        heating_true_values = filtered_df[filtered_df['heating_season'] == True][f'{substance}_impcrtlog10']
        heating_false_values = filtered_df[filtered_df['heating_season'] == False][f'{substance}_impcrtlog10']
        if len(heating_true_values) > 0 and len(heating_false_values) > 0:
            result = stats.ttest_ind(heating_true_values, heating_false_values)
            p_value = round(result.pvalue, 4)
        else:
            p_value = None

        # x-axis
        xticks = [0, 1]
        ax.set_xticks(xticks)
        ax.set_xticklabels(n_dict.items())
        ax.set_xlabel(f'p-value: {p_value}')
        
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


# In[4]:


#  parent compounds × státy × topna sezona

parent_columns = parent_to_metabolite.keys()

fig, axes = plt.subplots(5, 1, figsize=(15, 20))
for idx, substance in enumerate(parent_columns):

    ax = axes[idx]
    
    df_filtered = df.copy()
    df_filtered = df_filtered[[substance+'_impcrtlog10', 'samplingmonth', 'heating_season', 'country']].dropna()
    
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
    
        color = parents_colors[substance]
        modified_color = modify_color(color)
        palette = {True: modified_color, False: color}
    
        # boxplot
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
        ax.set_title(f'{substance} x heating. (CRT-standardized, imputed values, no smokers.)')

        ax.set_xlabel(n_dict)
        xticks = list(range(len(all_countries)))  
        ax.set_xticks(xticks)
        xtick_labels = [f'{item}\n p-value = {country_pvalue.get(item, "")}' for item in all_countries]
        ax.set_xticklabels(xtick_labels, rotation = 45)

        y_ticks = ax.get_yticks()  # Get current y-tick positions (log-scale)
        y_labels = [round(np.power(10, tick), 2) for tick in y_ticks]  # Convert and round to 2 decimals
        ax.set_yticks(y_ticks)  # Keep positions
        ax.set_yticklabels(y_labels)  # Replace labels with linear equivalents
        ax.set_ylabel("Log-transformed values (log10)")
        
plt.tight_layout()
plt.show()
print('\n' * 4)


# In[5]:


# Metabolity x heating season x staty


fig, axes = plt.subplots(len(pah_columns), 1, figsize=(18, 4 * len(pah_columns)))  

for idx, substance in enumerate(pah_columns):
    ax = axes[idx] 
    
    df_filtered = df.copy()   
    df_filtered = df_filtered[[substance+'_impcrtlog10', 'samplingmonth', 'heating_season', 'country']].dropna()

    # n
    grouped = df_filtered.groupby(['country', 'heating_season'], observed=False)[substance+'_impcrtlog10'].size().reset_index(name='count')
    n_list = grouped.to_dict('split')['data']
    n_list = sorted(n_list, key=lambda x: x[0])
    n_dict = {item[0]: [] for item in n_list}
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

        # Boxplot
        df_filtered['country'] = pd.Categorical(df_filtered['country'], categories=all_countries, ordered=True)
        sns.boxplot(
            x=df_filtered['country'],
            y=df_filtered[f'{substance}_impcrtlog10'],
            hue=df_filtered['heating_season'],
            hue_order=[False, True],
            whis=(5, 95),
            ax=ax,
            dodge=True,
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
        
        # Axes, titles
        ax.set_title(f'{substance} x heating. (CRT-standardized, imputed values, no smokers.)')

        ax.set_xlabel(n_dict)
        xticks = list(range(len(all_countries)))
        ax.set_xticks(xticks)
        xtick_labels = [f'{item}\n p-value = {country_pvalue.get(item, "")}' for item in all_countries]
        ax.set_xticklabels(xtick_labels, rotation=45)

        y_ticks = ax.get_yticks()  # Get current y-tick positions (log-scale)
        y_labels = [round(np.power(10, tick), 2) for tick in y_ticks]  # Convert and round to 2 decimals
        ax.set_yticks(y_ticks)  # Keep positions
        ax.set_yticklabels(y_labels)  # Replace labels with linear equivalents
        ax.set_ylabel("log10-transformed")

plt.tight_layout()
plt.show()


# In[ ]:




