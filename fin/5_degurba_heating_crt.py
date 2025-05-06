#!/usr/bin/env python
# coding: utf-8

# ## Degree of urbanization and heating season (CRT-standardized values)

# In[1]:


import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
import pickle
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import numpy as np
import scipy.stats as stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.contingency_tables import Table
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


# In[2]:


# Degree of urbanization & heating season × koncentrace parent compounds pro vsechny staty dohromady


fig, axes = plt.subplots(1, 5, figsize=(20, 7), sharey = False) 
fig.suptitle('Degree of urbanization & heating season × parent compounds, all countries together. Smokers excluded, BAP mostly imputed', fontsize=16, y=1.02)
print('')

parent_columns = parent_to_metabolite.keys()
for idx, parent_compound in enumerate(parent_columns):
    ax = axes[idx]

    filtered_df_parents = df[['country', 'degurba', 'samplingmonth', 'heating_season', parent_compound+'_impcrtlog10']].dropna()
    filtered_df_parents['degurba'] = filtered_df_parents['degurba'].cat.rename_categories({1.0: 'city', 2.0: 'town/suburb', 3.0: 'rural'})
    filtered_df_parents = filtered_df_parents[~filtered_df_parents['country'].isin(['CH', 'HR', 'DK', 'DE', 'PL'])]   # staty, ktere nemerily vsechny 3 degurba a obe heating seasons -------------------- iceland osetrit

    # n
    counts = filtered_df_parents.groupby(['degurba', 'heating_season'], observed = False).size()
    if counts.values.sum() > 0:

        # plot color
        def modify_color(input_color):  # potrebne ke grafickemu rozliseni True a False
            rgb_color = list(to_rgb(input_color))
            rgb_color[0] -= 0.002   # random small number
            return to_hex(rgb_color)
    
        color = parents_colors[parent_compound]
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


# In[5]:


# Degree of urbanization & heating season × koncentrace PAHů pro vsechny staty dohromady

all_countries = sorted(list(df['country'].unique()))

fig, axes = plt.subplots(3, 5, figsize=(20, 27), sharey = False) 
fig.suptitle('Degree of urbanization & heating season × PAH concentration for all countries together. Smokers excluded', fontsize=16, y=1.02)
fig.subplots_adjust(hspace=0.8)
print('')

for idx, substance in enumerate(pah_columns):
    row = idx // 5
    col = idx % 5

    ax = axes[row, col]

    filtered_df = df[['country', 'degurba', 'samplingmonth', 'heating_season', substance+'_impcrtlog10']].dropna()
    filtered_df['degurba'] = filtered_df['degurba'].cat.rename_categories({1.0: 'city', 2.0: 'town/suburb', 3.0: 'rural'})
    filtered_df = filtered_df[~filtered_df['country'].isin(['CH', 'HR', 'DK', 'DE', 'PL', 'IS'])]   # staty, ktere nemerily vsechny 3 degurba a obe heating seasons

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


# In[6]:


# degurba (degree of urbanization) × státy × topna sezona

all_countries = sorted(list(df['country'].unique()))
degurba_legend = {1.0: 'city', 2.0: 'town/suburb', 3.0: 'rural'}

for substance in pah_columns:
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # 3 rady podle degruba
    for order, degree in enumerate(df['degurba'].unique()):
        row = order - 1
        df_filtered = df[df['degurba'] == degree].copy()
        if df_filtered.shape[0] > 0:

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




