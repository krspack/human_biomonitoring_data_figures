#!/usr/bin/env python
# coding: utf-8

# ## Degree of urbanization (CRT-standardized values)

# In[1]:


import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import to_rgb, to_hex
import pandas as pd
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


# In[2]:


# Degree of urbanization × koncentrace parent compounds pro vsechny staty dohromady 
color_palette = {key+"_impcrtlog10": value for key, value in parents_colors.items()}
parent_impcrtlog10_columns = [f'{parent}_impcrtlog10' for parent in parent_to_metabolite.keys()]
   
fig, axes = plt.subplots(1, 5, figsize=(18, 6), sharey = False) 
fig.suptitle('Degree of urbanization × parent compounds concentration. All countries together. Smokers excluded', fontsize=12, y=1.02)
print('')

for idx, parent_log10 in enumerate(parent_impcrtlog10_columns):
    col = idx % 5
    ax = axes[col]
    
    filtered_df = df[['degurba', parent_log10]].dropna()
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


# In[3]:


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


# In[4]:


# Degree of urbanization × koncentrace parent compounds zvlášť pro jednotlivé státy
countries_list = sorted(list(df.country.unique()))

color_palette = {key+"_impcrtlog10": value for key, value in parents_colors.items()}

for country in countries_list:
    df_one_country = df[df.country == country].copy()
       
    fig, axes = plt.subplots(1, 5, figsize=(18, 6), sharey = False) 
    fig.suptitle(f'{country}, degree of urbanization × parent compounds concentration. Smokers excluded', fontsize=12, y=1.02)
    print('')

    for idx, parent_log10 in enumerate(parent_impcrtlog10_columns):
        col = idx % 5
        ax = axes[col]
        
        filtered_df = df_one_country[['degurba', parent_log10]].dropna()
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


# In[5]:


# Degree of urbanization × koncentrace metabolitů v jednotlivých státech

for country in sorted(list(df.country.unique())):
    df_one_country = df[df.country == country].copy()
      
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




