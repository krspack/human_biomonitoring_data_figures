#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import to_rgb, to_hex
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
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
log_columns = [x+'_log' for x in pah_columns]

imp_columns = [x+'_imp' for x in pah_columns]
implog_columns = [x+'_implog' for x in pah_columns]

impcrt_columns = [x+'_impcrt' for x in pah_columns]
impcrtlog_columns = [x for x in df.columns if 'impcrtlog' in x]

# CRT: vyhodit chybejici data a vzorky nad a pod limitem (mene nez 10 radek)
df.dropna(subset=['crt'], axis=0, inplace=True)   # maže 5 řádek
df['crt_g_l'] = df['crt']/1000000
df['crt_limity'] = ''
df['crt_limity'] = df['crt_g_l'].apply(lambda x: 'pod limitem' if x < 0.05 else ('ok' if 0.05 <= x <= 5 else 'nad limitem'))
df = df[df['crt_limity'] == 'ok']  # maže 6 řádek

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
    "oneohnap": "#1f7cb4",      # 1-hydroxynaphthalene
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
    'PYR': 'e9843f', 
    'FLU': 'f9c013',
    'PHE': 'abcf93', 
    'NAP': '659fd3',
    'BAP': '#E21A3F'   # opraven kod
}

def get_parent(compound):
    for parent, metabolites in parent_to_metabolite.items():
        for m in metabolites:
            if compound == m:
                return parent
    return ''


# In[2]:


# Počet kuřáků v jednotlivých zemích
smoking_bool_cols = ['smoking', 'smoking_passive', 'smoking_2h', 'smoking_24h', 'smoking_passive24h']
smoking_other_cols = ['smoking_cigday', 'smoking_passive_freq']
df_copy = df.copy()

fig = plt.figure(figsize=[12, 3])
ax = plt.gca()
df_copy['smoking'] = df_copy['smoking'].astype(str)
color_palette = {'True': 'lightgrey', 'False': 'lightblue', '<NA>': 'black'}
smoking_counts = df_copy.groupby(['country', 'smoking'], observed = False).size().reset_index(name = 'count')
sns.barplot(x = 'country', y = 'count', hue = 'smoking', data = smoking_counts, palette = color_palette, dodge = True)
ax.set(title = 'Number of smokers in the sample accross countries')
ax.legend(loc="best")
plt.show()


# In[3]:


# počet cigaret za den u kuřáků
df_copy = df.copy()
df_copy['smoking_cigday'] = df_copy['smoking_cigday'].fillna(-1)  
smoking_counts = df_copy[df_copy.smoking == True].groupby(['smoking_cigday']).size().reset_index(name = 'count')

fig = plt.figure(figsize=[12, 3])
ax = plt.gca()

sns.barplot(x='smoking_cigday', y='count', data=smoking_counts, color = 'xkcd:greyish blue', errorbar=('ci', False))

xticks= ax.get_xticks()
ax.set_xticks(xticks)
x_labels = smoking_counts['smoking_cigday'].unique()
x_labels = ['NA' if label == -1 else int(label) for label in x_labels]
ax.set(title = 'Number of cigarettes smoked per day as stated by the smokers')
ax.set_xticklabels(x_labels)

plt.show()


# In[4]:


# Kolik z kuřáků mělo cigaretu do 24 hodin před odběrem vzorku?

fig = plt.figure(figsize=[12, 3])
ax = plt.gca()
df_copy['smoking_24h'] = df_copy['smoking_24h'].astype(str)
color_palette = {'True': 'lightgrey', 'False': 'lightblue', '<NA>': 'black'}
smoking_counts = df_copy[df_copy['smoking'] == True].groupby(['country', 'smoking_24h'], observed = False).size().reset_index(name = 'count')
sns.barplot(x = 'country', y = 'count', hue = 'smoking_24h', data = smoking_counts, palette = color_palette, dodge = True)
ax.set(title = 'Number of smokers who had a cigarette up to 24 hours before sample was taken')
plt.show()



# In[5]:


# Kolik z pasivnich kuraku pasivne kourilo 24 hodin pred odebranim vzorku?

fig = plt.figure(figsize=[12, 3])
ax = plt.gca()

df_copy_passive_smoking = df.copy()
df_copy_passive_smoking['smoking_passive24h'] = df_copy_passive_smoking['smoking_passive24h'].astype(str)
color_palette = {'True': 'lightgrey', 'False': 'lightblue', '<NA>': 'black'}
smoking_counts = df_copy_passive_smoking[(df_copy_passive_smoking['smoking'] != True) & (df_copy_passive_smoking['smoking_passive'] == True)].groupby(['country', 'smoking_passive24h'], observed = False).size().reset_index(name = 'count')
sns.barplot(x = 'country', y = 'count', hue = 'smoking_passive24h', data = smoking_counts, palette = color_palette, dodge = True)
ax.set(title = 'Number of passive smokers exposed to smoking up to 24 hours before sample was taken')
plt.show()



# In[6]:


# Kolik z kuřáků mělo cigaretu do 2 hodin před odběrem vzorku?

fig = plt.figure(figsize=[12, 3])
ax = plt.gca()
df_copy['smoking_2h'] = df_copy['smoking_2h'].astype(str)
color_palette = {'True': 'lightgrey', 'False': 'lightblue', '<NA>': 'black'}
smoking_counts = df_copy[df_copy['smoking'] == True].groupby(['country', 'smoking_2h'], observed = False).size().reset_index(name = 'count')
sns.barplot(x = 'country', y = 'count', hue = 'smoking_2h', data = smoking_counts, palette = color_palette, dodge = True)
ax.set(title = 'Number of smokers who had a cigarette up to 2 hours before sample was taken')
plt.show()

print('Většina dat chybí >>> proměnnou nepoužívat.')




# In[7]:


# Kolik nekuřáků je vystaveno pasivnímu kouření?
fig = plt.figure(figsize=[12, 3])
ax = plt.gca()
df_copy['smoking_passive'] = df_copy['smoking_passive'].astype(str)
color_palette = {'True': 'lightgrey', 'False': 'lightblue', '<NA>': 'black'}
smoking_counts = df_copy[df_copy['smoking'] != True].groupby(['country', 'smoking_passive'], observed = False).size().reset_index(name = 'count')
sns.barplot(x = 'country', y = 'count', hue = 'smoking_passive', data = smoking_counts, palette = color_palette, dodge = True)
ax.set(title = 'Number of non-smokes who smoke passively')
plt.show()




# In[8]:


# intenzita pasivního kouření
fig = plt.figure(figsize = [12, 3])
ax = plt.gca()
smoking_counts = df_copy[df_copy.smoking == False].groupby(['country', 'smoking_passive_freq'], observed = False).size().reset_index(name = 'count')
sns.barplot(x = 'country', y = 'count', data = smoking_counts, hue = 'smoking_passive_freq', dodge = True, palette = 'cool')
ax.set_title('Number of cigarettes a day as stated by smokers')
plt.show()



# In[9]:


# kouření (aktivní, pasivní) × koncentrace latky
fig, axes = plt.subplots(3, 4, figsize=(20, 20), sharey = False)  
fig.suptitle('Smoking status x PAH-metabolites', fontsize=16)
fig.subplots_adjust(hspace=0.6, wspace = 0.3)
print('')

for idx, substance in enumerate(pah_columns[:12]):
    row = idx // 4
    col = idx % 4

    
    # smoking + smoking passive combined in one dataframe
    filtered_df_0 = df[['smoking', 'smoking_passive', substance+'_impcrtlog10']].copy()
    filtered_df_0.dropna(subset = [substance+'_impcrtlog10'], inplace = True)   # vyhodit radky s prazdnymi hodnotami ve sloupci substance+'_impcrtlog10'
    filtered_df_0[['smoking', 'smoking_passive']] = filtered_df_0[['smoking', 'smoking_passive']].fillna(value = False)    # nevyplnene hodnoty ve sloupcich smoking a smoking_passive se zmeni na False     

    filtered_df_0['smoking_status'] = filtered_df_0.apply(lambda row: 'smoker' if row['smoking'] == True else ('passive_smoker' if row['smoking_passive'] == True else 'non-smoker'), axis = 1)
    filtered_df_0['smoking_status'] = pd.Categorical(filtered_df_0['smoking_status'], categories=['smoker', 'passive_smoker', 'non-smoker'], ordered=True)
    
    counts = filtered_df_0.groupby(['smoking_status'], observed = False).size()
    if sum(counts.values) > 0:
        print(f'{substance}: post-hoc test results')
        print('====================================')

        # boxplot
        sns.boxplot(ax = axes[row, col],
                         data = filtered_df_0, 
                         x='smoking_status', 
                         y=substance+'_impcrtlog10', 
                         color = metabolites_colors.get(substance), 
                         whis = (5, 95))
    
        # ANOVA with log-transformed values
        anova_results = stats.f_oneway(
            filtered_df_0[filtered_df_0['smoking_status'] == 'smoker'][substance+'_impcrtlog10'],
            filtered_df_0[filtered_df_0['smoking_status'] == 'non-smoker'][substance+'_impcrtlog10'],
            filtered_df_0[filtered_df_0['smoking_status'] == 'passive_smoker'][substance+'_impcrtlog10']
        )

        # Tukey's HSD post-hoc test
        tukey = pairwise_tukeyhsd(endog = filtered_df_0[substance+'_impcrtlog10'], groups = filtered_df_0['smoking_status'], alpha=0.05)
        tukey_summary = tukey.summary()
        
        tukey_headers = tukey_summary.data[0] 
        tukey_data = tukey_summary.data[1:]    
        tukey_summary = pd.DataFrame(tukey_data, columns=tukey_headers)
    
        pairs = [(row["group1"], row["group2"]) for _, row in tukey_summary.iterrows()]
        p_values = [row["p-adj"] for _, row in tukey_summary.iterrows()]
    
        # post-hoc test annonations
        annotator = Annotator(axes[row, col], pairs = pairs, data=filtered_df_0, x="smoking_status", y=substance+'_impcrtlog10')
        annotator.configure(text_format="star", loc="outside", verbose = 2)
        annotator.set_pvalues(p_values)
        annotator.annotate()
        print('')
    
        # title
        axes[row, col].set_title(substance, pad=90)

        # x-axis
        n_labels = {'smoker': counts.get('smoker', 0), 'passive_sm': counts.get('passive_smoker', 0), 'non-smoker': counts.get('non-smoker', 0), }
        xticks = axes[row, col].get_xticks()
        axes[row, col].set_xticks(xticks)
        axes[row, col].set_xticklabels([f"{key}\n n = {value}" for key, value in n_labels.items()])
        axes[row, col].set_xlabel('')

        # y-axis in linear units
        y_ticks = axes[row, col].get_yticks()  # Get current y-tick positions (log-scale)
        y_labels = [round(np.power(10, tick), 2) for tick in y_ticks]  # Convert and round to 2 decimals
        axes[row, col].set_yticks(y_ticks)  # Keep positions
        axes[row, col].set_yticklabels(y_labels)  # Replace labels with linear equivalents
        axes[row, col].set_ylabel("CRT-standardized, imputed, log10-transformed values")

plt.show()



# In[10]:


# role promenne smoking_24h a smoking_passive24h: studie tri statu ['CH', 'HR', 'PT'], ktere maji prislusna data vyplnena

def categorize(row):
    # vyrobi kategorie k porovnani
    if row['smoking_24h'] == True and row['smoking'] == True:
        return 'smoker, 24h: True'
    elif row['smoking_24h'] != True and row['smoking'] == True:
        return 'smoker, 24h: False'
    elif row['smoking'] != True and row['smoking_passive'] == True and row['smoking_passive24h'] == True:
        return 'passive smoker, 24h: True'
    elif row['smoking'] != True and row['smoking_passive'] == True and row['smoking_passive24h'] != True:
        return 'passive smoker, 24h: False'
    elif row['smoking'] != True and row['smoking_24h'] != True and row['smoking_passive24h'] != True:
        return 'nonsmoker'
    else:
        return None

fig, axes = plt.subplots(3, 4, figsize=(20, 28), sharey = False)  
fig.suptitle('Various smoking behaviors compared to nonsmokers. Data from CH, HR and PT. Only significant post-hoc results are shown', fontsize=16)
fig.subplots_adjust(hspace=0.9, wspace = 0.3)
print('')

for idx, substance in enumerate(pah_columns[:12]):
    row = idx // 4
    col = idx % 4

    # get dataframe with relevant columns
    filtered_df_1 = df[df['country'].isin(['CH', 'HR', 'PT'])].copy()  # countries which have data in smoking_24h and smoking_passive24h 
    filtered_df_1 = filtered_df_1[['smoking', 'smoking_passive', 'smoking_24h', 'smoking_passive24h', substance+'_impcrtlog10']]
    filtered_df_1[['smoking', 'smoking_passive', 'smoking_24h', 'smoking_passive24h']] = filtered_df_1[['smoking', 'smoking_passive', 'smoking_24h', 'smoking_passive24h']].fillna(value = False)  
    
    filtered_df_1.dropna(subset = [substance+'_impcrtlog10'], inplace = True)

    filtered_df_1['groups'] = filtered_df_1.apply(categorize, axis=1)      
    categories = ['smoker, 24h: True', 'smoker, 24h: False', 'passive smoker, 24h: True', 'passive smoker, 24h: False', 'nonsmoker']
    filtered_df_1['groups'] = pd.Categorical(filtered_df_1['groups'], categories = categories, ordered=True)  # kvuli poradi boxplotu v grafu
   
    counts = filtered_df_1.groupby(['groups'],observed = False).size()
    if sum(counts.values) > 0:  # pokud ma latka vubec namerena nejaka data, a ne jen inputovana

        # plot
        sns.boxplot(ax = axes[row, col],
                    data = filtered_df_1, 
                    x='groups', 
                    y=substance+'_impcrtlog10', 
                    color = metabolites_colors[substance], 
                    whis = (5, 95))

        # ANOVA with log-transformed values
        anova_results = stats.f_oneway(
            filtered_df_1[filtered_df_1['groups'] == 'smoker, 24h: True'][substance+'_impcrtlog10'],
            filtered_df_1[filtered_df_1['groups'] == 'smoker, 24h: False'][substance+'_impcrtlog10'],
            filtered_df_1[filtered_df_1['groups'] == 'passive smoker, 24h: True'][substance+'_impcrtlog10'],
            filtered_df_1[filtered_df_1['groups'] == 'passive smoker, 24h: False'][substance+'_impcrtlog10'],
            filtered_df_1[filtered_df_1['groups'] == 'nonsmoker'][substance+'_impcrtlog10']
        )

        # Dunnett's version of Tukey's test, comparing non-smokers to each infividual category
        control_group = 'nonsmoker'   
        control_values = filtered_df_1.loc[filtered_df_1['groups'] == control_group][substance+'_impcrtlog10'].to_numpy()  
        noncontrol_ordered_values = [filtered_df_1.loc[filtered_df_1['groups'] == category, substance+'_impcrtlog10'].to_numpy()
                                     for category in categories if category != control_group]

        def discard_empty_categories(smoking_categories, noncontrol_values):  # skip the test in case of no data
            categories_filtered = copy.deepcopy(smoking_categories)
            noncontrol_filtered_values = copy.deepcopy(noncontrol_values)

            categories_filtered = [category for i, category in enumerate(categories_filtered[:-1]) if len(noncontrol_values[i]) > 0]
            noncontrol_filtered_values = [values for values in noncontrol_values if len(values) > 0]
            return categories_filtered, noncontrol_filtered_values
            
        categories_filtered, noncontrol_ordered_values = discard_empty_categories(categories, noncontrol_ordered_values)
        
        results = stats.dunnett(*noncontrol_ordered_values, control=control_values, alternative='two-sided')
        print('')
        print(f'{substance}, post-hoc test results:')
        print('====================================')

        significant_pairs = []
        significant_pvalues = []
        ci = results.confidence_interval(confidence_level=0.95)
        for i, group in enumerate(categories_filtered):
            if group != control_group:
                print(f"Comparison: {group} vs {control_group}")
                print(f"Statistic: {results.statistic[i]}, p-value: {results.pvalue[i]}")
                print(f"Confidence interval: Low = {ci[0][i]}, High = {ci[1][i]}")
                print("-" * 30)

                if results.pvalue[i] < 0.05:
                    significant_pairs.append((group, control_group))
                    significant_pvalues.append(results.pvalue[i])

        annotator = Annotator(ax = axes[row, col], pairs = significant_pairs, data=filtered_df_1, x="groups", y=substance+'_impcrtlog10')
        annotator.configure(text_format="star", loc="outside", fontsize=10)
        annotator.set_pvalues(significant_pvalues)
        annotator.annotate()
        print('')

        # title 
        axes[row, col].set_title(substance, pad = 90)
        
        # x-axis
        n_labels = {category: counts.get(category, 0) for category in categories}
        xticks = list(range(0,5))
        axes[row, col].set_xticks(xticks)
        axes[row, col].set_xticklabels([f"{key}\n n = {value}" for key, value in n_labels.items()], rotation = 45)
        axes[row, col].set_xlabel('', visible = False)

        # y-axis in linear units
        y_ticks = axes[row, col].get_yticks()  # Get current y-tick positions (log-scale)
        y_labels = [round(np.power(10, tick), 2) for tick in y_ticks]  # Convert from log to linear and round to 2 decimals
        axes[row, col].set_yticks(y_ticks)  # Keep positions
        axes[row, col].set_yticklabels(y_labels)  # Replace labels with linear equivalents
        axes[row, col].set_ylabel("CRT-standardized, imputed, log10-transformed values")
    
    # title for empty subplots
    axes[row, col].set_title(substance, pad = 90)
    
plt.show()


# In[11]:


# intenzita kouřní × koncentrace, hodnoty standardizovány na kreatinin, log. osa y

fig, axes = plt.subplots(3, 4, figsize=(18, 15))
fig.suptitle('Number of cigarettes a day × compound concentration', fontsize=12)

for idx, substance in enumerate(pah_columns[:12]):
    row = idx // 4
    col = idx % 4

    # scatterplot
    df_part = df[df.smoking == True][['smoking_cigday', substance+'_impcrtlog10']].dropna()
    x = df_part['smoking_cigday'] = df_part['smoking_cigday'].astype('float')
    y = df_part[substance+'_impcrtlog10'].astype('float')
    sns.regplot(ax = axes[row, col], x=x, y=y, scatter=True, fit_reg=True, ci = None, color = metabolites_colors[substance], line_kws={"color": "black"})

    # Spearmannuv korelacni koeficient
    spearman_corr, p_value = stats.spearmanr(df_part['smoking_cigday'], df_part[substance+'_impcrtlog10'])
    spearman_corr = round(spearman_corr, 2)
    if p_value < 0.01:  # If p-value is very small
        rounded_p_value = f"{p_value:.2e}"  # Keep in scientific notation
    else:
        rounded_p_value = f"{p_value:.2f}"  # Round to 2 decimal places

    axes[row, col].set_title(f'{substance}')
    axes[row, col].set_xlabel(f"Spearman coeff.: {spearman_corr}, P-value: {rounded_p_value}")
    axes[row, col].set_ylabel("CRT-standardized, imputed, log10-transformed values")
    

plt.tight_layout()
plt.show()


