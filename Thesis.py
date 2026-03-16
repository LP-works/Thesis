#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data
df_raw = pd.read_excel(r'C:/Users//S Suresh kumar//OneDrive//Desktop//Lekshmi//BONN-L//Thesis_data(AutoRecovered).xlsx')
print("Raw data shape:", df_raw.shape)

# CLEANING STRATEGY
# 1. Core variables only (required)
core_vars = ['Log_Price = ln([Price (in $)])', 'Log_Warranty_Presence = ln(Warranty Presence + 1)', 
             'Log_Warranty_Duration = In(Warranty duration (in months) + 1)', 'Trusted Seller']

# 2. Drop ONLY core missing
df = df_raw.dropna(subset=core_vars)

# 3. Fix data types
for var in core_vars:
    df[var] = pd.to_numeric(df[var], errors='coerce')
df = df.dropna(subset=core_vars)  # Final core check

# 4. IMPUTE optional variables (ratings, condition)
df['Log_Rating = ln(Rating + 1)'] = df['Log_Rating = ln(Rating + 1)'].fillna(df['Log_Rating = ln(Rating + 1)'].median())
df['Condition'] = df['Condition'].fillna(df['Condition'].median())

# Rename
df = df.rename(columns={
    'Log_Price = ln([Price (in $)])': 'Log_Price',
    'Log_Warranty_Presence = ln(Warranty Presence + 1)': 'Log_Warranty_Presence',
    'Log_Warranty_Duration = In(Warranty duration (in months) + 1)': 'Log_Warranty_Duration',
    'Log_Rating = ln(Rating + 1)': 'Log_Rating',
    'Trusted Seller': 'TrustedSeller',
    'Condition': 'Condition'
})

print("✅ FINAL N =", len(df))
print("Rows imputed (ratings/condition):", (len(df_raw) - len(df)) + df_raw['Log_Rating = ln(Rating + 1)'].isnull().sum())
print("TrustedSeller %:", round(100*df['TrustedSeller'].mean(), 1), "%")

# BASELINE MODEL
X_base = df[['Log_Warranty_Presence', 'Log_Warranty_Duration', 'TrustedSeller', 'Log_Rating', 'Condition']]
X_base = sm.add_constant(X_base)
y = df['Log_Price']
model_base = sm.OLS(y, X_base).fit(cov_type='HC1')

print("\n## BASELINE MODEL")
print(model_base.summary())

# FULL INTERACTED
interactions = ['Log_Warranty_Presence', 'Log_Warranty_Duration', 'Log_Rating', 'Condition']
X_full = pd.DataFrame(index=df.index)
X_full['const'] = 1
X_full['TrustedSeller_main'] = df['TrustedSeller']

# Non-trusted effects
for var in interactions:
    X_full[var + '_nontrusted'] = df[var] * (1 - df['TrustedSeller'])
# Trusted effects
for var in interactions:
    X_full[var + '_trusted'] = df[var] * df['TrustedSeller']

model_full = sm.OLS(y, X_full).fit(cov_type='HC1')
print("\n## FULL INTERACTED MODEL")
print(model_full.summary())

# MARGINAL EFFECTS TABLE
print("\n## MARGINAL EFFECTS TABLE")
marginals = pd.DataFrame({
    'Attribute': interactions,
    'Non-Trusted β_k (p)': [f"{model_full.params[var+'_nontrusted']:.3f} ({model_full.pvalues[var+'_nontrusted']:.3f})" 
                            for var in interactions],
    'Trusted γ_k (p)': [f"{model_full.params[var+'_trusted']:.3f} ({model_full.pvalues[var+'_trusted']:.3f})" 
                        for var in interactions],
    'Δ Effect γ_k (p)': [f"{model_full.params[var+'_trusted'] - model_full.params[var+'_nontrusted']:.3f} ({model_full.pvalues[var+'_trusted']:.3f})" 
                         for var in interactions]  # FIXED
})
print(marginals.to_string(index=False))

# JOINT F-TEST FOR WARRANTIES
print("\n## JOINT F-TEST (All warranty coefficients = 0)")
warranty_hyp = 'Log_Warranty_Presence_nontrusted = Log_Warranty_Duration_nontrusted = Log_Warranty_Presence_trusted = Log_Warranty_Duration_trusted = 0'
f_res = model_full.f_test(warranty_hyp)
print(f"F-statistic: {f_res.fvalue:.3f}")
print(f"p-value: {f_res.pvalue:.3f}")

# DESCRIPTIVE STATISTICS (Data section)
print("\n## DESCRIPTIVE STATISTICS")
desc_vars = ['Price (in $)', 'Warranty Presence', 'Warranty duration (in months)', 
             'TrustedSeller', 'Seller Rating', 'Condition']
print(df[desc_vars].describe().round(3))

# EXPORT FOR LaTeX TABLES
results_base = pd.DataFrame({
    'Variable': model_base.params.index,
    'Coef': model_base.params.round(3),
    'p_value': model_base.pvalues.round(3)
})
results_base.to_csv('baseline_table1.csv', index=False)

results_full = pd.DataFrame({
    'Variable': model_full.params.index,
    'Coef': model_full.params.round(3),
    'p_value': model_full.pvalues.round(3)
})
results_full.to_csv('interacted_table2.csv', index=False)

print(f"✅ N = {len(df)}, Baseline R² = {model_base.rsquared:.3f}, Full R² = {model_full.rsquared:.3f}")



# In[ ]:




