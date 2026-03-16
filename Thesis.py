#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data
df_raw = pd.read_excel(r'C:/Users//S Suresh kumar//OneDrive//Desktop//Lekshmi//BONN-L//Thesis_data(AutoRecovered).xlsx')
print("Raw data shape:", df_raw.shape)

# CLEANING STRATEGY
# 1. Core variables only (required)
core_vars = ['Log_Price = ln([Price (in $)])', 'Log_Warranty_Presence = ln(Warranty Presence + 1)', 
             'Log_Warranty_Duration = In(Warranty duration (in months) + 1)', 'Professional Seller']

# 2. Drop ONLY core missing
df = df_raw.dropna(subset=core_vars)

# 3. Fix data types
for var in core_vars:
    df[var] = pd.to_numeric(df[var], errors='coerce')
df = df.dropna(subset=core_vars)  # Final core check

# 4. IMPUTE optional variables (ratings, condition)
df['Log_Rating = ln(Rating + 1)'] = df['Log_Rating = ln(Rating + 1)'].fillna(df['Log_Rating = ln(Rating + 1)'].median())
df['Condition (0=fair, 1=good, 2=very good)'] = df['Condition (0=fair, 1=good, 2=very good)'].fillna(df['Condition (0=fair, 1=good, 2=very good)'].median())

# Rename
df = df.rename(columns={
    'Log_Price = ln([Price (in $)])': 'Log_Price',
    'Log_Warranty_Presence = ln(Warranty Presence + 1)': 'Log_Warranty_Presence',
    'Log_Warranty_Duration = In(Warranty duration (in months) + 1)': 'Log_Warranty_Duration',
    'Log_Rating = ln(Rating + 1)': 'Log_Rating',
    'Professional Seller': 'Professional Seller',
    'Condition (0=fair, 1=good, 2=very good)': 'Condition'
})

print("✅ FINAL N =", len(df))
print("Rows imputed (ratings/condition):", (len(df_raw) - len(df)) + df_raw['Log_Rating = ln(Rating + 1)'].isnull().sum())
print("Professional Seller %:", round(100*df['Professional Seller'].mean(), 1), "%")

# BASELINE MODEL
X_base = df[['Log_Warranty_Presence', 'Log_Warranty_Duration', 'Professional Seller', 'Log_Rating', 'Condition']]
X_base = sm.add_constant(X_base)
y = df['Log_Price']
model_base = sm.OLS(y, X_base).fit(cov_type='HC1')

print("\n## BASELINE MODEL")
print(model_base.summary())

# FULL INTERACTED
interactions = ['Log_Warranty_Presence', 'Log_Warranty_Duration', 'Log_Rating', 'Condition']
X_full = pd.DataFrame(index=df.index)
X_full['const'] = 1
X_full['Professional Seller_main'] = df['Professional Seller']

# Non-trusted effects
for var in interactions:
    X_full[var + '_Private'] = df[var] * (1 - df['Professional Seller'])
# Trusted effects
for var in interactions:
    X_full[var + '_Professional'] = df[var] * df['Professional Seller']

model_full = sm.OLS(y, X_full).fit(cov_type='HC1')
print("\n## FULL INTERACTED MODEL")
print(model_full.summary())

# MARGINAL EFFECTS TABLE
print("\n## MARGINAL EFFECTS TABLE")
marginals = pd.DataFrame({
    'Attribute': interactions,
    'Private β_k (p)': [f"{model_full.params[var+'_Private']:.3f} ({model_full.pvalues[var+'_Private']:.3f})" 
                            for var in interactions],
    'Professional γ_k (p)': [f"{model_full.params[var+'_Professional']:.3f} ({model_full.pvalues[var+'_Professional']:.3f})" 
                        for var in interactions],
    'Δ Effect γ_k (p)': [f"{model_full.params[var+'_Professional'] - model_full.params[var+'_Private']:.3f} ({model_full.pvalues[var+'_Professional']:.3f})" 
                         for var in interactions]  # FIXED
})
print(marginals.to_string(index=False))

# JOINT F-TEST FOR WARRANTIES
print("\n## JOINT F-TEST (All warranty coefficients = 0)")
warranty_hyp = 'Log_Warranty_Presence_Private = Log_Warranty_Duration_Private = Log_Warranty_Presence_Professional = Log_Warranty_Duration_Professional = 0'
f_res = model_full.f_test(warranty_hyp)
print(f"F-statistic: {f_res.fvalue:.3f}")
print(f"p-value: {f_res.pvalue:.3f}")

# DESCRIPTIVE STATISTICS (Data section)
print("\n## DESCRIPTIVE STATISTICS")
desc_vars = ['Price (in $)', 'Warranty Presence', 'Warranty duration (in months)', 
             'Professional Seller', 'Seller Rating', 'Condition']
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




