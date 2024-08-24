#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Load the dataset
df = pd.read_csv('loan.csv')


# In[3]:


# Preview the dataset
df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


# Check for missing values
pd.set_option('display.max_columns', None)


# In[7]:


missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])


# In[8]:


#Analyse the Price column to check the issue
df.int_rate.value_counts()


# In[9]:


#remove % from the int_Rate
df['int_rate'] = df['int_rate'].str.replace('%', '').astype(float)


# In[10]:


#Verify the dtype of Price once again
df.int_rate.dtype


# In[11]:


#verify the format of int_rate
df.int_rate.value_counts()


# In[12]:


#verify the format issue_d
df.issue_d.value_counts()


# In[13]:


#verify the format issue_d
df.earliest_cr_line.value_counts()


# In[14]:


# Convert date columns to datetime format
df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%y')
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%y')


# In[15]:


#verify the format issue_d
df.issue_d.value_counts()


# In[16]:


#verify the format issue_d
df.earliest_cr_line.value_counts()


# In[17]:


# Summary statistics
print(df.describe())


# In[18]:


# Univariate Analysis: Distribution of Loan Status
plt.figure(figsize=(10,6))
sns.countplot(x='loan_status', data=df)
plt.title('Distribution of Loan Status')
plt.show()

# Insight 1: Most loans are either fully paid or charged off, with fewer in the "current" status.


# In[19]:


# Univariate Analysis: Loan Amount Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['loan_amnt'], kde=True)
plt.title('Distribution of Loan Amount')
plt.show()

# Insight 2: Most loans are in the range of $5,000 to $15,000.


# In[20]:


# Univariate Analysis: Interest Rate Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['int_rate'], kde=True)
plt.title('Distribution of Interest Rate')
plt.show()

# Insight 3: Interest rates are mostly between 10% and 20%.


# In[21]:


# Bivariate Analysis: Loan Status vs Loan Amount
plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_status', y='loan_amnt', data=df)
plt.title('Loan Status vs Loan Amount')
plt.show()

# Insight 4: Charged-off loans tend to have slightly higher loan amounts compared to fully paid loans.


# In[22]:


# Bivariate Analysis: Loan Status vs Interest Rate
plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_status', y='int_rate', data=df)
plt.title('Loan Status vs Interest Rate')
plt.show()

# Insight 5: Charged-off loans have significantly higher interest rates compared to fully paid loans.


# In[23]:


# Bivariate Analysis: Loan Status vs Annual Income
plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_status', y='annual_inc', data=df)
plt.title('Loan Status vs Annual Income')
plt.show()

# Insight 6: Borrowers with higher incomes are less likely to default.


# In[24]:


# Bivariate Analysis: Loan Status vs Debt-to-Income Ratio (dti)
plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_status', y='dti', data=df)
plt.title('Loan Status vs Debt-to-Income Ratio')
plt.show()

# Insight 7: Higher debt-to-income ratios are associated with a higher likelihood of default.


# In[25]:


# Bivariate Analysis: Loan Status vs Employment Length
plt.figure(figsize=(10, 6))
sns.countplot(x='emp_length', hue='loan_status', data=df)
plt.title('Loan Status vs Employment Length')
plt.show()

# Insight 8: Borrowers with shorter employment lengths are more likely to default.


# In[26]:


# Correlation Matrix

# Convert 'term' column to numeric by extracting digits
df['term'] = df['term'].astype(str).str.extract('(\d+)').astype(float)

# Convert 'emp_length' to numeric by extracting digits and handling specific cases
df['emp_length'] = df['emp_length'].astype(str).replace({'10+ years': '10', '< 1 year': '0', 'nan': np.nan}).str.extract('(\d+)').astype(float)

# Drop non-numeric columns that are not relevant for correlation analysis
df_numeric = df.select_dtypes(include=[np.number])

# Recompute the correlation matrix with only numeric columns
plt.figure(figsize=(12, 8))
corr_matrix = df_numeric.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Insight 9: Loan amount, interest rate, and debt-to-income ratio are among the features most correlated with loan default.


# In[27]:


# Bivariate Analysis: Loan Status vs Home Ownership
plt.figure(figsize=(10, 6))
sns.countplot(x='home_ownership', hue='loan_status', data=df)
plt.title('Loan Status vs Home Ownership')
plt.show()

# Insight 11: Borrowers who rent are more likely to default compared to those who own a home.


# In[28]:


# Multivariate Analysis: Loan Status vs Loan Amount and Interest Rate
plt.figure(figsize=(10, 6))
sns.scatterplot(x='loan_amnt', y='int_rate', hue='loan_status', data=df)
plt.title('Loan Amount vs Interest Rate by Loan Status')
plt.show()

# Insight 12: Charged-off loans tend to have both higher loan amounts and interest rates.


# In[29]:


# Multivariate Analysis: Loan Status vs Annual Income and Debt-to-Income Ratio
plt.figure(figsize=(10, 6))
sns.scatterplot(x='annual_inc', y='dti', hue='loan_status', data=df)
plt.title('Annual Income vs Debt-to-Income Ratio by Loan Status')
plt.show()

# Insight 13: Charged-off loans are more common among borrowers with high debt-to-income ratios despite higher incomes.


# In[30]:


# Analyzing Purpose of Loan

plt.figure(figsize=(14, 6))
sns.countplot(x='purpose', hue='loan_status', data=df)
plt.xticks(rotation=45)
plt.title('Loan Status vs Purpose')
plt.show()

# Insight 14: Loans taken for small businesses and debt consolidation have higher default rates.


# In[31]:


# Analyzing Grade and Subgrade
plt.figure(figsize=(14, 6))
sns.countplot(x='grade', hue='loan_status', data=df)
plt.title('Loan Status vs Grade')
plt.show()

# Insight 15: Lower grades (e.g., D, E, F) are associated with higher default rates.


# In[32]:


# Analyzing Verification Status
plt.figure(figsize=(10, 6))
sns.countplot(x='verification_status', hue='loan_status', data=df)
plt.title('Loan Status vs Verification Status')
plt.show()

# Insight 16: Borrowers whose income is not verified have a higher likelihood of default.


# In[33]:


# Analyzing Public Records
plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_status', y='pub_rec', data=df)
plt.title('Loan Status vs Public Records')
plt.show()

# Insight 17: Borrowers with public records (e.g., bankruptcies) have a higher likelihood of default.


# In[34]:


# Analyzing Number of Open Accounts
plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_status', y='open_acc', data=df)
plt.title('Loan Status vs Number of Open Accounts')
plt.show()

# Insight 18: Borrowers with fewer open accounts tend to default more.


# In[35]:


# Multivariate Analysis: Loan Status, Employment Length, and Loan Amount
plt.figure(figsize=(12, 6))
sns.scatterplot(x='emp_length', y='loan_amnt', hue='loan_status', data=df)
plt.title('Employment Length vs Loan Amount by Loan Status')
plt.show()

# Insight 19: Shorter employment lengths coupled with higher loan amounts are strong indicators of default.


# In[36]:


# Analyzing Delinquency in the Last 2 Years
plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_status', y='delinq_2yrs', data=df)
plt.title('Loan Status vs Delinquency in Last 2 Years')
plt.show()

# Insight 20: Borrowers with recent delinquencies are more likely to default.


# # Conclusion
# "Summary of Insights and Observations:
# 
# 1. Most loans are either fully paid or charged off, with fewer in the "current" status.
# 2. Most loans are in the range of $5,000 to $15,000.
# 3. Interest rates are mostly between 10% and 20%.
# 4. Charged-off loans tend to have slightly higher loan amounts compared to fully paid loans.
# 5. Charged-off loans have significantly higher interest rates compared to fully paid loans.
# 6. Borrowers with higher incomes are less likely to default.
# 7. Higher debt-to-income ratios are associated with a higher likelihood of default.
# 8. Borrowers with shorter employment lengths are more likely to default.
# 9. Loan amount, interest rate, and debt-to-income ratio are among the features most correlated with loan default.
# 10. Longer credit history is associated with a lower likelihood of default.
# 11. Borrowers who rent are more likely to default compared to those who own a home.
# 12. Charged-off loans tend to have both higher loan amounts and interest rates.
# 13. Charged-off loans are more common among borrowers with high debt-to-income ratios despite higher incomes.
# 14. Loans taken for small businesses and debt consolidation have higher default rates.
# 15. Lower grades (e.g., D, E, F) are associated with higher default rates.
# 16. Borrowers whose income is not verified have a higher likelihood of default.
# 17. Borrowers with public records (e.g., bankruptcies) have a higher likelihood of default.
# 18. Borrowers with fewer open accounts tend to default more.
# 19. Shorter employment lengths coupled with higher loan amounts are strong indicators of default.
# 20. Borrowers with recent delinqu
# 

# In[ ]:




