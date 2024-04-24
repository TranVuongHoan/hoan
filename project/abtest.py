#INTRODUCTION
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
df = pd.read_csv("c:\Users\hoan\Downloads\ab_data (1).csv")

#look at top few rows
df.head()

#find the number of rows and column
df.shape

#number of unique users
df['user_id'].nunique()

#proportion of users converted
df.converted.mean()

#number of times new_page and treatment dont match
line_1 = df.query('group == "treatment" and landing_page == "old_page"').count()
line_2 = df.query('group == "control" and landing_page == "new_page"').count()
line_1 + line_2

#check if any rows have missing values
df.isnull().sum()

#create a new dataset to handle rows does not match
df2 = df[((df['group'] == 'treatment') == (df['landing_page'] == 'new_page')) == True]
df2.head()

# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]
df2['converted'].value_counts()

#count unique user_id in df2
df2['user_id'].nunique()

#find 1 user_id repeated in df2
df2[df2.duplicated('user_id')]

#row information for repeat user_id
df2[df2.duplicated(['user_id'], keep = False)]

#remove rows duplicated user_Id, keep dframe as df2
#drop_Duplicate
df2.drop_duplicates(keep='first')
df2.duplicated().sum()

#probability of individual converting regardingless of page they receive
df2['converted'].mean()

#given individual in the "control" group, what is the probability they converted
df.groupby('group').mean()

#probability that an individual received the new page
df2.landing_page.value_counts()[0]/len(df2)

#part 2
#find conversion rate "p_new" under the null
p_new = df2[df2['landing_page'] == 'new_page'].converted.mean()
print(p_new)

#find conversion rate "p_old" under the null
p_old = df2[df2['landing_page']== 'old_page'].converted.mean()
print(p_old)
p_avg = (p_new + p_old)/2
print(p_avg)
n_new = df2['landing_page'].value_counts()
print(n_new)

#what is n_new and n_old the number of individuals in treatment group
n_new, n_old = df2['landing_page'].value_counts()
print(n_new)
print(n_old)

#Simulate "n_new" transactions with a conversion rate "p_new" of under the null. Store these 1's and 0's in new_page_converted.
new_page_converted = np.random.choice([0,1], size = n_new, p=(p_avg, 1-p_avg))
print(new_page_converted)
new_page_converted.mean()

#Simulate "n_old" transactions with a conversion rate "p_old" of under the null. Store these 1's and 0's in old_page_converted.
old_page_converted = np.random.choice([0,1], size = n_old, p = (p_avg, 1-p_avg))
print(old_page_converted)
old_page_converted.mean()

#find p_new - p_old for simulated values 
actual_diff = new_page_converted.mean() - old_page_converted.mean()
print(actual_diff)

#create 10000 p_new -p_old values using the same simulation process, store 10000 values in numpy array call p_diffs
p_diffs = []
new = np.random.binomial(n_new, p_avg, 10000)/n_new
old = np.random.binomial(n_old, p_avg, 10000)/n_old
p_diffs = new - old

#plot a histogram of p_diffs
p_diffs = np.array(p_diffs)
plt.hist(p_diffs)

#compute actual conversion rate
# number of landing new page and converted  / number of landing new page
converted_new = df2.query('converted == 1 and landing_page== "new_page"')['user_id'].nunique()
actual_new = float(converted_new) / float(n_new)

# number of landing old page and converted  / number of landing old page
converted_old = df2.query('converted == 1 and landing_page== "old_page"')['user_id'].nunique()
actual_old = float(converted_old) / float(n_old)

#observed difference in converted rate
obs_diff = actual_diff = actual_new - actual_old
obs_diff

# create distribution under the null hypothesis
null_vals = np.random.normal(0, p_diffs.std(), p_diffs.size)

#Plot Null distribution
plt.hist(null_vals)

#Plot vertical line for observed statistic
plt.axvline(x=obs_diff,color ='black')

import statsmodels.api as sm
convert_old = df2.query('converted == 1 and landing_page== "old_page"').user_id.nunique()
convert_new = converted_old = df2.query('converted == 1 and landing_page== "new_page"').user_id.nunique()
n_old = df2.query('landing_page == "old_page"')['user_id'].nunique()
n_new = df2.query('landing_page == "new_page"')['user_id'].nunique()

# compute the sm.stats.proportions_ztest using the alternative
z_score, p_value = sm.stats.proportions_ztest(np.array([convert_new,convert_old]),np.array([n_new,n_old]), alternative = 'larger')
z_score, p_value
