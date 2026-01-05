#!/usr/bin/env python
# coding: utf-8

# In[1]:


import missingno as msno
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np, statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
import seaborn 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LinearRegression,Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('future.no_silent_downcasting', True)

allowed_nans = ['', 'NA']
df = pd.read_csv("C:\\Users\\Lynh\\OneDrive\\Desktop\\School\\Masters\\Term 5\Capstone\\student_depression_dataset.csv", na_values=allowed_nans, keep_default_na=False)
df.head(5)


# In[2]:


df.shape


# In[3]:


#counts dupes. there are 27901 False which means there are no dupes.
print(df.duplicated().value_counts())


# In[4]:


#sum of nulls
df.isnull().sum()


# In[5]:


#info of table
df.info()


# In[6]:


res = list(dict.fromkeys(df['Gender']))
print('Gender: ', res)


# In[7]:


res = list(dict.fromkeys(df['Age']))
sort = sorted(res)
print('Age: ', sort)


# In[8]:


res = list(dict.fromkeys(df['Profession']))
print('Profession: ', res)


# In[9]:


res = list(dict.fromkeys(df['Academic Pressure']))
print('Academic Pressure: ', res)


# In[10]:


res = list(dict.fromkeys(df['Work Pressure']))
print('Work Pressure: ', res)


# In[11]:


res = list(dict.fromkeys(df['Study Satisfaction']))
print('Study Satisfaction: ', res)


# In[12]:


res = list(dict.fromkeys(df['Job Satisfaction']))
print('Job Satisfaction: ', res)


# In[13]:


res = list(dict.fromkeys(df['Sleep Duration']))
print('Sleep Duration: ', res)


# In[14]:


res = list(dict.fromkeys(df['Dietary Habits']))
print('Dietary Habits: ', res)


# In[15]:


res = list(dict.fromkeys(df['Degree']))
sort = sorted(res)
print('Degree: ', sort)


# In[16]:


res = list(dict.fromkeys(df['Work/Study Hours']))
print('Work/Study Hours: ', res)


# In[17]:


res = list(dict.fromkeys(df['Financial Stress']))
print('Financial Stress: ', res)


# In[18]:


df.loc[df['Financial Stress'] == '?', 'Financial Stress'] = 0


# In[19]:


df = df.astype({'Financial Stress': float}) #convert to column to float. it was initially string because of the '?' value
df.head()


# In[20]:


res = list(dict.fromkeys(df['Family History of Mental Illness']))
print('Family History of Mental Illness: ', res)


# In[21]:


#drop variables that are not a part of the research
df=df.drop(['id', 'City', 'CGPA', 'Have you ever had suicidal thoughts ?'], axis=1)
df.head()


# In[22]:


#data encoding for cat-binary columns
bin_cols =['Family History of Mental Illness']
bin_dict = {'No':0, 'Yes':1}
for col in bin_cols:
    df[col] = df[col].replace(bin_dict)

bin_cols2 = ['Gender']
bin_dict2 = {'Male': 0, 'Female': 1}
for col2 in bin_cols2:
    df[col2] = df[col2].replace(bin_dict2)

print(df.head())


# In[23]:


#categorize by age range
def age_range(df):
    if df['Age'] >= 18.0 and df['Age'] <= 29.0: #age range 18-29
        return 0
    elif df['Age'] >= 30.0 and df['Age'] <= 39.0: #age range 30-39
        return 1
    elif df['Age'] >= 40.0 and df['Age'] <= 49.0: #age range 40-49
        return 2
    else:
        return 3

df['Age_Range'] = df.apply(age_range, axis = 1)


# In[24]:


#drop Age
df=df.drop(['Age'], axis=1)
df.head()


# In[25]:


#categorize by degree type
def degree_type(df):
    if df['Degree'] == 'Class 12': #high school
        return 0
    elif df['Degree'] in ('B.Arch', 'B.Com', 'B.Ed', 'B.Pharm', 'B.Tech', 'BA', 'BBA', 'BCA', 'BE', 'BHM', 'BSc', 'LLB'): #bachelors
        return 1
    elif df['Degree'] in ('LLM', 'M.Com', 'M.Ed', 'M.Pharm', 'M.Tech', 'MA', 'MBA', 'MBBS', 'MCA', 'ME', 'MHM'): #masters
        return 2
    elif df['Degree'] in ('MD', 'PhD'): #doctorates
        return 3
    else: #others
        return 4

df['Degree_Type'] = df.apply(degree_type, axis = 1)


# In[26]:


#drop Degree
df=df.drop(['Degree'], axis=1)
df.head()


# In[27]:


#encoding categorical variables, drop_first true to avoid multicollinearity
df = pd.get_dummies(df, columns = ['Profession', 'Dietary Habits', 'Sleep Duration'], drop_first = True)
df.replace({False:0, True:1}, inplace=True)
df.head()


# In[28]:


#save prepared dataset to CSV
#df.to_csv(r'C:\\Users\Lynh\OneDrive\Desktop\School\Masters\Term 5\Capstone\capstone_prepared_dataset.csv')


# In[29]:


print(df.columns)


# In[30]:


#VIF before fitting for regression model
x=df[['Gender', 'Academic Pressure', 'Work Pressure', 'Study Satisfaction',
       'Job Satisfaction', 'Work/Study Hours', 'Financial Stress',
       'Family History of Mental Illness', 'Depression', 'Age_Range',
       'Degree_Type', "Profession_'Content Writer'",
       "Profession_'Digital Marketer'", "Profession_'Educational Consultant'",
       "Profession_'UX/UI Designer'", 'Profession_Architect',
       'Profession_Chef', 'Profession_Doctor', 'Profession_Entrepreneur',
       'Profession_Lawyer', 'Profession_Manager', 'Profession_Pharmacist',
       'Profession_Student', 'Profession_Teacher', 'Dietary Habits_Moderate',
       'Dietary Habits_Others', 'Dietary Habits_Unhealthy',
       "Sleep Duration_'7-8 hours'", "Sleep Duration_'Less than 5 hours'",
       "Sleep Duration_'More than 8 hours'", 'Sleep Duration_Others']]
x = x.apply(pd.to_numeric, errors='coerce')
y=df['Depression']
vif_data = pd.DataFrame()
vif_data['Variable'] = x.columns
vif_data['VIF'] = [variance_inflation_factor(x.values, i)
                   for i in range(len(x.columns))]
print(vif_data)


# In[31]:


#X and y values
X = df[['Gender', 'Academic Pressure', 'Work Pressure', 'Study Satisfaction',
       'Job Satisfaction', 'Work/Study Hours', 'Financial Stress',
       'Family History of Mental Illness', 'Depression', 'Age_Range',
       'Degree_Type', "Profession_'Content Writer'",
       "Profession_'Digital Marketer'", "Profession_'Educational Consultant'",
       "Profession_'UX/UI Designer'", 'Profession_Architect',
       'Profession_Chef', 'Profession_Doctor', 'Profession_Entrepreneur',
       'Profession_Lawyer', 'Profession_Manager', 'Profession_Pharmacist',
       'Profession_Student', 'Profession_Teacher', 'Dietary Habits_Moderate',
       'Dietary Habits_Others', 'Dietary Habits_Unhealthy',
       "Sleep Duration_'7-8 hours'", "Sleep Duration_'Less than 5 hours'",
       "Sleep Duration_'More than 8 hours'", 'Sleep Duration_Others']].values
features= df.columns[0:31]
target = df.columns[-1]

#split into test and training 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

print("The dimension of X_train is {}".format(X_train.shape))
print("The dimension of X_test is {}".format(X_test.shape))

#Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[32]:


#lr Model
lr = LinearRegression()
lr.fit(X_train, y_train)
#train and test score for linear regression
train_score_lr = lr.score(X_train, y_train)
test_score_lr = lr.score(X_test, y_test)
#print scores
print("Linear Regression Model............................................\n")
print("The train score for lr model is {}".format(train_score_lr))
print("The test score for lr model is {}".format(test_score_lr))


#Ridge Regression Model
ridgeReg = Ridge(alpha=10)
ridgeReg.fit(X_train,y_train)
#train and test scorefor ridge regression
train_score_ridge = ridgeReg.score(X_train, y_train)
test_score_ridge = ridgeReg.score(X_test, y_test)
#print scores
print("\nRidge Model............................................\n")
print("The train score for ridge model is {}".format(train_score_ridge))
print("The test score for ridge model is {}".format(test_score_ridge))


# In[33]:


plt.figure(figsize = (20, 10))
plt.plot(features,ridgeReg.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 10$',zorder=7)
#plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$')
plt.plot(features,lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')
plt.xticks(rotation = 90)
plt.legend()
plt.show()


# In[34]:


#initial multiple linear regression
label = "Depression"
y = df['Depression']
x=df.drop(columns='Depression')
x = sm.add_constant(x)
x = x.astype(float)

model = sm.OLS(y,x.astype(int))
results=model.fit()
print(results.summary())


# In[35]:


#residual standard error calc - initial 
model=sm.OLS(y,x).fit()
residuals = model.resid
n = len(residuals)
p = x.shape[1]
ssr = np.sum(residuals**2)
rse = np.sqrt(ssr / (n - p))

print(f"Residual Standard Error: {rse:.4f}")


# In[36]:


#reduced linear regression
def significant_variables(x, y, significant_value=0.05):
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)
    while True:
        model = sm.OLS(y, x).fit()
        p_values = model.pvalues
        max_p_value = p_values.max()
        if max_p_value > significant_value:
            p_values_series = pd.Series(p_values, index=x.columns)
            insignificant = p_values_series.idxmax()
            x = x.drop(columns=insignificant)
        else:
            break
    return x, model


x_optimized, reduced_model = significant_variables(x, y)
print(reduced_model.summary())


# In[37]:


#residual standard error calc - reduced
n = len(residuals)
p = x.shape[1]
ssr = np.sum(residuals**2)
rse = np.sqrt(ssr / (n - p))

print(f"Residual Standard Error: {rse:.4f}")


# In[38]:


reduce = reduced_model.pvalues.index.tolist()
print(reduce)


# In[39]:


#scatterplot of residuals
y = df['Depression']
x = df[['Academic Pressure', 'Study Satisfaction', 'Work/Study Hours', 'Financial Stress', 
        'Family History of Mental Illness', 'Age_Range', 'Degree_Type', 'Dietary Habits_Moderate', 
        'Dietary Habits_Others', 'Dietary Habits_Unhealthy', "Sleep Duration_'7-8 hours'", 
        "Sleep Duration_'Less than 5 hours'", "Sleep Duration_'More than 8 hours'"]]
y = pd.to_numeric(df['Depression'], errors='coerce')
x = x.apply(pd.to_numeric, errors='coerce')
x = sm.add_constant(x)
model=sm.OLS(y,x).fit()
residuals = model.resid
predictions = model.fittedvalues
plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Scatterplot of Residuals')
plt.show()


# In[40]:


#Q-Q plot of residuals
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.title('Q-Q Plot of Residuals')
plt.show()


# In[ ]:




