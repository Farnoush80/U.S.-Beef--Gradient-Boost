#!/usr/bin/env python
# coding: utf-8

# Intensifying meat demand is the result of sustainable economic growth since the 2009 Great Recession and decrease retail prices brought about by low animal feed costs. 
# The U.S. has a deep love for the hamburger. There are some cultures out there that find livers and other organs divine, while most consumers in the U.S., for example, have not developed a palate for those cuts. In the United States, the hamburger is by and large the most popular cut of meat. In fact, ground beef accounts for 45 percent of U.S. beef consumption with the average American eating approximately 26 pounds of hamburger a year. Forecasting total supply is crucial due to the increased meat demand in the upcoming future
# 

# In[1]:


import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error


# In[2]:



beef = pd.read_csv('https://query.data.world/s/yzcm5dndwjmjekhzuytcq7cnl4xn7i')


# In[3]:


beef.rename(columns={'U.S. population of the year':'U.S_population_of_the_year','Beginning stocks':'Beginning_stocks','Total supply':'Total_supply','Shipments to US territories':'Shipments','Ending stocks':'Ending_stocks','Food disappearance Total':'Food_disappearance_Total','Food disappearance per capita in pounds':'Food_disappearance'},inplace=True)


# In[4]:


beef.head()


# In[5]:


beef.tail()


# In[6]:


beef.info()


# Clean Missing values

# In[7]:


beef.isnull().sum().reset_index()


# In[8]:


beef.fillna(beef.median(),inplace=True)


# In[9]:


beef=beef.replace(',.','',regex=True).astype('float')


# In[10]:


beef=beef.replace(',.','',regex=True)


# In[11]:


beef.dtypes


# Data Exploration

# In[12]:


beef.groupby('Year2').Food_disappearance_Total.mean().fig=plt.figure(figsize=(10,6))
plt.bar(x=beef.Year2,height=beef.Food_disappearance)
plt.show()


# In[13]:


beef.groupby('Total_supply').Food_disappearance_Total.mean().plot(kind='line',figsize=(15,7))


# In[14]:


sns.pairplot(beef,x_vars=['Beginning_stocks','Ending_stocks'],y_vars=['Food_disappearance_Total'],size=7,kind='reg')


# In[15]:


sns.pairplot(beef,x_vars=['Beginning_stocks','Ending_stocks'],y_vars=['Food_disappearance_Total'],size=7,kind='reg')


# In[16]:


beef.drop(['Shipments','Year2'],axis=1,inplace=True)


# In[17]:


from sklearn.preprocessing import StandardScaler

N=StandardScaler()

N.fit(beef)

beef_norm=N.transform(beef)


# In[18]:


beef1=beef.iloc[0:30,:]
beef2=beef.iloc[81:105,:]


# In[19]:


beef2.describe()


# In[20]:


corr_matrix=beef2.corr()
corr_matrix['Food_disappearance_Total'].sort_values(ascending=False)


#  The U.S. biggest import sources are Australia, New Zealand, Canada and Mexico. Australia and New Zealand are a big source of lean trimmings to go into ground beef
# Cattle from Canada and Mexico may be traded on the hoof, while meat from Australia and New Zealand may come in either fresh or as a pre-packaged product.

# In[23]:


x, y = beef[['Total_supply','Imports','Ending_stocks','Beginning_stocks']].values, beef['Food_disappearance_Total'].values


# In[24]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

print(x_train.shape)
print(x_test.shape)


# In[25]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score
from sklearn.metrics import accuracy_score

# Use a Gradient Boosting algorithm
alg = GradientBoostingRegressor()

# Try these hyperparameter values
params = {
 'learning_rate': [0.1, 0.5, 1.0],
 'n_estimators' : [50, 100, 150],
 'max_depth': [3, 5, 7],
 'min_samples_split': [2, 5, 10],
 'min_samples_leaf':[1,2,4,6]
    
}

# Find the best hyperparameter combination to optimize the R2 metric
score = make_scorer(r2_score)
gridsearch = GridSearchCV(alg, params, scoring=score, cv=5, return_train_score=True)
gridsearch.fit(x_train, y_train)
print("Best parameter combination:", gridsearch.best_params_, "\n")

# Get the best model
model=gridsearch.best_estimator_
print(model, "\n")

# Evaluate the model using the test data
predictions = model.predict(x_test)
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, predictions)
print("R2:", r2)


# In[26]:


# Plot predicted vs actual
plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Food_disappearance_Total')
# overlay the regression line
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()


# In[ ]:




