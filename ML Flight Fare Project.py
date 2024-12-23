#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



import warnings
warnings.filterwarnings('ignore')
sns.set()


# # Importing Data 

# In[2]:


cd C:\Users\91774\Downloads


# In[3]:


df = pd.read_excel('Flight Fare data.xlsx')


# In[4]:


df.head()


# In[5]:


df.shape


# # set max coulmns or rows to None so we can see all columns or rows from dataset 

# In[6]:


pd.set_option('display.max_columns', None)


# In[7]:


df


# # checking data 

# In[8]:


df.isnull().sum()


# In[9]:


df.describe()


# In[10]:


df.info()


# # Cleaning Data 

# changing the format of date and seperating it in month, day not taking year because it will not impact on our result

# In[11]:


pd.to_datetime(df['Date_of_Journey'], format = '%d/%m/%Y')


# In[12]:


pd.to_datetime(df['Date_of_Journey'], format = '%d/%m/%Y').dt.day_name()


# In[13]:


df['Journey_month'] = pd.to_datetime(df['Date_of_Journey'], format = '%d/%m/%Y').dt.month


# In[14]:


df['Journey_day'] = pd.to_datetime(df['Date_of_Journey'], format = '%d/%m/%Y').dt.day


# In[15]:


df.head(1)


# In[16]:


# dropping date_of_journay column 
df.drop('Date_of_Journey', axis = 1, inplace = True)
df.head(2)


# In[17]:


# dropping route and additional_info columns
df.drop(['Route', 'Additional_Info'], axis = 1, inplace = True)


# In[18]:


df.head(2)


# In[19]:


# seperating departure time
df['Dep_Time'].value_counts()


# In[20]:


df['Dep_hour'] = pd.to_datetime(df['Dep_Time']).dt.hour


# In[21]:


df['Dep_minute'] = pd.to_datetime(df['Dep_Time']).dt.minute


# In[22]:


df.head(2)


# In[23]:


#dropping departure time column
df.drop('Dep_Time', axis = 1, inplace = True)
df.head(2)


# In[24]:


# seperating arrival time and dropping it
df['Arr_hour'] = pd.to_datetime(df['Arrival_Time']).dt.hour
df['Arr_minute'] = pd.to_datetime(df['Arrival_Time']).dt.minute


# In[25]:


df.drop('Arrival_Time', axis = 1, inplace = True)
df.head(2)


# In[26]:


# checking value count of duration
df['Duration'].value_counts()


# # creating loop to check durations contains only hour and minute and if yes then adding minute and hour it it

# In[27]:


duration = list(df['Duration'])


# In[28]:


for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + ' 0m'
        else:
            duration[i] = '0h ' + duration[i]


# # Extract hour and min from duration column and creating two new columns Duration_hours & Duration_mins 

# In[29]:


duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split('h')[0]))
    duration_mins.append(int(duration[i].split("m")[0].split()[1]))


# In[30]:


# Adding duration_hours and duration_mins list to train_data dataframe
df['Duration_hour'] = duration_hours
df['Duration_mins'] = duration_mins


# In[31]:


# dropping duration column
df.drop('Duration', axis = 1, inplace = True)


# In[32]:


df.head(3)


# # Handeling categorical data

# In[36]:


# checking values_count on airline 
df['Airline'].value_counts()


# In[37]:


#displaying price accourding to airline
df.groupby('Airline')['Price'].mean().plot(figsize = (20,10))


# # here we can say that jet airways price higher than any other airline
# checking the average price according to Airline 

# In[38]:


sns.catplot(y = "Price", x = "Airline", data = df.sort_values("Price", ascending = False), kind="boxen", height = 6, aspect = 3)
plt.show()


# # performing one hot encoding on airline, source, destination column 

# In[39]:


Airline = df['Airline']
Airline = pd.get_dummies(Airline)
Airline.head()


# In[40]:


Source = df['Source']
Source = pd.get_dummies(Source)
Source.head()


# In[41]:


Destination = df['Destination']
Destination = pd.get_dummies(Destination)
Destination.head()


# In[42]:


df.head()


# In[43]:


# removing null values 
df.dropna(inplace = True)


# # replacing categorical value in Total_stop with numeric value by manually 

# In[44]:


df['Total_Stops'].value_counts()


# In[45]:


df.replace({'non-stop': 0,
            '1 stop': 1,
            '2 stops': 2,
            '3 stops': 3,
            '4 stops': 4 },
           inplace = True)


# In[46]:


df.head()


# In[47]:


# droping airline, source, destination columns 
df.drop(['Airline', 'Source', 'Destination'], axis = 1, inplace = True)
df.head(2)


# # now concatenate all the data on which we have applied one hot encoding 

# In[48]:


df = pd.concat([df, Airline, Source, Destination], axis = 1)


# In[49]:


df.head(3)


# In[50]:


df.shape


# # Feature Selection or Dimension Reduction 

# In[51]:


#creating target and feature set
x = df.drop('Price', axis = 1)
y = df.Price


# In[57]:


#finding correlation between indipendent and dependent attributes
plt.figure(figsize=(30,20))
sns.heatmap(df.corr(), annot = True, cmap = 'RdYlGn')
plt.show()


# In[67]:


#checking if we have null values 
y.isnull().sum()


# In[68]:


#dropping null values
y.dropna(inplace = True)


# # finding important features using ExtraTreesRegressor

# In[69]:


from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(x,y)


# # printing and plotting graph of all th important features 

# In[70]:


print(selection.feature_importances_)


# In[82]:


plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=x.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# # creating training and testing dataset

# In[83]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# # applying linear regression on our training dataset 

# In[84]:


from sklearn.linear_model import LinearRegression


# In[85]:


model_li = LinearRegression()
model_li.fit(x_train, y_train)


# In[87]:


#printing training and testing score
model_li.score(x_train, y_train)


# In[88]:


model_li.score(x_test, y_test)


# # trying all diffrent regression algorithm and finding the testing score

# In[93]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR


# In[95]:


model = [DecisionTreeRegressor, SVR, RandomForestRegressor, KNeighborsRegressor, AdaBoostRegressor]
for mod in model:
    reg = mod()
    reg = reg.fit(x_train, y_train)
    print(mod , 'accuracy', reg.score(x_test, y_test))


# # Now applying Kflod and cross validation technique 

# In[96]:


from sklearn.model_selection import KFold, cross_val_score


# In[99]:


models = []
models.append(('KNN', KNeighborsRegressor()))
models.append(('DecisionTree', DecisionTreeRegressor()))
models.append(('RandomForest', RandomForestRegressor()))
models.append(('SVM', SVR()))
models.append(('AdaBoost', AdaBoostRegressor()))


results = []
names = []
for name, model in models:
    kfold = KFold(n_splits = 10)
    cv_result = cross_val_score(model, x_train, y_train, cv = kfold)
    names.append(name)
    results.append(cv_result)
for i in range(len(names)):
    print(names[i], results[i].mean())


# # Here we see RandomForestRegressor gives us best score so we can use RandomForest Regressor algorithm 

# In[100]:


from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(x_train, y_train)


# In[102]:


y_pred = reg_rf.predict(x_test)


# In[105]:


reg_rf.score(x_train, y_train)


# In[107]:


reg_rf.score(x_test, y_test)


# # performing hyper-parameter tuning using RandomizedSearchCV or GridSearchCV

# In[108]:


from sklearn.model_selection import RandomizedSearchCV


# In[109]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]


# In[110]:


random_grid = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth':max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}


# # Random search of parameters, using 5 fold cross validation and search across 100 different combinations 

# In[111]:


rf_random = RandomizedSearchCV(estimator = reg_rf,
                              param_distributions = random_grid,
                              scoring = 'neg_mean_squared_error',
                              cv = 5,
                              verbose = 2,
                              random_state = 42)


# In[112]:


rf_random.fit(x_train, y_train)


# In[114]:


rf_random.best_params_


# # comparing y_test and y_pred using graph 

# In[115]:


sns.distplot(y_test-y_pred)
plt.show()


# In[116]:


#making scatter plot
plt.scatter(y_test, y_pred, alpha = 0.5)
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()


# # Model Evalution 

# In[117]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[118]:


# checking mean_absolute_error
mean_absolute_error(y_test, y_pred)


# In[119]:


# checking mean_squared_error
mean_squared_error(y_test, y_pred)


# In[120]:


#checking r2_score
r2_score(y_test, y_pred)


# # checking the score after applying all the tools

# In[123]:


rf = RandomForestRegressor(n_estimators= 700, min_samples_split= 15, min_samples_leaf= 1, max_features= 'auto', max_depth= 20)


# In[124]:


rf.fit(x_train, y_train)


# In[125]:


r_pred=rf.predict(x_test)


# In[126]:


r2_score(y_test, r_pred)


# In[ ]:




