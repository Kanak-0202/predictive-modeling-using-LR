#!/usr/bin/env python
# coding: utf-8

# In[18]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# In[19]:


# Load the dataset
data = pd.read_csv(r'C:\Users\user\Desktop\Datasets\advertising.csv')


# In[20]:


data.head()


# In[22]:


data.info()


# In[24]:


data.corr()


# In[9]:


# Select features (X) and target variable (y)
X = data[['TV(thousands)', 'Radio(thousands)', 'Newspaper(thousands)']]  # Features
y = data['Sales(millions)']  # Target variable


# In[10]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[12]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[13]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[15]:


print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# In[16]:


# Visualize the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Linear Regression Predictions on Advertising Dataset')
plt.show()


# In[ ]:




