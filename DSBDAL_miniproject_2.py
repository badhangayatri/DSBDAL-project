#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score


# In[2]:


df = pd.read_csv(r"C:\Users\91766\Downloads\mipro.csv")


# In[3]:


print(df)


# In[4]:


df.fillna(0, inplace=True)#replace missing values by 0


# In[5]:


df


# In[11]:


# Separating features (X) and target variable (y)
x= df[['fsgpa', 'ssgpa', 'tsgpa', 'avgsem','Unnamed: 8']]
y = df['lastsem']


# In[12]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[13]:


print(x) 


# In[14]:


print(y)


# In[15]:


# Print the shapes of the training and testing sets
print("Training set shape (x, y):", x_train.shape, y_train.shape)
print("Testing set shape (x, y):", x_test.shape, y_test.shape)


# In[16]:


# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = model.predict(x_test)


# In[17]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)


# In[18]:


r2 = r2_score(y_test, y_pred)


# In[19]:


print("Mean Squared Error:", mse)


# In[21]:


print("R-squared:", r2)


# In[22]:


# Assuming 'new_data' is a DataFrame containing SGPA values for a new student
new_data = pd.DataFrame({
    'fsgpa': [8.0],
    'ssgpa': [7.5],
    'tsgpa': [8.0],
    'avgsem': [7.83],
    'gender_Male':[0]  # Calculated average SGPA for the new student
})

# Use the trained model to make predictions
predicted_marks = model.predict(new_data)

# Print the predicted marks
print("Predicted Marks for the new student:", predicted_marks[0])


# In[23]:


# Assuming 'new_data' is a DataFrame containing SGPA values for a new student
new_data = pd.DataFrame({
    'fsgpa': [7.0],
    'ssgpa': [9.5],
    'tsgpa': [8.4],
    'avgsem': [8.3],
    'gender_Male':[0]  # Calculated average SGPA for the new student
})

# Use the trained model to make predictions
predicted_marks = model.predict(new_data)

# Print the predicted marks
print("Predicted Marks for the new student:", predicted_marks[0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[24]:


import matplotlib.pyplot as plt

# Scatter plot of actual vs. predicted marks
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs. Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red', label='Perfect Prediction')
plt.xlabel('Actual Marks')
plt.ylabel('Predicted Marks')
plt.title('Actual vs. Predicted Marks')
plt.legend()
plt.show()


# In[25]:


import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
np.random.seed(0)
gender = np.random.choice(['Male', 'Female'], size=100)
actual_marks = np.random.randint(50, 100, size=100)
predicted_marks = actual_marks + np.random.normal(0, 5, size=100)

# Scatter plot with color
plt.figure(figsize=(8, 6))
colors = {'Male': 'blue', 'Female': 'red'}
plt.scatter(actual_marks, predicted_marks, c=[colors[g] for g in gender])
plt.xlabel('Actual Marks')
plt.ylabel('Predicted Marks')
plt.title('Actual vs. Predicted Marks (Color by Gender)')
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Male'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Female')],
           title='Gender')
plt.show()


# In[ ]:




