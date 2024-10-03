# EX2 Implementation of Simple Linear Regression Model for Predicting the Marks Scored
## AIM:
To implement simple linear regression using sklearn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y by reading the dataset.
2. Split the data into training and test data.
3. Import the linear regression and fit the model with the training data.
4. Perform the prediction on the test data.
5. Display the slop and intercept values.
6. Plot the regression line using scatterplot.
7. Calculate the MSE.

## Program:
```
/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: NITHIN BILGATES C
RegisterNumber: 2305001022  
*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/ex1.csv')
df.head()
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
m=lr.coef_
m
b=lr.intercept_
b
pred=lr.predict(X_test)
pred
X_test
Y_test
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_test, pred)
print(f"Mean Squared Error (MSE): {mse}")
## Output:
![image](https://github.com/user-attachments/assets/337a98ec-20e9-46a3-af22-fa3d61f3f3cd)
![image](https://github.com/user-attachments/assets/1f34e883-862e-41e4-b71e-3dcb0561647a)
![image](https://github.com/user-attachments/assets/7d3eaf56-2bc3-4d17-83be-365170986703)
![image](https://github.com/user-attachments/assets/eac01545-5dfe-46bc-9e7f-7594955a3f19)
![image](https://github.com/user-attachments/assets/f02f73e8-b1d6-40d4-a73c-e4f804fc168c)
![image](https://github.com/user-attachments/assets/f0879ca4-ac4b-4fd5-80ab-a6f17be968fd)
![image](https://github.com/user-attachments/assets/fdb8f5d8-bb93-4d1d-b9ef-eee720cc61ac)
![image](https://github.com/user-attachments/assets/46056b34-2a54-459f-b124-f5939ebb9d51)


## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
