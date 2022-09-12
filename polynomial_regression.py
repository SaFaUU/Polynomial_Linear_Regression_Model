#Data Preprocessing
#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv("Position_Salaries.csv")

#Independent Variable Matrix/ Vector
X = dataset.iloc[:,1:2].values

#Making Dependent Variable Matrix/ Vector
y= dataset.iloc[:, 2].values


#Fitting Lineaer Regression to the dataset
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X,y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatur es(degree=4)
X_poly = polynomial_regressor.fit_transform(X)
 
polynomial_linear_regressor = LinearRegression()
polynomial_linear_regressor.fit(X_poly, y)

#Visualising the Linear Regression Results
plt.scatter(X,y, color='red')
plt.plot(X, linear_regressor.predict(X), color='blue')
plt.title("Salary vs Levels (Linear Regression)")
plt.xlabel("Levels")
plt.ylabel("Salary")
plt.show()


#Visualising the Polynomial Linear Regression Results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y, color='red')
plt.plot(X_grid, polynomial_linear_regressor.predict(polynomial_regressor.fit_transform(X_grid)), color='purple')
plt.title("Salary vs Levels (Polynomial Regression)")
plt.xlabel("Levels")
plt.ylabel("Salary")
plt.show()

#Predicting Single Value/ new result with linear regression
linear_regressor.predict(np.array(6.5).reshape(1,-1))
#Predicting Single Value/ new result with linear regression
polynomial_linear_regressor.predict(polynomial_regressor.fit_transform(np.array(6.5).reshape(1,-1)))
