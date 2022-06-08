#!/usr/bin/env python
# coding: utf-8

# # Programming Assignment 2: Linear Regression
# 
# ## Instructions
# 
# - The aim of this assignment is to give you a hands-on with a real-life machine learning application.
# - Use separate training, and testing data as discussed in class.
# - You can only use Python programming language and Jupyter Notebooks.
# - There are three parts of this assignment. In parts 1 & 2, you can only use **numpy, scipy, pandas, matplotlib and are not allowed to use NLTK, scikit-learn or any other machine learning toolkit**. However, you have to use **scikit-learn** in part 3.
# - Carefully read the submission instructions, plagiarism and late days policy below.
# - Deadline to submit this assignment is: **Monday, 8th November 2021**.
# 
# ## Submission Instructions
# 
# Submit your code both as notebook file (.ipynb) and python script (.py) on LMS. The name of both files should be your roll number. If you don’t know how to save .ipynb as .py [see this](https://i.stack.imgur.com/L1rQH.png). **Failing to submit any one of them will result in the reduction of marks**.
# 
# ## Plagiarism Policy
# 
# The code MUST be done independently. Any plagiarism or cheating of work from others or the internet will be immediately referred to the DC. If you are confused about what constitutes plagiarism, it is YOUR responsibility to consult with the instructor or the TA in a timely manner. No “after the fact” negotiations will be possible. The only way to guarantee that you do not lose marks is “DO NOT LOOK AT ANYONE ELSE'S CODE NOR DISCUSS IT WITH THEM”.
# 
# ## Late Days Policy
# 
# The deadline of the assignment is final. However, in order to accommodate all the 11th hour issues there is a late submission policy i.e. you can submit your assignment within 3 days after the deadline with 25% deduction each day.
# 
# 
# ## Introduction
# 
# In this exercise, you will implement linear regression and get to see it work on data. After completing this assignment, you will know:
# - How to implement linear regression from scratch.
# - How to estimate linear regression parameters using gradient descent.
# - How to make predictions on new data using learned parameters.
# 
# Let's start with the necessary imports.

# In[283]:


import os
import numpy as np
from matplotlib import pyplot
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Linear Regression with one variable
# 
# Linear regression assumes a linear relationship between the input variables (X) and the single output variable (Y). More specifically, that output (Y) can be calculated from a linear combination of the input variables (X). When there is a single input variable, the method is referred to as a simple linear regression.
# 
# Now you will implement simple linear regression to predict profits for a food truck. Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet. The chain already has trucks in various cities and you have data for profits and populations from the cities. You would like to use this data to help you select which city to expand to next.

# ### 1.1 Load the dataset
# 
# The file `Data/ex1data1.txt` contains the dataset for our linear regression problem. The first column is the population of a city (in 10,000s) and the second column is the profit of a food truck in that city (in $10,000s). A negative value for profit indicates a loss. 
# 
# We provide you with the code needed to load this data. The dataset is loaded from the data file into the variables `X` and `Y`.

# In[284]:


data = np.loadtxt(os.path.join('Data', 'ex1data.txt'), delimiter=',')
X, Y = data[:, 0], data[:, 1]


# ### 1.2 Plot the dataset
# Before starting on any task, it is often useful to understand the data by visualizing it. For this dataset, you can use a scatter plot to visualize the data, since it has only two properties to plot (profit and population). Execute the next cell to visualize the data.

# In[285]:


pyplot.plot(X, Y, 'ro', ms=10, mec='k')
pyplot.ylabel('Profit in $10,000')
pyplot.xlabel('Population of City in 10,000s')


# ### 1.3 Learn the parameters
# In this part, you will fit the linear regression parameters $\theta$ to the food truck dataset using gradient descent.
# 
# The objective of linear regression is to minimize the cost function
# 
# $$ J(\theta) = \frac{1}{2m} \sum_{i=1}^m \left( h_{\theta}(x^{(i)}) - y^{(i)}\right)^2 ------ (i)$$ 
# 
# where the hypothesis $h_\theta(x)$ is given by the linear model
# $$ h_\theta(x) = \theta_0 + \theta_1 x ------ (ii)$$
# 
# The parameters of your model are the $\theta_j$ values. These are
# the values you will adjust to minimize cost $J(\theta)$. One way to do this is to
# use the batch gradient descent algorithm. In batch gradient descent, each
# iteration performs the update
# 
# $$ \theta_0 = \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)}\right) ------ (iii)$$
# 
# $$ \theta_1 = \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)}\right)x^{(i)} ------ (iv)$$
# 
# With each step of gradient descent, your parameters $\theta_j$ come closer to the optimal values that will achieve the lowest cost J($\theta$).
# 
# Let's start by implementing the hypothesis $h_\theta(x)$.

# In[286]:


### GRADED FUNCTION ###
def predict(x, theta0, theta1):
    '''
    Calculates the hypothesis for any input sample `x` given the parameters `theta`.
    
    Arguments
    ---------
    x : float
        The input sample.
    
    theta0 : float
        The parameter for the regression function.
        
    theta1 : float
        The parameter for the regression function.
    
    Returns
    -------
    h_x : float
        The hypothesis for input sample.
    
    Hint(s)
    -------
    Compute equation (ii).
    '''
    # You need to return the following variable(s) correctly
    h_x = 0.0
    
    ### START CODE HERE ### (≈ 1 line of code)
    
    h_x = (theta1*x) + theta0
    
    ### END CODE HERE ###
    
    return h_x


# Execute the next cell to verify your implementation.

# In[287]:


h_x = predict(x=2, theta0=1.0, theta1=1.0)
print('With x = 2, theta0 = 1.0, theta1 = 1.0\nPredicted Hypothesis h(x) = %.2f' % h_x)
print("Expected hypothesis h(x) = 3.00\n")


# As you perform gradient descent to learn minimize the cost function  $J(\theta)$, it is helpful to monitor the convergence by computing the cost. In this section, you will implement a function to calculate  $J(\theta)$ so you can check the convergence of your gradient descent implementation.

# In[288]:


### GRADED FUNCTION ###
def computeCost(X, Y, theta0, theta1):
    '''
    Computes cost for linear regression. Computes the cost of using `theta` as the
    parameter for linear regression to fit the data points in `X` and `Y`.
    
    Arguments
    ---------
    X : array
        The input dataset of shape (m, ), where m is the number of training examples.
    
    Y : array
        The values of the function at each data point. This is a vector of
        shape (m, ), where m is the number of training examples.
    
    theta0 : float
        The parameter for the regression function.
        
    theta1 : float
        The parameter for the regression function.
    
    Returns
    -------
    J : float
        The value of the regression cost function.
    
    Hint(s)
    -------
    Compute equation (i).
    '''
    # initialize some useful values
    m = Y.size  # number of training examples
    
    # You need to return the following variable(s) correctly
    J = 0
        
    ### START CODE HERE ### (≈ 3-4 lines of code)
    
    J = np.square(predict(X, theta0, theta1) - Y)  
    J = np.sum(J)
    J = J/(2*m)
    
    ### END CODE HERE ###
    
    return J


# Execute the next cell to verify your implementation.

# In[289]:


J = computeCost(X, Y, theta0=1.0, theta1=1.0)
print('With theta0 = 1.0, theta1 = 1.0\nPredicted cost J = %.2f' % J)
print("Expected cost J = 10.27\n")


# Next, you will complete a function which implements gradient descent. The loop structure has been written for you, and you only need to supply the updates to parameters $\theta_j$  within each iteration (epoch). 
# 
# The starter code for the function `gradientDescent` calls `computeCost` on every iteration and saves the cost to a `python` list. Assuming you have implemented `gradientDescent` and `computeCost` correctly, your value of $J(\theta)$ should never increase, and should converge to a steady value by the end of the algorithm.

# In[290]:


### GRADED FUNCTION ###
def gradientDescent(X, Y, alpha, n_epoch):
    """
    Performs gradient descent to learn `theta`. Updates `theta` by taking `n_epoch`
    gradient steps with learning rate `alpha`.
    
    Arguments
    ---------
    X : array
        The input dataset of shape (m, ), where m is the number of training examples.
    
    Y : array
        The values of the function at each data point. This is a vector of
        shape (m, ), where m is the number of training examples.
    
    alpha : float
        The learning rate.
    
    n_epoch : int
        The number of iterations for gradient descent. 
    
    Returns
    -------
    theta0 : float
        The parameter for the regression function.
        
    theta1 : float
        The parameter for the regression function.
    
    J : list
        A python list for the values of the cost function after each iteration.
    
    Hint(s)
    -------
    Compute equation (iii) and (iv).

    While debugging, it can be useful to print out the values of 
    the cost function (computeCost) here.
    """
    # initialize some useful values
    m = Y.size  # number of training examples
    J = list()  # list to store cost
    
    # You need to return the following variables correctly
    theta0 = 0.0
    theta1 = 0.0
    
    for epoch in range(n_epoch):
        ### START CODE HERE ### (≈ 5-10 lines of code)
        
        h = predict(X, theta0, theta1)
        
        theta0 = theta0 - ( alpha*(1/m)*np.sum(h- Y) )
        theta1 = theta1 - ( alpha*(1/m)*np.sum( (h - Y)*X ) )
        
        ### END CODE HERE ###

        J.append(computeCost(X, Y, theta0, theta1))
    return theta0, theta1, J


# Execute the next cell to verify your implementation.

# In[291]:


n_epoch = 1500
alpha = 0.01

theta0, theta1, J = gradientDescent(X ,Y, alpha, n_epoch)
print('Predicted theta0 = %.4f, theta1 = %.4f, cost = %.4f' % (theta0, theta1, J[-1]))
print('Expected theta0 = -3.6303, theta1 = 1.1664, cost = 4.4834')


# ### 1.4 Plot the linear fit
# 
# Use your learned parameters $\theta_j$ to plot the linear fit.

# In[292]:


h_x = list()
for x in X:
    h_x.append(predict(x, theta0, theta1))
pyplot.plot(X, Y, 'ro', ms=10, mec='k')
pyplot.ylabel('Profit in $10,000')
pyplot.xlabel('Population of City in 10,000s')
pyplot.plot(X, h_x, '-')
pyplot.legend(['Training data', 'Linear regression'])


# ### 1.5 Make predictions
# 
# Use your learned parameters $\theta_j$ to make food truck profit predictions in areas with population of 40,000 and 65,000.

# In[293]:


print('For population = 40,000, predicted profit = $%.2f' % (predict(4, theta0, theta1)*10000))
print('For population = 65,000, predicted profit = $%.2f' % (predict(6.5, theta0, theta1)*10000))


# ## 2. Multivariate Linear Regression
# 
# Now, you will implement multivariate linear regression (from scratch) to predict the the median price of homes in a Boston suburb during the mid-1970s. To do this, you are given with the dataset that has 404 examples in the train set and 102 examples in test set. Each example has 13 input variables (features) and one output variable (price in $10,000s). Below is the description of input variables:
# 
# - Per capita crime rate.
# - The proportion of residential land zoned for lots over 25,000 square feet.
# - The proportion of non-retail business acres per town.
# - Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
# - Nitric oxides concentration (parts per 10 million).
# - The average number of rooms per dwelling.
# - The proportion of owner-occupied units built before 1940.
# - Weighted distances to five Boston employment centers.
# - Index of accessibility to radial highways.
# - Full-value property-tax rate per $10,000.
# - Pupil-teacher ratio by town.
# - 1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
# - Percentage lower status of the population.
# 
# Each one of these input features is stored using a different scale. Some features are represented by a proportion between 0 and 1, other features are ranges between 1 and 12, some are ranges between 0 and 100, and so on. This is often the case with real-world data, and understanding how to explore and clean such data is an important skill to develop.
# 
# A common way to normalize features that use different scales and ranges is:
# 
# - Subtract the mean value of each feature from the dataset.
# - After subtracting the mean, additionally scale (divide) the feature values by their respective standard deviations.
# 
# Note: We only use examples of the train set to estimate the mean and standard deviation.
# 
# You have to follow exactly the same steps as above i.e. implement hypothesis, cost function and gradient descent for multivariate linear regression to learn parameters $\theta$ using train set. Finally, report the cost (error) using your learned parameters $\theta$ on test set. Expected Mean Square Error on this dataset is 11.5 - 12.5 approximately. 
# 
# We provide you with the code needed to load this dataset. The dataset is loaded from the data files into the variables `train_X`, `train_Y`, `test_X` and `test_Y`.

# In[294]:


train_X = np.loadtxt(os.path.join('Data', 'ex2traindata.txt'))
train_Y = np.loadtxt(os.path.join('Data', 'ex2trainlabels.txt'))
test_X = np.loadtxt(os.path.join('Data', 'ex2testdata.txt'))
test_Y = np.loadtxt(os.path.join('Data', 'ex2testlabels.txt'))


# ### Data Normalization

# In[295]:


means = train_X.mean(axis=0)
stds = train_X.std(axis=0)

m = train_X.shape[0]
t = test_Y.shape[0]

train_X = train_X - means
train_X = train_X/stds
train_X = np.c_[ np.ones(m), train_X ]

test_X = test_X - means
test_X = test_X/stds
test_X = np.c_[ np.ones(t), test_X ]


# ### Hypothesis 

# In[296]:


def hypothesis(thetas, train_X):
    h = train_X*thetas
    h = np.sum(h, axis=1, keepdims=True)
    h = h.flatten('F')
    return h 


# ### Cost Function

# In[297]:


def cost(thetas, train_X, train_Y):
    rows = train_Y.shape[0]
    J = np.square(hypothesis(thetas, train_X) - train_Y)
    J = np.sum(J)
    J = J/(2*rows)
    return J


# ### Gradient Descent

# In[298]:


def gradientDes(train_X, train_Y, alpha, epochs):
    
    J = []
    thetas = np.zeros(train_X.shape[1])
    
    for epoch in range(epochs):
        
        temp = np.zeros(train_X.shape[1])
        
        h = hypothesis(thetas, train_X)
        difference = h-train_Y
        
        for i in range(train_X.shape[1]):
            
            temp[i] = (alpha*(1/m)*np.sum( difference*train_X[:,i] ) )
        
        thetas = thetas - temp
        
        J.append(cost(thetas, train_X, train_Y))
        
    return thetas, J


# ### Learning the parameters

# In[299]:


theta, J = gradientDes(train_X, train_Y, 0.01, 2000)
print("Bias + Thetas = ", theta)
print("Error on Training Data = ", J[-1])


# ### Running on Test Data

# In[300]:


test_error = cost(theta, test_X, test_Y)
print("Error on Test Data = ", test_error)


# ## 3. Regularized Linear Regression
# 
# Now, you'll use the [scikit-learn](https://scikit-learn.org/stable/index.html) to implement [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html), [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge), [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso), [Elastic Net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet) and apply them to Boston house pricing dataset (provided in part 2). Try out different values of regularization coefficient (known as alpha in scikit-learn) and use the [Mean Squared Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) to report loss with each regression. Finally, plot the regularization coefficients alpha (x-axis) with learned parameters $\theta$ (y-axis) for Ridge and Lasso. Please read [this blog](https://scienceloft.com/technical/understanding-lasso-and-ridge-regression/) to get better understanding of the desired plots.

# In[301]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error


# In[302]:


train_X = np.loadtxt(os.path.join('Data', 'ex2traindata.txt'))
train_Y = np.loadtxt(os.path.join('Data', 'ex2trainlabels.txt'))
test_X = np.loadtxt(os.path.join('Data', 'ex2testdata.txt'))
test_Y = np.loadtxt(os.path.join('Data', 'ex2testlabels.txt'))


# ### Linear Regression

# In[303]:


linear_regression = LinearRegression()
linear_regression.fit(train_X, train_Y)
predicted = linear_regression.predict(test_X)
loss = mean_squared_error(test_Y, predicted)/2
print("Thetas = ", linear_regression.coef_)
print("Loss on Linear Regression = ", loss)


# ### Ridge Regression

# In[304]:


ridge = Ridge()
alphas = np.logspace(0, 6, 200)
ridge_coeffs = []
ridge_costs = []

for i in alphas:
    ridge.set_params(alpha=i)
    ridge.fit(train_X, train_Y)
    ridge_coeffs.append(ridge.coef_)
    predicted = ridge.predict(test_X)
    loss = mean_squared_error(test_Y, predicted)
    ridge_costs.append(loss)


# ### Plotting Ridge Coefficients

# In[305]:


# took help from here https://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_path.html

import matplotlib.pyplot as plt
columns = ['Per capita crime rate', 
           'proportion of residential land', 
           'proportion of non-retail business acres per town', 
           'Charles River', 'Nitric oxides concentration', 
           'average number of rooms per dwelling', 
           'proportion of owner-occupied units', 
           'Weighted distances to five Boston employment centers', 
           'Index of accessibility to radial highways', 
           'Full-value property-tax rate', 
           'Pupil-teacher ratio by town', 
           'proportion of Black people by town', 
           'Percentage lower status of the population']

plt.figure(figsize=(20,10))
ax = plt.gca()
ax.plot(alphas, ridge_coeffs)
ax.set_xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Coefficients")
plt.title("Ridge Regression Coefficients Vs Alphas")
plt.axis("tight")
plt.legend(columns, loc=4)
plt.tight_layout()
plt.show()


# ### Minimum Cost on Ridge

# In[306]:


min_index = ridge_costs.index(min(ridge_costs))
print("Minimum Cost on Ridge =", min(ridge_costs)/2, "at alpha =", alphas[min_index])


# ### Lasso Rigression

# In[307]:


lasso = Lasso()
alphas = np.logspace(-2, 2, 200)
lasso_coeffs = []
lasso_costs = []

for i in alphas:
    lasso.set_params(alpha=i)
    lasso.fit(train_X, train_Y)
    lasso_coeffs.append(lasso.coef_)
    predicted = lasso.predict(test_X)
    loss = mean_squared_error(test_Y, predicted)
    lasso_costs.append(loss)


# ### Plotting Lasso Coefficient

# In[308]:


plt.figure(figsize=(20,10))
ax = plt.gca()
ax.plot(alphas, lasso_coeffs)
ax.set_xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Coefficients")
plt.title("Lasso Regression Coefficients Vs Alphas")
plt.axis("tight")
plt.legend(columns, loc=4)
plt.tight_layout()
plt.show()


# ### Minimum Cost on Lasso

# In[309]:


min_index = lasso_costs.index(min(lasso_costs))
print("Minimum Cost on Lasso =", min(lasso_costs)/2, "at alpha =", alphas[min_index])


# ### Elastic Net

# In[310]:


elastic = ElasticNet()
alphas = np.logspace(-3, 2, 200)
elastic_coeffs = []
elastic_costs = []

for i in alphas:
    elastic.set_params(alpha=i)
    elastic.fit(train_X, train_Y)
    elastic_coeffs.append(elastic.coef_)
    predicted = elastic.predict(test_X)
    loss = mean_squared_error(test_Y, predicted)
    elastic_costs.append(loss)


# ### Minimum Cost on Elastic Net

# In[311]:


min_index = elastic_costs.index(min(elastic_costs))
print("Minimum Cost on Elastic =", min(elastic_costs)/2, "at alpha =", alphas[min_index])

