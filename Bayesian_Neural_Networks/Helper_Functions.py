import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

#Finds the mean of data X

def mean(X):
  return float(sum(X))/float(len(X))

#Finds the standard deviation of data X
def std_dev(X):
  mu = mean(X)
  return (sum([(float(x)-mu)**2 for x in X])/(float(len(X))-1))**(1/2)

#Finds the value of x over the normal distribution with mean mu and standard deviation sigma
def normal(x, mu, sigma):
  num = np.exp((-1/2)*((x-mu)/sigma)**2)
  denom = sigma*math.sqrt(2*math.pi)
  return num/denom

#The sigmoid activation function
def sigmoid(x):
  return 1/(1+np.exp(-x))

#A binary step activation function
def binary_step(x, inflection_point):
  ret = []
  for i in x:
    if i>inflection_point:
      ret.append(1)
    else:
      ret.append(0)

  return np.array(ret)

#Returns the argmax of a vector in the form of a vector
def argmax_array(x):
  i = np.argmax(x)
  rets = np.zeros_like(x)
  rets[i]=1
  return rets
#Afunction to retrieve data from the given path and split it into the input and output data
def data_retriever(path):
    df = pd.read_csv(str(path))
    df_np = np.array(df[:][:])

    X_data = np.array(df_np[:, 1:3])

    Y_data = []

    for i in range(len(X_data)):
        if df_np[i, 3] == 0:
            Y_data.append([1.0, 0.0])
        if df_np[i, 3] == 1:
            Y_data.append([0.0, 1.0])

    Y_data = np.array(Y_data)

    return (X_data, Y_data)

