import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from Helper_Functions import *


def init_parameters(I, H, O):
    weights = []
    biases = []
    l = I
    for i in H:
        w = np.random.random(size=(i, l))
        b = np.random.random(size=(i,))

        weights.append(w)
        biases.append(b)

        l = i

    w = np.random.random(size=(O, l))
    b = np.random.random(size=(O,))

    weights.append(w)
    biases.append(b)

    return (weights, biases)


def NN_Compute(input, parameters):
    (weights, biases) = parameters
    output = None
    for i in range(len(weights)):
        output = np.dot(input, np.transpose(weights[i]))
        if i < len(weights) - 1:
            output = sigmoid(output)
        input = output

    # output = binary_step(output, 0.5)

    return output


def new_params(parameters, scale):
    (w_o, b_o) = parameters

    new_w = []
    new_b = []

    for i in range(len(w_o)):
        w_flat = w_o[i].flatten()
        b_flat = b_o[i].flatten()

        w = np.array([np.random.normal(we, scale) for we in w_flat]).reshape(w_o[i].shape)
        b = np.array([np.random.normal(ba, scale) for ba in b_flat]).reshape(b_o[i].shape)

        new_w.append(w)
        new_b.append(b)

    new_parameters = (new_w, new_b)

    return new_parameters

def LL_Calc(X, Y, sigma):
  ll = sum(np.log([normal(x, y, sigma) for x, y in zip(X, Y)]))
  return ll

