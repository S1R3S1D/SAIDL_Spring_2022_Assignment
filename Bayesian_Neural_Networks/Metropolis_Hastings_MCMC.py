import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from Helper_Functions import *
from Modelling import *


def NN_MCMC_MH(epochs, model, inp_data, out_data):
    accepted_samples = []
    rejected_samples = []

    parameters = init_parameters(model[0], model[1], model[2])

    sigma = 3

    prior_pt = 0

    ll_ft = 0
    for i in range(len(inp_data)):
        ll_ft += LL_Calc(NN_Compute(inp_data[i], parameters), out_data[i], sigma)

    ft = ll_ft + prior_pt

    for _ in tqdm(range(epochs)):
        ptnext = 0
        scale = 0.1
        new_parameters = new_params(parameters, scale)
        ll_ftnext = 0
        for i in range(len(inp_data)):
            ll_ftnext += LL_Calc(NN_Compute(inp_data[i], new_parameters), out_data[i], sigma)

        ftnext = ll_ftnext + ptnext

        accepted = np.exp((ftnext - ft)) > 0.6

        if accepted:

            accepted_samples.append(new_parameters)

            ft = ftnext

            parameters = new_parameters

            if sigma > 0.01:
                sigma -= 1 / epochs

            if scale > 0.01:
                scale -= scale * 1 / 100

        else:
            rejected_samples.append(new_parameters)

    return (accepted_samples, rejected_samples)
