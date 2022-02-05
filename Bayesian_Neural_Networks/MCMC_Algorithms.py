import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from Helper_Functions import *
from Modelling import *

#The Markov Chain Monte Carlo based Metropolis Hastings Algorithm for a Neural Network
def NN_MCMC_MH(epochs, model, inp_data, out_data):
    accepted_samples = []
    rejected_samples = []
    
    #Initial Parameters drawing randomly
    parameters = init_parameters(model[0], model[1], model[2])
    
    #The standard deviation for calculating the log likelihood
    sigma = 3
    
    #The prior for the parameters , taken 0 as it wouldn't matter much
    prior_pt = 0
    
    #Calculating the log likelihood of the parameters at time t given the data
    ll_ft = 0
    for i in range(len(inp_data)):
        ll_ft += LL_Calc(NN_Compute(inp_data[i], parameters), out_data[i], sigma)
    #The final function of parameters at time t i.e, the numerator according to the Metropolis Hastings algorithm, proportional to the probability P(D|theta)*P(theta) 
    ft = ll_ft + prior_pt

    for _ in tqdm(range(epochs)):
        
        #Prior for parameters at time t+1 , again taken 0 as it wouldnt matter much
        ptnext = 0
        
        #Standard deviation (scale) takes to draw new parameters
        scale = 0.1
        
        #New parameters drawn at time t+1 from original parameters at time t and normal distribution with standard deviation scale
        new_parameters = new_params(parameters, scale)
        
        #The final function of parameters at time t i.e, the numerator according to the Metropolis Hastings algorithm, proportional to the probability P(D|theta)*P(theta)
        ll_ftnext = 0
        for i in range(len(inp_data)):
            ll_ftnext += LL_Calc(NN_Compute(inp_data[i], new_parameters), out_data[i], sigma)

        ftnext = ll_ftnext + ptnext
        
        #Acceptance criteria taken as 0.6
        accepted = np.exp((ftnext - ft)) > 0.6

        if accepted:
            
            #Accepted parameters appended 
            accepted_samples.append(new_parameters)
            
            #The numerator f(t) updated
            ft = ftnext
            
            #Parameters updated as they are accepted
            parameters = new_parameters
            
            #Standard deviations updated for drawing samples and for calculating likelihood
            if sigma > 0.01:
                sigma -= 1 / epochs

            if scale > 0.01:
                scale -= scale * 1 / 100

        else:
            #Accepted parameters appended 
            rejected_samples.append(new_parameters)
    #Returns the accepted and rejectedsamples as a set
    return (accepted_samples, rejected_samples)
