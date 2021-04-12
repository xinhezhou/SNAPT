import copy
import torch
import numpy as np
import torch.nn as nn


        

def add_noise(model, std = 0, division = 1):
    new_model = copy.deepcopy(model)
    with torch.no_grad():
        if std == 0:
            for param in new_model.parameters():
                noise = torch.randn(param.size()) * 0.1
                param.add_(noise)
        else:
            for param, std_param in zip(new_model.parameters(), std.parameters()):
                param.add_(torch.randn(param.size()) * std_param / division)
                
    return new_model


def get_first_param(nnet):
    for param in nnet.parameters():
        p = copy.deepcopy(param)
        while len(p.size()) > 0:
            p = p[0]
        return p.item()
    

def weighted_sum(models, weights):
    total = copy.deepcopy(models[0])
    with torch.no_grad():
        for param in total.parameters():
            param.multiply_(weights[0])

        for model, weight in zip(models[1:], weights[1:]):
            for param, total_param in zip(model.parameters(), total.parameters()):
                total_param.add_(param*weight)
    
    return total

def weighted_std(models, weights, mean, noise = 0):
    total = copy.deepcopy(models[0])
    with torch.no_grad():
        for param in total.parameters():
            param.multiply_(0)
            
        for model, weight in zip(models, weights):
            for param, mean_param, total_param in zip(model.parameters(), mean.parameters(), total.parameters()):
                diff = param - mean_param
                diff.multiply_(diff)
                total_param.add_(diff*weight)
                total_param.add_(noise)
        
        for param in total.parameters():
            param.sqrt_()
    return total


def log_weights(elite_size):
    weights = [np.log(elite_size + 1)/ (k+1) for k in range(elite_size)]
    return weights/sum(weights)