# Majority of code is slightly modified from 
# https://github.com/suragnair/alpha-zero-general

import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
from NeuralNet import NeuralNet
import os


class SNAPT_AC(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=1e-4):
        super(SNAPT_AC, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
        
    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))

        value = F.relu(self.critic_linear1(state))
        value = torch.tanh(self.critic_linear2(value))

        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.log_softmax(self.actor_linear2(policy_dist), dim=-1)

        return policy_dist, value
    
    
     
args = {
    'lr': 0.001,
    'cuda': torch.cuda.is_available(),
}

class NNetWrapper(NeuralNet):
    def __init__(self, g):
        self.g = g
        board, _ = g.getInitBoard()
        attacker_inputs = len(g.get_attack_vector(board))
        defender_inputs = len(g.get_defend_vector(board))
        self.att_nnet = SNAPT_AC(attacker_inputs, g.size, 64)
        self.def_nnet = SNAPT_AC(defender_inputs, 2 * g.size, 64)




    def predict(self, canonicalBoard):
        """
        board: np array with board
        """
        # timing
        board, player = canonicalBoard
        
        if player == 1:
            nnet = self.att_nnet
            board = self.g.get_attack_vector(board)
        elif player == -1:
            nnet = self.def_nnet
            board = self.g.get_defend_vector(board)
        else:
            print('burh')
        

        nnet.eval()
        with torch.no_grad():
            pi, v = nnet.forward(board)
          
        probs = pi.squeeze().detach().numpy()
        
        if player == 1:
            probs = np.concatenate((probs, np.zeros(2 * self.g.size)))
        else:
            probs = np.concatenate((np.zeros(self.g.size), probs))
        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return probs, v.squeeze().detach().numpy()
    
    
    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]
    
    def log_loss_pi(self, targets, outputs):
        return -torch.sum(targets * torch.log(outputs)) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', att_filename ='att_checkpoint.pth.tar', def_filename = 'def_checkpoint.pth.tar'):
        att_filepath = os.path.join(folder, att_filename)
        def_filepath = os.path.join(folder, def_filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.att_nnet.state_dict(),
        }, att_filepath)
        torch.save({
            'state_dict': self.def_nnet.state_dict(),
        }, def_filepath)

    def load_checkpoint(self, folder='checkpoint', att_filename ='att_checkpoint.pth.tar', def_filename = 'def_checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        att_filepath = os.path.join(folder, att_filename)
        def_filepath = os.path.join(folder, def_filename)
        if not os.path.exists(att_filepath):
            raise ("No model in path {}".format(att_filepath))
        if not os.path.exists(def_filepath):
            raise ("No model in path {}".format(def_filepath))
        map_location = None if args[cuda] else 'cpu'
        att_checkpoint = torch.load(att_filepath, map_location=map_location)
        self.att_nnet.load_state_dict(att_checkpoint['state_dict'])
        
        def_checkpoint = torch.load(def_filepath, map_location=map_location)
        self.def_nnet.load_state_dict(def_checkpoint['state_dict'])