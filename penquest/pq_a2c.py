import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
from pq_net import PQAC
from penquest import *
sys.path.append('../')
from es_utils import *
import time
import numpy as np
import gc
import torch.optim as optim
import torch.nn.functional as F
import random

class RandomAttackAgent():
    def __init__(self, g):
        self.g = g
        
    def forward(self, board):
        return torch.FloatTensor([1] * self.g.size), 0
    
class RandomDefenseAgent():
    def __init__(self, g):
        self.g = g
        
    def forward(self, board):
        return torch.FloatTensor([1] * (self.g.size*2)), 0

def play_game(agent_1, agent_2, render = False, temp = 1):
    board, player = pq.getInitBoard()
    boards = []
    players = []
    pis = []
    values = []
    while pq.getGameEnded(board, player) == 0:
        valids = pq.getValidMoves(board, player)
        if player == 1:
            vec = pq.get_attack_vector(board)
            probs, value = agent_1.forward(vec)
            valids = valids[:pq.size]
        elif player == -1:
            vec = pq.get_defend_vector(board)
            probs, value = agent_2.forward(vec)
            valids = valids[pq.size:]
        else:
            print('bruh')
            
        probs = probs.detach().numpy()

        
        boards.append(board)
        players.append(player)
        pis.append(probs)
        values.append(value)
        
        if temp != 0:
            probs = np.power(probs, temp)
        probs = np.array(probs) * np.array(valids)
        probs = np.squeeze(probs)
        if sum(probs) == 0:
            print('bruh')
        probs = probs / np.sum(probs)
        if temp == 0:
            action = np.argmax(probs)
        else:
            action = np.random.choice(len(probs), p = probs)
        if player == -1:
            action += pq.size
        
        board, player = pq.getNextState(board, player, action, render = render)
        if render:
            pq.render(board, player)
          
    
    return pq.getGameEnded(board, player), [(b, pl, pi, v, pl * pq.getGameEnded(board, player)) for (b, pl, pi, v) in zip(boards, players, pis, values)]
        

    
def attacker_vs_random(attacker, episode_count, temp = 1):
    wins = 0
    for k in range(episode_count):
        winner = play_game(attacker, RandomDefenseAgent(pq), temp = temp)[0]
        if winner == 1:
            wins += 1
            
    return wins/episode_count

def defender_vs_random(defender, episode_count, temp = 1):
    wins = 0
    for k in range(episode_count):
        winner = play_game(RandomAttackAgent(pq), defender, temp = temp)[0]
        if winner == -1:
            wins += 1
            
    return wins/episode_count

def attacker_vs_defender(attacker, defender, episode_count, temp = 1):
    wins = 0
    for k in range(episode_count):
        winner = play_game(attacker, defender, temp = temp)[0]
        if winner == 1:
            wins += 1
            
    return wins
    
    

weights = [[1, 1, 0],
         [1, 1, 1],
         [0, 1, 1]]

m_atts = [[1, 0, 0.5, 0, 0],
          [0, 0, 0.5, 0, 0],
          [0, 1, 0.5, 0, 0]]

p1_atts = [1, 1, 1, 20, 1]
p2_atts = [1, 1, 1, 20, 1]

pq = PenQuest(weights, m_atts, p1_atts, p2_atts)
board, player = pq.getInitBoard()

attacker_inputs = len(pq.get_attack_vector(board))
defender_inputs = len(pq.get_defend_vector(board))

def a2c_pq(g, iters = 100, t_max = 3600, temp = 1, seed = 0):
    torch.manual_seed(seed)
    np.random.seed(0)
    random.seed(0)
    start =time.time()
    learning_rate = 0.00001
    att_ac = PQAC(attacker_inputs, pq.size, 64)
    def_ac = PQAC(defender_inputs, 2 * pq.size, 64)
    att_optimizer = optim.Adam(att_ac.parameters(), lr=learning_rate)
    def_optimizer = optim.Adam(def_ac.parameters(), lr=learning_rate)
    att_ac.eval()
    def_ac.eval()
    entropy_term = 0
    
    for i in range(iters):
        gc.collect()
        att_log_probs = []
        att_values = []
        
        def_log_probs = []
        def_values = []
        
        players = []

        board, player = g.getInitBoard()
        player = 1
        attack_vec = g.get_attack_vector(board)
        #print(att_ac.forward(attack_vec))
        while True:
            valids = pq.getValidMoves(board, player)
            if player == 1:
                vec = pq.get_attack_vector(board)
                probs, value = att_ac(vec)
                valids = valids[:pq.size]
                att_log_prob = torch.log(probs.squeeze())
                att_log_probs.append(att_log_prob)
                att_values.append(value.squeeze())
                
            elif player == -1:
                vec = pq.get_defend_vector(board)
                probs, value = def_ac.forward(vec)
                valids = valids[pq.size:]
                def_log_prob = torch.log(probs.squeeze())
                def_log_probs.append(def_log_prob)
                def_values.append(value)
                
            probs = np.squeeze(probs.detach().numpy())
            entropy = -np.sum(np.mean(probs) * np.log(probs))
            entropy_term += entropy
            
            probs = np.array(probs) * np.array(valids)
            
            
            if temp == 0:
                action = np.argmax(probs)
                
            else:
                probs = np.power(probs, temp)
                probs = probs/np.sum(probs)
                action = np.random.choice(len(probs), p = probs)
            
            
            board, player = g.getNextState(board, player, action)
            
            r = g.getGameEnded(board, player)
            
            if r != 0:
                break
        
        # compute Q values
        att_Qvals = np.zeros_like(att_values)
        att_Qvals += r
        
        def_Qvals = np.zeros_like(def_values)
        def_Qvals -= r
            
        
        #update actor critic
        att_ac.train()
        att_values = torch.stack(att_values)
        att_Qvals = att_Qvals.astype(np.float32)
        att_Qvals = torch.FloatTensor(att_Qvals)
        att_log_probs = torch.stack(att_log_probs)
        
        att_advantage = att_Qvals - att_values
        att_advantage = att_advantage.unsqueeze(-1)
        att_actor_loss = (-att_log_probs * att_advantage).mean()
        att_critic_loss = 0.5 * att_advantage.pow(2).mean()
        att_loss = att_actor_loss + att_critic_loss + 0.0001 * entropy_term
        #print(att_loss)
        #print([((a.grad!=None), a.requires_grad) for a in list(att_ac.parameters())])
        att_loss.backward()
        #print([((a.grad!=None), a.requires_grad) for a in list(att_ac.parameters())])
        att_optimizer.step()
        att_optimizer.zero_grad()
        #print([((a.grad!=None), a.requires_grad) for a in list(att_ac.parameters())])
        att_ac.eval()
        
        #update actor critic
        
        def_ac.train()
        def_values = torch.stack(def_values)
        def_Qvals = def_Qvals.astype(np.float32)
        def_Qvals = torch.FloatTensor(def_Qvals)
        def_log_probs = torch.stack(def_log_probs)
        
        def_advantage = def_Qvals - def_values
        def_advantage = def_advantage.unsqueeze(-1)
        def_actor_loss = (-def_log_probs * def_advantage).mean()
        def_critic_loss = 0.5 * def_advantage.pow(2).mean()
        def_loss = def_actor_loss + def_critic_loss + 0.0001 * entropy_term
        
        
        def_optimizer.zero_grad()
        def_loss.backward()
        def_optimizer.step()
        def_ac.eval()
        
        
        current = time.time()
        if current -start > t_max:
            print('Finished in {} seconds and {} iterations'.format(np.round(current-start), i+1))
            break
        
        if (i + 1)  % max(1,(iters// 20)) == 0:
            
            print('Finished iteration {}. {} seconds elapsed.'.format(i+1, np.round(current-start, 2)))
        
        
    return att_ac, def_ac


att_ac, def_ac = a2c_pq(pq, iters = 20000, t_max = 3600, seed = 2)

import os
att_fname= os.path.join('checkpoint', 'a2c_att_pq_3600.pth.tar')
torch.save(att_ac.state_dict(), att_fname)

def_fname= os.path.join('checkpoint', 'a2c_def_pq_3600.pth.tar')
torch.save(def_ac.state_dict(), def_fname)