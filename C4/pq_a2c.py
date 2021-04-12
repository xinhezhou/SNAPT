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

class RandomAttackAgent():
    def __init__(self, g):
        self.g = g
        
    def forward(self, board):
        return [1] * self.g.size, 0
    
class RandomDefenseAgent():
    def __init__(self, g):
        self.g = g
        
    def forward(self, board):
        return [1] * (self.g.size*2), 0

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
wins = 0
games = 100
for j in range(games): 
    winner = play_game(RandomAttackAgent(pq), RandomDefenseAgent(pq))[0]
    if winner == 1:
        wins += 1

print(wins/games)

attacker_inputs = len(pq.get_attack_vector(board))
defender_inputs = len(pq.get_defend_vector(board))



def cem_pq(g, iters = 100, batch_size = 20, elite_size = 10, episode_count = 10, weight_type = 'even', temp = 1, t_max = 3600):
    att_mu = PQAC(attacker_inputs, pq.size, 64)
    def_mu = PQAC(defender_inputs, 2 * pq.size, 64)
    
    att_sigma = copy.deepcopy(att_mu)
    def_sigma = copy.deepcopy(def_mu)
    
    att_rewards = []
    def_rewards = []
    
    with torch.no_grad():
        for param in att_sigma.parameters():
            param.divide_(10)
        for param in def_sigma.parameters():
            param.divide_(10)
    start = time.time()
    for i in range(iters):
            
        att_models = [add_noise(att_mu, std = att_sigma) for k in range(batch_size)]
        def_models = [add_noise(def_mu, std = def_sigma) for k in range(batch_size)]
        
        att_values = [attacker_vs_defender(att_model, def_mu, episode_count, temp = temp) for att_model in att_models]
        def_values = [attacker_vs_defender(att_mu, def_model, episode_count, temp = temp) for def_model in def_models]
        
        att_dict = {(att_values[k] + (k/(10**5))): att_models[k] for k in range(batch_size)}
        def_dict = {(def_values[k] + (k/(10**5))): def_models[k] for k in range(batch_size)}
        
        att_keys = list(reversed(sorted(att_dict)))
        def_keys = list(sorted(def_dict))
        
        att_elites = [att_dict[key] for key in att_keys[:elite_size]]
        def_elites = [def_dict[key] for key in def_keys[:elite_size]]

        if weight_type == 'even':
            weights = [1/elite_size for k in range(elite_size)]
        elif weight_type == 'log':
            weights = log_weights(elite_size)
        else:
            print('bruh')
            return None
        
        att_sigma = weighted_std(att_elites, weights, att_mu, noise = 0.000001)
        att_mu = weighted_sum(att_elites, weights)
        
        def_sigma = weighted_std(def_elites, weights, def_mu, noise = 0.000001)
        def_mu = weighted_sum(def_elites, weights)
        
        att_rewards.append(attacker_vs_random(att_mu, episode_count, temp = temp))
        def_rewards.append(defender_vs_random(def_mu, episode_count, temp = temp))
        
        gc.collect()
        current = time.time()
        if current -start > t_max:
            print('Finished in {} seconds and {} iterations'.format(np.round(current-start), i+1))
            break
        if (i + 1)  % (iters// 20) == 0:
            
            print('Finished iteration {}. {} seconds elapsed.'.format(i+1, np.round(current-start, 2)))
            print(att_rewards[-1], def_rewards[-1])
            
    return att_mu, att_rewards, def_mu, def_rewards


def oneone_pq(g, iters = 100, episode_count = 10, temp = 1, t_max = 3600):
    att_mu = PQAC(attacker_inputs, pq.size, 64)
    def_mu = PQAC(defender_inputs, 2 * pq.size, 64)
    
    att_sigma = copy.deepcopy(att_mu)
    def_sigma = copy.deepcopy(def_mu)
    
    att_rewards = []
    def_rewards = []
    
    with torch.no_grad():
        for param in att_sigma.parameters():
            param.divide_(10)
        for param in def_sigma.parameters():
            param.divide_(10)
    start = time.time()
    
    for i in range(iters):
            
        att_model = add_noise(att_mu, std = att_sigma)
        def_model = add_noise(def_mu, std = def_sigma)
        
        default = attacker_vs_defender(att_mu, def_mu, episode_count, temp = temp)
        att_value = attacker_vs_defender(att_model, def_mu, episode_count, temp = temp)
        def_value = attacker_vs_defender(att_mu, def_model, episode_count, temp = temp)
        
        if att_value > default:
            att_sigma = weighted_std([att_model], [1], att_mu, noise = 0.000001)
            att_mu = att_model
        
        if def_value < default:
            def_sigma = weighted_std([def_model], [1], def_mu, noise = 0.000001)
            def_mu = def_model
        
        att_rewards.append(attacker_vs_random(att_mu, episode_count, temp = temp))
        def_rewards.append(defender_vs_random(def_mu, episode_count, temp = temp))
        
        gc.collect()
        current = time.time()
        if current -start > t_max:
            print('Finished in {} seconds and {} iterations'.format(np.round(current-start), i+1))
            break
        if (i + 1)  % (iters// 20) == 0:
            print('Finished iteration {}. {} seconds elapsed.'.format(i+1, np.round(current-start, 2)))
            print(att_rewards[-1], def_rewards[-1])
            
    return att_mu, att_rewards, def_mu, def_rewards



att_mu, att_rewards, def_mu, def_rewards = oneone_pq(pq, iters = 10000, episode_count = 50, temp = 1, t_max = 3600)


import os
att_fname= os.path.join('checkpoint', 'oneone_att_pq_3600.pth.tar')
torch.save(att_mu.state_dict(), att_fname)

def_fname= os.path.join('checkpoint', 'oneone_att_pq_3600.pth.tar')
torch.save(def_mu.state_dict(), def_fname)