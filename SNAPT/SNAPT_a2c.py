# Some of a2c algorithm is based off of 
# https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

import numpy as np
import matplotlib.pyplot as plt
from SNAPT_net import SNAPT_AC
import time
import numpy as np
import gc
import torch
import torch.optim as optim
import random

def play_game(g, agent_1, agent_2, render = False, temp = 1):
    board, player = g.getInitBoard()

    while g.getGameEnded(board, player) == 0:
        valids = g.getValidMoves(board, player)
        if player == 1:
            vec = g.get_attack_vector(board)
            probs, value = agent_1.forward(vec)
            valids = valids[:g.size]
        elif player == -1:
            vec = g.get_defend_vector(board)
            probs, value = agent_2.forward(vec)
            valids = valids[g.size:]
        
        if torch.is_tensor(probs):
            probs = probs.detach().numpy()
            
        if torch.is_tensor(value):
            value = value.detach().numpy()
            
        
        if temp != 0:
            probs = np.power(probs, temp)
        probs = np.array(probs) * np.array(valids)
        probs = np.squeeze(probs)
        if sum(probs) == 0:
            print('Probabilities are all zero! That\'s not good!')
        probs = probs / np.sum(probs)
        if temp == 0:
            action = np.argmax(probs)
        else:
            action = np.random.choice(len(probs), p = probs)
        if player == -1:
            action += g.size
        
        board, player = g.getNextState(board, player, action, render = render)
        if render:
            g.render(board, player)
          
    
    return g.getGameEnded(board, player)
        


def attacker_vs_defender(g, attacker, defender, episode_count, temp = 1):
    wins = 0
    for k in range(episode_count):
        winner = play_game(g, attacker, defender, temp = temp)
        if winner == 1:
            wins += 1
            
    return wins
    
    
    


def a2c(g, iters = 100, t_max = 3600, temp = 1):
    start =time.time()
    
    board, _ = g.getInitBoard()
    attacker_inputs = len(g.get_attack_vector(board))
    defender_inputs = len(g.get_defend_vector(board))


    learning_rate = 0.001
    att_ac = SNAPT_AC(attacker_inputs, g.size, 64)
    def_ac = SNAPT_AC(defender_inputs, 2 * g.size, 64)
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

        while True:
            valids = g.getValidMoves(board, player)
            if player == 1:
                vec = g.get_attack_vector(board)
                probs, value = att_ac(vec)
                valids = valids[:g.size]
                att_log_prob = torch.log(probs.squeeze())
                att_log_probs.append(att_log_prob)
                att_values.append(value.squeeze())
                
            elif player == -1:
                vec = g.get_defend_vector(board)
                probs, value = def_ac.forward(vec)
                valids = valids[g.size:]
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
            
        
        #update attack actor critic
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

        att_loss.backward()

        att_optimizer.step()
        att_optimizer.zero_grad()

        att_ac.eval()
        
        #update defense actor critic
        
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


