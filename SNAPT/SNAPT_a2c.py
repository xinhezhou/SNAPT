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
    att_log_probs = []
    att_values = []
        
    def_log_probs = []
    def_values = []

    entropy_term = 0


    while g.getGameEnded(board, player) == 0:
        # compute prob distribution
        valids = g.getValidMoves(board, player) # valid positions
        if player == 1:
            vec = g.get_attack_vector(board)
            log_probs, value = agent_1.forward(vec)
        elif player == -1:
            vec = g.get_defend_vector(board)
            log_probs, value = agent_2.forward(vec)
        
        if player == 1:
            att_values.append(value)
            att_log_probs.append(log_probs)
        elif player == -1:
            def_values.append(value)
            def_log_probs.append(log_probs)
        
        if torch.is_tensor(log_probs):
            log_probs = log_probs.detach().numpy()
        if torch.is_tensor(value):
            value = value.detach().numpy()

        probs = np.exp(log_probs)
        entropy = -np.sum(np.mean(probs) * log_probs)
        entropy_term += entropy
            
        # pick an action
        if temp != 0:
            probs = np.power(log_probs, temp)
        probs = np.array(probs) * np.array(valids)
        probs = np.squeeze(probs)
        if sum(probs) == 0:
            print('Probabilities are all zero! That\'s not good!')
        probs = probs / np.sum(probs)
        if temp == 0:
            action = np.argmax(probs)
        else:
            action = np.random.choice(len(probs), p = probs)
        
        board, player = g.getNextState(board, player, action, render = render)
        # if render:
        #     g.render(board, player)
    
        r = g.getGameEnded(board, player)
        if r != 0:
            att_reward = r
            def_reward = -1 * r
            return att_values, def_values, att_log_probs, def_log_probs, att_reward, def_reward, entropy_term
        
def update_params(optimizer,values,log_probs,reward, entropy_term):
    values = torch.stack(values)
    log_probs = torch.stack(log_probs)
    rewards = np.zeros_like(values.detach().numpy()) + reward
    rewards = torch.FloatTensor(rewards.astype(np.float32))
    advantage = (rewards - values).unsqueeze(-1)
    # advantage = torch.abs((rewards - values).unsqueeze(-1))

    actor_loss = (-log_probs * advantage).mean()
    critic_loss = advantage.pow(2).mean()
    loss = actor_loss + critic_loss + 0.001* entropy_term
    # print(actor_loss, critic_loss, entropy_term, loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss



def attacker_vs_defender(g, attacker, defender, episode_count, temp = 1):
    wins = 0
    for k in range(episode_count):
        winner = play_game(g, attacker, defender, temp = temp)[4]
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
    att_losses = []
    def_losses = []
    
    for i in range(iters):
        gc.collect()
        att_values, def_values, att_log_probs, def_log_probs, att_reward, def_reward, entropy_term = play_game(g, att_ac, def_ac, render = False, temp = 1)

        
        
        
        #update attack actor critic
        att_ac.train()
        att_loss = update_params(att_optimizer, att_values, att_log_probs,att_reward, entropy_term)
        att_losses.append(att_loss.detach().numpy())
        att_ac.eval()
        
        #update defense actor critic
        
        def_ac.train()
        def_loss = update_params(def_optimizer, def_values, def_log_probs,def_reward, entropy_term)
        def_losses.append(def_loss.detach().numpy())
        def_ac.eval()
        
        
        current = time.time()
        if current -start > t_max:
            print('Finished in {} seconds and {} iterations'.format(np.round(current-start), i+1))
            break
        
        if (i + 1)  % max(1,(iters// 20)) == 0:
            
            print('Finished iteration {}. {} seconds elapsed.'.format(i+1, np.round(current-start, 2)))
        
    plt.figure(figsize=(15,10))
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.plot(att_losses)
    plt.savefig("attacker_losses.pdf", format="pdf")

    plt.figure(figsize=(15,10))
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.plot(def_losses)
    plt.savefig("defender_losses.pdf", format="pdf")
        
    return att_ac, def_ac


