# much of the a2c code is taken from 
# https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

import sys

sys.path.append('../')
from Connect4Game import C4Game
from C4_net import C4_net
from es_utils import *
import numpy as np
import time
import torch.optim as optim
from matplotlib import pylab as plt
import gc

def play_game(g, player1 = 0, player2 = 0, render = False, temp = 0):
    board = g.getInitBoard()
    player = 1
    move_count = 0
    while True:
        canon = g.getCanonicalForm(board, player)
        if player == 1:
            agent = player1
        if player == -1:
            agent = player2
            
        if agent == 0:
            probs = g.getValidMoves(board, player)
            probs = probs/sum(probs)
        else:    
            probs = agent.getActionProb(canon, temp = temp)
        
        probs = probs/np.sum(probs)
        a = np.random.choice(len(probs), p = probs)
        board, player = g.getNextState(board, player, a)
        move_count+=1
        if render:
            print('Move {}: {} to {}'.format(move_count,-player, a))
            print(board, '\n\n')
        
        r = g.getGameEnded(board, player)
        if r != 0:
            return r * player
            
        
def play_games(g, total = 100, player1 = 0, player2 = 0, temp = 0, render = False):
    wins = 0
    start = time.time()
    current = time.time()
    for k in range(total):
        w = play_game(g, player1 = player1, player2 = player2, render = False, temp = temp)
        if w == 1:
            wins += 1

        if time.time() - current > 60:
            current = time.time()
            if render:
                print('{} wins in {} games. {} seconds elapsed'.format(wins, k + 1, np.round(time.time() - start)))

    if render:
        print('Won {} games out of {}. Done in {} seconds'.format(wins, total, np.round(time.time() - start)))
    return wins
        
def play_equal(g, each = 50, player1 = 0, player2 = 0, temp = 0, render = False):
    wins = play_games(g, total = each, player1 = player1, player2 = player2, temp = temp, render = render)
    return wins + each - play_games(g, total = each, player1 = player2, player2 = player1, temp = temp, render = render)


def a2c(g, iters = 100, t_max = 3600):
    start =time.time()
    learning_rate = 0.001
    actor_critic = C4_net(g)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)
    actor_critic.eval()
    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0
    losses = []
    
    for i in range(iters):
        gc.collect()
        log_probs = []
        values = []
        players = []

        board = g.getInitBoard()
        player = 1
        while True:
            state = g.getCanonicalForm(board, player)
            s = torch.FloatTensor(state.astype(np.float64))
            
            log_prob, value = actor_critic.forward(s)
            probs = torch.exp(log_prob)
            value = value.detach().numpy()[0,0]
            probs = np.squeeze(probs.detach().numpy())
            
            action = np.random.choice(len(probs), p=probs)

            entropy = -np.sum(np.mean(probs) * np.log(probs))
            board, player = g.getNextState(board, player, action)
            

            values.append(value)
            log_probs.append(log_prob)
            players.append(player)
            entropy_term += entropy
            
            r = g.getGameEnded(board, player)
            
            if r != 0:
                
                break
        
        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(values))):
            Qvals[t] = r * players[t]
            
        
        #update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)
        
        advantage = Qvals - values
        advantage = advantage.unsqueeze(-1)
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.0001 * entropy_term
        losses.append(ac_loss.detach())
        
        #actor_critic.train()
        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()
        current = time.time()
        if current -start > t_max:
            print('Finished in {} seconds and {} iterations'.format(np.round(current-start), i+1))
            break
        
        if (i + 1)  % max(1,(iters// 20)) == 0:
            
            print('Finished iteration {}. {} seconds elapsed.'.format(i+1, np.round(current-start, 2)))
            print(get_first_param(actor_critic))
            print(play_equal(g, player1 = actor_critic))
            print(ac_loss)
        
    
    plt.figure(figsize=(15,10))
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.plot(losses)
    plt.savefig("a2c_losses.pdf", format="pdf")
    return actor_critic
