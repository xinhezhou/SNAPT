import sys
sys.path.append('../')
from es_utils import *
from Connect4Game import C4Game
from C4_net import NNetWrapper as nn
from C4_net import C4_net
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import gc

g = C4Game()

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


def get_win_count(g, mu, models, games_as_each = 50, temp = 0):
    win_dict = {}
    count = 0
    for m in models:
        count += 1
        wins = play_equal(g, each = games_as_each, player1 = mu, player2 = m, temp = temp)
        win_dict[wins + count/(10*len(models))] = m
    return win_dict


def cem_C4(g, iters = 100, batch_size = 20, elite_size = 10, episode_count = 10, weight_type = 'even', temp = 1, t_max = 3600, t_checks = 10):
    mu = C4_net(g)
    sigma = C4_net(g)
    rewards = []
    with torch.no_grad():
        for param in sigma.parameters():
            param.divide_(10)
    start = time.time()
    for i in range(iters):
            
        models = [add_noise(mu, std = sigma) for k in range(batch_size)]
        d = get_win_count(g, mu, models, games_as_each = episode_count, temp = temp)
        sort_keys = list(reversed(sorted(d)))
        elites = [d[key] for key in sort_keys[:elite_size]]

        if weight_type == 'even':
            weights = [1/elite_size for k in range(elite_size)]
        elif weight_type == 'log':
            weights = log_weights(elite_size)
        else:
            print('bruhbruh')
            return None

        sigma = weighted_std(elites, weights, mu, noise = 0.000001)
        mu = weighted_sum(elites, weights)
        rewards.append(play_equal(g, player1 = mu, player2 = 0, each = episode_count, temp = temp))
        gc.collect()
        current = time.time()
        if current -start > t_max:
            print('Finished in {} seconds and {} iterations'.format(np.round(current-start), i+1))
            break
        if (i + 1)  % max(1,(iters// 20)) == 0:
            
            print('Finished iteration {}. {} seconds elapsed.'.format(i+1, np.round(current-start, 2)))
            print(get_first_param(mu))
            print(get_first_param(sigma))
            
        
            
    return mu, rewards


def oneone_C4(g, iters = 100, episode_count = 10, temp = 1, t_max = 3600, t_checks = 10):
    mu = C4_net(g)
    sigma = C4_net(g)
    rewards = []
    with torch.no_grad():
        for param in sigma.parameters():
            param.divide_(10)
    start = time.time()
    for i in range(iters): 
        new_mu = add_noise(mu, std = sigma)
        
        wins = play_equal(g, player1 = mu, player2 = new_mu, each = episode_count, temp = temp)
        if wins > episode_count:
            sigma = weighted_std([new_mu], [1], mu, noise = 0.000001)
            mu = new_mu

        rewards.append(play_equal(g, player1 = mu, player2 = 0, each = episode_count, temp = temp))
        gc.collect()
        current = time.time()
        if current -start > t_max:
            print('Finished in {} seconds and {} iterations'.format(np.round(current-start), i+1))
            break
        if (i + 1)  % max(1,(iters// 50)) == 0:
            
            print('Finished iteration {}. {} seconds elapsed.'.format(i+1, np.round(current-start, 2)))
            print(get_first_param(mu))
            print(get_first_param(sigma))
    return mu, rewards


mu, r = oneone_C4(g, iters = 5000, episode_count = 50, temp = 1, t_max = 3600)

import os
fname= os.path.join('checkpoint', 'nnet_one_one_3600.pth.tar')
torch.save(mu.state_dict(), fname)