import copy
import numpy as np
import sys
from SNAPT_net import SNAPT_AC
sys.path.append('../')
from es_utils import *
import time
import numpy as np
import gc

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
    


def cem(g, iters = 100, batch_size = 20, elite_size = 10, episode_count = 10, weight_type = 'even', temp = 1, t_max = 3600):
    
    start = time.time()
    
    #initialize networks
    board, _ = g.getInitBoard()
    attacker_inputs = len(g.get_attack_vector(board))
    defender_inputs = len(g.get_defend_vector(board))
    
    att_mu = SNAPT_AC(attacker_inputs, g.size, 64)
    def_mu = SNAPT_AC(defender_inputs, 2 * g.size, 64)
    
    att_sigma = copy.deepcopy(att_mu)
    def_sigma = copy.deepcopy(def_mu)
    
    # set sigma_0 = mu_0/10
    with torch.no_grad():
        for param in att_sigma.parameters():
            param.divide_(10)
        for param in def_sigma.parameters():
            param.divide_(10)
            
    for i in range(iters):
        
        # get samples
        att_models = [add_noise(att_mu, std = att_sigma) for k in range(batch_size)]
        def_models = [add_noise(def_mu, std = def_sigma) for k in range(batch_size)]
        
        # evaluate each sample
        att_values = [attacker_vs_defender(g, att_model, def_mu, episode_count, temp = temp) for att_model in att_models]
        def_values = [attacker_vs_defender(g, att_mu, def_model, episode_count, temp = temp) for def_model in def_models]
        
        # sort samples by value
        att_dict = {(att_values[k] + (k/(10**5))): att_models[k] for k in range(batch_size)}
        def_dict = {(def_values[k] + (k/(10**5))): def_models[k] for k in range(batch_size)}
        
        att_keys = list(reversed(sorted(att_dict)))
        def_keys = list(sorted(def_dict))
        
        # get elites of samples
        att_elites = [att_dict[key] for key in att_keys[:elite_size]]
        def_elites = [def_dict[key] for key in def_keys[:elite_size]]
        
        # get weights
        if weight_type == 'even':
            weights = [1/elite_size for k in range(elite_size)]
        elif weight_type == 'log':
            weights = log_weights(elite_size)
        else:
            print('Invalid weight type')
            return None
        
        att_sigma = weighted_std(att_elites, weights, att_mu, noise = 0.000001)
        att_mu = weighted_sum(att_elites, weights)
        
        def_sigma = weighted_std(def_elites, weights, def_mu, noise = 0.000001)
        def_mu = weighted_sum(def_elites, weights)
        
        gc.collect()
        current = time.time()
        if current - start > t_max:
            print('Finished in {} seconds and {} iterations'.format(np.round(current-start), i+1))
            break
        if (i + 1)  % (iters// 20) == 0:
            print('Finished iteration {}. {} seconds elapsed.'.format(i+1, np.round(current-start, 2)))
            
    return att_mu, def_mu


def oneone(g, iters = 100, episode_count = 10, temp = 1, t_max = 3600):
    start = time.time()
    
    #initialize networks
    board, _ = g.getInitBoard()
    attacker_inputs = len(g.get_attack_vector(board))
    defender_inputs = len(g.get_defend_vector(board))
    
    att_mu = SNAPT_AC(attacker_inputs, g.size, 64)
    def_mu = SNAPT_AC(defender_inputs, 2 * g.size, 64)
    
    att_sigma = copy.deepcopy(att_mu)
    def_sigma = copy.deepcopy(def_mu)
    
    att_rewards = []
    def_rewards = []
    
    # set sigma_0 = mu_0/10
    with torch.no_grad():
        for param in att_sigma.parameters():
            param.divide_(10)
        for param in def_sigma.parameters():
            param.divide_(10)

    
    for i in range(iters):
        
        # get attacker and defender sample
        att_model = add_noise(att_mu, std = att_sigma)
        def_model = add_noise(def_mu, std = def_sigma)
        
        # Evaluate samples and current mus
        default = attacker_vs_defender(g, att_mu, def_mu, episode_count, temp = temp)
        att_value = attacker_vs_defender(g, att_model, def_mu, episode_count, temp = temp)
        def_value = attacker_vs_defender(g, att_mu, def_model, episode_count, temp = temp)
        
        # if attacker sample performs better than mu, it replaces mu
        if att_value > default:
            att_sigma = weighted_std([att_model], [1], att_mu, noise = 0.000001)
            att_mu = att_model
        
        # if defender sample performs better than mu, it replaces mu
        if def_value < default:
            def_sigma = weighted_std([def_model], [1], def_mu, noise = 0.000001)
            def_mu = def_model
       
        
        gc.collect()
        current = time.time()
        if current -start > t_max:
            print('Finished in {} seconds and {} iterations'.format(np.round(current-start), i+1))
            break
        if (i + 1)  % (iters// 20) == 0:
            print('Finished iteration {}. {} seconds elapsed.'.format(i+1, np.round(current-start, 2)))
            
    return att_mu, def_mu
