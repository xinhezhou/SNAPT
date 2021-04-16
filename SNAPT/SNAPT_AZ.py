# Majority of code is slightly modified from 
# https://github.com/suragnair/alpha-zero-general

import sys
sys.path.append('../')
from es_utils import *
import math
import numpy as np
from utils import *
import torch
import torch.optim as optim
import time
import copy
import gc

EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        board, player = canonicalBoard
        s = self.game.stringRepresentation(canonicalBoard)
        
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(board, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s] * player

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(*canonicalBoard)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(board, player, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
    


lr = 0.001


class AlphaZero():
    def __init__(self, game, nnet, mcts, args):
        self.g = game
        self.args = args
        self.nnet = nnet
        self.mcts = mcts
        self.sigma = copy.copy(nnet)
        
        with torch.no_grad():
            for param in self.sigma.att_nnet.parameters():
                param.divide_(10)
            for param in self.sigma.def_nnet.parameters():
                param.divide_(10)
        
    def execute_episode(self, render = False):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.
        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        board, player = self.g.getInitBoard()
        trainExamples = []
        att_boards = []
        att_pis = []
        
        def_boards = []
        def_pis = []
        
        total_moves = 0
        while True:
            canonicalBoard = self.g.getCanonicalForm(board, player)
            temp = 1
            pi = self.mcts.getActionProb(canonicalBoard, temp = temp)
            a = np.random.choice(len(pi), p = pi)
            
            if player == 1:
                att_boards.append(board)
                att_pis.append(pi)
                
            elif player == -1:
                def_boards.append(board)
                def_pis.append(pi)
                
            else:
                print("bruhhhh")

                
            board, player = self.g.getNextState(board, player, a) 
            total_moves +=1
            
            if render:
                print('Probs: {}\nAction: {}\n'.format(pi, a))
                
            r = self.g.getGameEnded(board, player)
            if r != 0:
                return att_boards, att_pis, def_boards, def_pis, r
            
    def att_pi_clip(self, pi):
        # get attacker probabilities from full probability vector
        return pi[:self.g.size]
    
    def def_pi_clip(self, pi):
        # get defender probabilities from full probability vector
        return pi[self.g.size:]
    
    def train_gradient(self):
        """
        Execute one training iteration of traditional/gradient AlphaZero.
        Updates self.nnet with trained models and returns total time taken
        """
        start = time.time()
        
        # execute episodes and get game histories
        att_data = []
        def_data = []
        for k in range(self.args.numEps):
            att_boards, att_pis, def_boards, def_pis, r = self.execute_episode()
            att_data += [(self.g.get_attack_vector(board), self.att_pi_clip(pi), r) for (board, pi) in zip(att_boards, att_pis)]
            def_data += [(self.g.get_defend_vector(board), self.def_pi_clip(pi), r) for (board, pi) in zip(def_boards, def_pis)]
            
        
        # calculate attack network loss and backpropagate
        att_boards, att_pis, att_vs = list(zip(*att_data))
        
        att_boards = np.array(list(att_boards))
        att_pis = torch.FloatTensor(np.array(list(att_pis)))
        att_vs = torch.FloatTensor(np.array(list(att_vs)))
        
        att_loss = self.calculate_loss(self.nnet.att_nnet, att_pis, att_vs, att_boards)
            
        att_optimizer = optim.Adam(self.nnet.att_nnet.parameters(), lr = lr)
        self.nnet.att_nnet.train()

        att_optimizer.zero_grad()
        att_loss.backward()
        att_optimizer.step()
        self.nnet.att_nnet.eval()
        
        
        
        # calculate defense network loss and backpropagate
        def_boards, def_pis, def_vs = list(zip(*def_data))
        
        def_boards = np.array(list(def_boards))
        def_pis = torch.FloatTensor(np.array(list(def_pis)))
        def_vs = torch.FloatTensor(np.array(list(def_vs)))
        
        def_loss = self.calculate_loss(self.nnet.def_nnet, def_pis, def_vs, def_boards)
            
        def_optimizer = optim.Adam(self.nnet.def_nnet.parameters(), lr = lr)
        self.nnet.def_nnet.train()

        def_optimizer.zero_grad()
        def_loss.backward()
        def_optimizer.step()
        self.nnet.def_nnet.eval()

        return time.time()-start
  
    
    def train_es(self):
        """
        Execute one training iteration of AlphaZero-ES.
        Updates self.nnet with trained models and returns total time taken
        """
        start = time.time()
        
        # execute episodes and get game histories
        att_data = []
        def_data = []
        for k in range(self.args.numEps):
            att_boards, att_pis, def_boards, def_pis, r = self.execute_episode()
            att_data += [(self.g.get_attack_vector(board), self.att_pi_clip(pi), r) for (board, pi) in zip(att_boards, att_pis)]
            def_data += [(self.g.get_defend_vector(board), self.def_pi_clip(pi), r) for (board, pi) in zip(def_boards, def_pis)]
            
        
        
        # calculate attack loss
        att_boards, att_pis, att_vs = list(zip(*att_data))
        
        att_boards = np.array(list(att_boards))
        att_pis = torch.FloatTensor(np.array(list(att_pis)))
        att_vs = torch.FloatTensor(np.array(list(att_vs)))
        
        sample_models = [add_noise(self.nnet.att_nnet, std = self.sigma.att_nnet) for k in range(self.args.batch_size)]
        att_models = [add_noise(self.nnet.att_nnet, std = self.sigma.att_nnet) for k in range(self.args.batch_size)]
        att_losses = [self.calculate_loss(model, att_pis, att_vs, att_boards) for model in att_models]
            
        # update attack network with CEM-style ES
        weights = log_weights(self.args.elite_size)
        d = {att_losses[k]: att_models[k] for k in range(self.args.batch_size)}
        sort_keys = sorted(d)
        elites = [d[key] for key in sort_keys[:self.args.elite_size]]
        self.sigma.att_nnet = weighted_std(elites, weights, self.nnet.att_nnet, noise = 0.0001)
        self.nnet.att_nnet = weighted_sum(elites, weights)
        gc.collect()
        
        #calculate defense loss
        def_boards, def_pis, def_vs = list(zip(*def_data))
        
        def_boards = np.array(list(def_boards))
        def_pis = torch.FloatTensor(np.array(list(def_pis)))
        def_vs = torch.FloatTensor(np.array(list(def_vs)))
        
        sample_models = [add_noise(self.nnet.def_nnet, std = self.sigma.def_nnet) for k in range(self.args.batch_size)]
        def_models = [add_noise(self.nnet.def_nnet, std = self.sigma.def_nnet) for k in range(self.args.batch_size)]
        def_losses = [self.calculate_loss(model, def_pis, def_vs, def_boards) for model in def_models]
            
            
        # update defense network with CEM-style ES
        d = {def_losses[k]: def_models[k] for k in range(self.args.batch_size)}
        sort_keys = sorted(d)
        elites = [d[key] for key in sort_keys[:self.args.elite_size]]
        self.sigma.def_nnet = weighted_std(elites, weights, self.nnet.def_nnet, noise = 0.0001)
        self.nnet.def_nnet = weighted_sum(elites, weights)
        gc.collect()
        
        return time.time()-start
    
    def calculate_loss(self, model, target_pis, target_vs, boards):
        """
        Calculate AlphaZero loss function.
        See original paper for formula.
        """
        out_pi, out_v = model.forward(boards)
        l_pi = self.nnet.loss_pi(target_pis, out_pi)
        l_v = self.nnet.loss_v(target_vs, out_v)
        return l_pi + l_v
        
