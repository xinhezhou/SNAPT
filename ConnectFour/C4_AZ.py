import numpy as np
import torch
import torch.optim as optim
import time
import copy
import gc


class AlphaZero():
    def __init__(self, game, nnet, mcts, args):
        self.g = game
        self.args = args
        self.nnet = nnet
        self.mcts = mcts
        self.sigma = copy.copy(nnet)
        
        with torch.no_grad():
            for param in self.sigma.nnet.parameters():
                param.divide_(10)
        
    def execute_episode(self, render = False):
        board = self.g.getInitBoard()
        player = 1
        trainExamples = []
        boards = []
        pis = []
        total_moves = 0
        while True:
            canonicalBoard = self.g.getCanonicalForm(board, player)
            temp = 1 
            pi = self.mcts.getActionProb(canonicalBoard, temp = temp)
            a = np.random.choice(len(pi), p = pi)
            trainExamples.append((board, player, pi))
            boards.append(board)
            pis.append(pi)
            board, player = self.g.getNextState(board, player, a) 
            total_moves +=1
            
            if render:
                print('Probs: {}\nAction: {}\nBoard: \n{}\n'.format(pi, a, board))
                
            r = self.g.getGameEnded(board, player)
            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != player))) for x in trainExamples]
            

        value = boards[-1][3]

        return [(b, p, value) for (b,p) in zip(boards, pis)] 
    
    def train_gradient(self):
        start = time.time()
        all_episodes = []
        for k in range(self.args.numEps):
            all_episodes += self.execute_episode()


        boards, pis, vs = list(zip(*all_episodes))

        boards = torch.FloatTensor(np.array(boards).astype(np.float64))
        target_pis = torch.FloatTensor(np.array(pis))
        target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

        out_pi, out_v = self.nnet.nnet.forward(boards)
        l_pi = self.nnet.loss_pi(target_pis, out_pi)
        log_l_pi = self.nnet.log_loss_pi(target_pis, out_pi)
        l_v = self.nnet.loss_v(target_vs, out_v)
        total_loss = l_pi + l_v
        
        optimizer = optim.Adam(self.nnet.nnet.parameters())
        self.nnet.nnet.train()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        return time.time()-start

        
        
    def weighted_sum(self, models, weights):
        total = copy.copy(models[0])
        with torch.no_grad():
            for param in total.nnet.parameters():
                param.multiply_(weights[0])

            for model, weight in zip(models[1:], weights[1:]):
                for param, total_param in zip(model.nnet.parameters(), total.nnet.parameters()):
                    total_param.add_(param*weight)

        return total

    def weighted_std(self, models, weights, mean, noise = 0):
        total = copy.copy(models[0])
        with torch.no_grad():
            for param in total.nnet.parameters():
                param.multiply_(0)

            for model, weight in zip(models, weights):
                for param, mean_param, total_param in zip(model.nnet.parameters(), mean.nnet.parameters(), total.nnet.parameters()):
                    diff = param - mean_param
                    diff.multiply_(diff)
                    total_param.add_(diff*weight)
                    total_param.add_(noise)

            for param in total.nnet.parameters():
                param.sqrt_()
        return total    
        
    def log_weights(self, elite_size):
        weights = [np.log(elite_size + 1)/ (k+1) for k in range(elite_size)]
        return weights/sum(weights)
    
    
    def add_noise(self, model, std = 0, division = 1):
        new_model = copy.copy(model)
        with torch.no_grad():
            if std == 0:
                for param in new_model.nnet.parameters():
                    noise = torch.randn(param.size()) * 0.1
                    param.add_(noise)
            else:
                for param, std_param in zip(new_model.nnet.parameters(), std.nnet.parameters()):
                    param.add_(torch.randn(param.size()) * std_param / division)

        return new_model

    
    def calculate_loss(self, model, target_pis, target_vs, boards):
        out_pi, out_v = model.nnet.forward(boards)
        l_pi = self.nnet.loss_pi(target_pis, out_pi)
        log_l_pi = self.nnet.log_loss_pi(target_pis, out_pi)
        l_v = self.nnet.loss_v(target_vs, out_v)
        return l_pi + l_v
    
        
    
    def train_es(self):
        start = time.time()
        all_episodes = []
        for k in range(self.args.numEps):
            all_episodes += self.execute_episode()


        boards, pis, vs = list(zip(*all_episodes))
        
        boards = torch.FloatTensor(np.array(boards).astype(np.float64))
        target_pis = torch.FloatTensor(np.array(pis))
        target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
        
        
        sample_models = [self.add_noise(self.nnet, std = self.sigma) for k in range(self.args.batch_size)]
        losses = [self.calculate_loss(model, target_pis, target_vs, boards) for model in sample_models]
        

        weights = self.log_weights(self.args.elite_size)
        
        
        d = {losses[k]: sample_models[k] for k in range(self.args.batch_size)}
        sort_keys = sorted(d)
        elites = [d[key] for key in sort_keys[:self.args.elite_size]]
        
        self.nnet = self.weighted_sum(elites, weights)
        sigma = self.weighted_std(elites, weights, self.nnet, noise = 0.0001)
        
        return time.time()-start