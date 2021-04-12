import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages')
from Connect4Game import C4Game
from C4_net import NNetWrapper as nn
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from mcts_c4 import MCTS
from AZ_C4 import AlphaZero
import time
import pandas as pd


args = dotdict({
    'numIters': 1000,
    'numEps': 20,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.5,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200,    # Number of game examples to train the neural networks.
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 10,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'batch_size' : 20,
    'elite_size' : 10,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})



g = C4Game()

nnet_grad = nn(g)
mcts_grad = MCTS(g, nnet_grad, args)
AZ_grad = AlphaZero(g, nnet_grad, mcts_grad, args)

nnet_es = nn(g)
mcts_es = MCTS(g, nnet_es, args)
AZ_es = AlphaZero(g, nnet_es, mcts_es, args)

mcts_untrained = MCTS(g, nn(g), args)



training_time = 3600

print(nnet_grad.predict(g.getInitBoard()))
print(nnet_es.predict(g.getInitBoard()))
AZ_grad.train_session_grad(time_cap = training_time)
AZ_es.train_session_es(time_cap = training_time)
print(nnet_es.predict(g.getInitBoard()))
print(nnet_grad.predict(g.getInitBoard()))


nnet_grad.save_checkpoint(filename = 'nnet_grad_checkpoint.pth.tar')
nnet_es.save_checkpoint(filename = 'nnet_es_checkpoint.pth.tar')



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
        
        a = np.random.choice(len(probs), p = probs)
        board, player = g.getNextState(board, player, a)
        move_count+=1
        if render:
            print('Move {}: {} to {}'.format(move_count,-player, a))
            print(board, '\n\n')
        
        r = g.getGameEnded(board, player)
        if r != 0:
            return r * player
            
        
def play_games(g, total = 100, player1 = 0, player2 = 0, temp = 0):
    wins = 0
    start = time.time()
    current = time.time()
    for k in range(total):
        w = play_game(g, player1 = player1, player2 = player2, render = False, temp = temp)
        if w == 1:
            wins += 1

        if time.time() - current > 60:
            current = time.time()
            print('{} wins in {} games. {} seconds elapsed'.format(wins, k + 1, np.round(time.time() - start)))


    print('Won {} games out of {}. Done in {} seconds'.format(wins, total, np.round(time.time() - start)))
    return wins
        

total_games = 100
    
all_wins = []    
for player1 in [mcts_grad, mcts_es, mcts_untrained, 0]:
    player_wins = []
    for player2 in [mcts_grad, mcts_es, mcts_untrained, 0]:
        player_wins.append(play_games(g, total = total_games, player1 = player1, player2 = player2, temp = 0))
    
    all_wins.append(player_wins)

    
df = pd.DataFrame(all_wins, columns =['grad', 'es','untrained','random']) 
df.to_csv(r'training_wins.csv')
print(all_wins)

