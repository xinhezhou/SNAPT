from SNAPT_Game import SNAPT_Game as SNAPT
from SNAPT_AZ import *
from SNAPT_net import NNetWrapper as nn
from matplotlib import pylab as plt
import time

weights = [[1, 1, 0],
         [1, 1, 1],
         [0, 1, 1]]

machine_attributes = [[1, 0, 0.5, 0, 0],
          [0, 0, 0.5, 0, 0],
          [0, 1, 0.5, 0, 0]]

p1_attributes = [1, 1, 1, 20, 1]
p2_attributes = [1, 1, 1, 20, 1]

g = SNAPT(weights, machine_attributes, p1_attributes, p2_attributes)
board, player = g.getInitBoard()
t_max = 60

args = dotdict({
    'numEps': 5,        # Number of complete self-play games to simulate during a new iteration.
    'numMCTSSims': 20,  # Number of games moves for MCTS to simulate.
    'cpuct': 1,         # hyperparameter for MCTS
    'batch_size' : 8,  # number of samples to take for AZ-ES, N in paper
    'elite_size' : 4,  # elite size for AZ-ES, K in paper
})

def train_AZ(g, nnet, use_gradient = True, t_max = 3600):
    """
    training method for AlphaZero and AlphaZero-ES
    g: Game to train on 
    nnet: neural network to train
    grad: If true, train with gradient/traditional AZ, otherwise use AlphaZero-ES
    t_max: total training time, 3600 in paper
    """
    
    # set up neural network, MCTS, and AlphaZero objects
    mcts = MCTS(g, nnet, args)
    AZ = AlphaZero(g, nnet, mcts, args)
    att_losses = []
    def_losses = []

    # training loop
    start = time.time()
    iterations = 0
    
    while (time.time() - start) < t_max:
        iterations += 1
        if use_gradient:
            att_loss, def_loss, _  = AZ.train_gradient()
        else:
            att_loss, def_loss, _  = AZ.train_es()
        att_losses.append(att_loss)
        def_losses.append(def_loss)

    # print number of iterations and total training time
    print(iterations)
    print(time.time() - start)


    # save attack and defense neural networks
    # they will be stored in folder 'checkpoint'
    # don't overwrite what is already there unless you want to train new models
    # plt.figure(figsize=(15,10))
    # plt.ylabel("Loss")
    # plt.xlabel("Training Steps")
    # plt.plot(att_losses)
    # if use_gradient:
    #     plt.savefig("AZgrad_att_losses.pdf", format="pdf")
    # else:
    #     plt.savefig("AZes_att_losses.pdf", format="pdf")
    
    # plt.figure(figsize=(15,10))
    # plt.ylabel("Loss")
    # plt.xlabel("Training Steps")
    # plt.plot(def_losses)
    # if use_gradient:
    #     plt.savefig("AZgrad_def_losses.pdf", format="pdf")
    # else:
    #     plt.savefig("AZes_def_losses.pdf", format="pdf")

    nnet.save_checkpoint(att_filename = 'att_temp.pth.tar', def_filename = 'def_temp.pth.tar')

nnet = nn(g)
train_AZ(g, nnet, use_gradient = True, t_max = t_max)
train_AZ(g, nnet, use_gradient = False, t_max = t_max)