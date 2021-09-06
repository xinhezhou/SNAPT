from Connect4Game import C4Game
from C4_a2c import *
from C4_AZ import *
from C4_es import *
from mcts_c4 import MCTS
from C4_net import NNetWrapper as wrapper
import time
from utils import *


g = C4Game(height=6, width=7, win_length=4)
t_max = 300
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
    losses = []

    # training loop
    start = time.time()
    iterations = 0
    
    while (time.time() - start) < t_max:
        iterations += 1
        if use_gradient:
            loss, _ = AZ.train_gradient()
        else:
            loss, _ = AZ.train_es()
        losses.append(loss)
    # print number of iterations and total training time
    print(iterations)
    print(time.time() - start)


    # save attack and defense neural networks
    # they will be stored in folder 'checkpoint'
    # don't overwrite what is already there unless you want to train new models
    plt.figure(figsize=(15,10))
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.plot(losses)
    if use_gradient:
        plt.savefig("AZgrad_losses.pdf", format="pdf")
    else:
        plt.savefig("AZes_losses.pdf", format="pdf")
    nnet.save_checkpoint(filename = 'temp.pth.tar')

nnet = wrapper(g)


# actor_critic = a2c(g, iters = 20000, t_max = t_max)
# train_AZ(g, nnet, use_gradient = True, t_max = t_max)
# train_AZ(g, nnet, use_gradient = False, t_max = t_max)
mu, rewards = cem(g, iters = 100000, batch_size = 8, elite_size = 4, episode_count = 50, weight_type = 'log', temp = 1, t_max = t_max)
plt.figure(figsize=(15,10))
plt.ylabel("Reward")
plt.xlabel("Training Steps")
plt.plot(rewards)
plt.savefig("cem_rewards.pdf", format="pdf")

mu, rewards = oneone(g, iters = 100000000, episode_count = 50, temp = 1, t_max = t_max)
plt.figure(figsize=(15,10))
plt.ylabel("Reward")
plt.xlabel("Training Steps")
plt.plot(rewards)
plt.savefig("oneone_rewards.pdf", format="pdf")