from Connect4Game import C4Game

g = C4Game(height=6, width=7, win_length=4)
from C4_a2c import *
import os
actor_critic = a2c(g, iters = 10, t_max = 10)