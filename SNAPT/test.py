from SNAPT_Game import SNAPT_Game as SNAPT
from SNAPT_a2c import *
import os

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

attacker_ac, defender_ac = a2c(g, iters = 20000, t_max = 60)