import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

class Player():
    def __init__(self, atts):
        """
        0: sophistication
        1: determination
        2: wealth
        3: initative -> the key number
        4: insight
        """
        # only initiative is implemented as of now, this is the value "player moves"
        self.atts = atts
        
    def spend_init(self, cost = 1):
        self.atts[3] -= cost
        
    def init(self):
        return self.atts[3]
        
        
        
        
class Machine():
    def __init__(self, atts):
        # 0: compromised state-> 0 = secure, 1 = vulnerable, 2 = exploited
        # 1: reward from hacking/value of machine
        # 2: probability of success from attacking action
        # 3: 1 if machine was just successfully attacked, 0 otherwise
        # 4: the number of times the machine has been secured
        self.atts = atts
        
    def attack_prob(self):
        return self.atts[2]
    
    def attack(self):
        if np.random.random_sample() < self.atts[2]:
            self.atts[0] = min(self.atts[0] + 1, 2) # increase security state by 1
            self.atts[3] = 1
            return 1
        
        return 0
            
    def detect(self, detect_prob = 0.5):
        if self.atts[3] == 0 or self.atts[0] != 2 : # if the device was not exploited in the previous turn
            return 0
        if np.random.random_sample() < detect_prob: # where does p_e come in?
            self.atts[0] = 1 # reduce security state from compromised to vulnerable
            return 1
        
        return 0
    
    def reset(self):
        self.atts[3] = 0
        
    def secure(self, factor = 0.1):
        if self.atts[4] < 3:
            self.atts[2]  = max(0, self.atts[2] - factor)
            self.atts[4] += 1
            return 1
        
        return 0
    
    def value(self):
        return self.atts[1]
    
    def get_reward(self):
        return (self.atts[0] == 2) * self.atts[1]
    
    def state(self):
        return self.atts[0]
    
    def public_data(self):
        return self.atts[:2] + self.atts[3:4]
    
    def defender_data(self):
        return self.atts[1:3] + self.atts[4:]
        
    


class SNAPT_Game():
    def __init__(self, weights, machine_atts, p1_atts, p2_atts, goal = 1):
        
        self.weights = weights # edges 
        self.p1 = Player(p1_atts)
        self.p2 = Player(p2_atts)
        self.machines = [Machine(m_atts) for m_atts in machine_atts]
        assert len(weights) == len(machine_atts)
        self.size = len(weights)
        self.goal = goal
        
    def getInitBoard(self):
        return (self.p1, self.p2, self.machines), 1
    
    def getBoardSize(self):
        return (len(machine_atts), len(machine_atts[0]))
    
    def getActionSize(self):
        return 3 * self.size
    
    def getNextState(self, board, player, action, render = False):
        action_dict = {0: 'Attacked', 1: 'Detected', 2: 'Secured'}
        result_dict = {0: 'failed', 1: 'success'}
        
        p1, p2, machines = copy.deepcopy(board)
        action_type = action // self.size
        target = machines[action % self.size]
        
        if action_type == 0:
            for m in machines:
                m.reset()
            result = target.attack()
            
            
        elif action_type == 1:
            result = target.detect()
            
        elif action_type == 2:
            result = target.secure()
            
        else:
            print('bruh')
            
        if render:
            print('{} machine {}. Result: {}'.format(action_dict[action_type], action % self.size, result_dict[result]))
            
        if player == 1:
            p1.spend_init()
            
        elif player == -1:
            p2.spend_init()
            
        else:
            print('bruh')
            
        return (p1, p2, machines), -player
            
        
    def getValue(self, board):
        p1, p2, machines = board
        return sum([m.get_reward() for m in machines])
    
    def getValidMoves(self, board, player):
        if player == -1:
            return ([0] * self.size) + ([1] * (2 * self.size))  # p_e 
        
        p1, p2, machines = board
        access_mat = [[machine.state()] * self.size for machine in machines]
         # use matrix multiplication to find all vulnerable (and connected to vulnerable) devices 
        access_mat = np.array(access_mat) * np.array(self.weights) # this is fishy...might need to transpose access_mat
        valids = list(np.sum(access_mat, axis = 0))
        valids = [v > 0 for v in valids]
        return valids + ([0] * (2 * self.size)) 
    
    
    def getGameEnded(self, board, player):
        p1, p2, machines = board
        if p1.init() > 0:
            return 0
        
        val = -1
        
        if self.getValue(board) >= self.goal:
            val = 1
            
        return val
        
        
    def getCanonicalForm(self, board, player):
        return board, player

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        return str(board)

    def render(self, board, player):
        p1, p2, machines = board
        
        color_dict = {0: 'green', 1: 'orange', 2: 'red'}
        state_dict = {0: 'S', 1: 'V', 2: 'C'}
        colors = [color_dict[m.state()] for m in machines]
        names = [str((state_dict[m.state()], m.value(), m.attack_prob())) for m in machines]
        G = nx.DiGraph(np.array(self.weights))
        nx.draw(G)
        
        layout = nx.spring_layout(G)
        
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog = 'neato')#, args = '-Grankdir=LR')
        nx.draw(G, pos, node_color=colors, labels = dict(zip(sorted(G), names)), with_labels = True)
        plt.show()
        
        print(p1.init(), p2.init(), self.getGameEnded(board, player))
        
    def get_attack_vector(self, board):
        p1, p2, machines = board
        
        v = copy.deepcopy(p1.atts)
        for m in machines:
            v += m.public_data()
            
        return np.array(v)
    
    def get_defend_vector(self, board):
        p1, p2, machines = board
        
        v = copy.deepcopy(p2.atts)
        
        for m in machines:
            v += m.defender_data()
            
        return np.array(v)
    
    def stringRepresentation(self, canon):
        board, player = canon
        if player == 1:
            return str(self.get_attack_vector(board))
        else:
            return str(self.get_defend_vector(board))
    
    

    


