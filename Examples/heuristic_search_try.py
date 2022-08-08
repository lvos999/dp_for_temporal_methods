from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import numpy as np
import timeit
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

#Construct directed reachability graph for the frozenlake problem
def construct_Graph(env):
    G = nx.DiGraph()
    MAP_string  = env.desc.flatten()
    for s in range(env.nS):
        G.add_node(s, attr=env.P[s])
        
        if (MAP_string[s].decode("utf-8") == 'G') or (MAP_string[s].decode("utf-8") == 'H'):
            G.nodes[s]['sink'] = True
        else:
            G.nodes[s]['sink'] = False
        
        for a in env.P[s]:
            for prob, n_s, reward, done in env.P[s][a]:
                if prob > 0:
                    G.add_edge(s,n_s)
    return G

def change_reward(env):
    for s in range(env.nS):
        for a in env.P[s]:
            for count, val in enumerate(env.P[s][a]):
                if (val[1] == 199) and (s != 199):
                    env.P[s][a][count] = (val[0], val[1], -1.0, val[3])
    return env



class FTVI:
    def __init__(self, env, G, gamma = 1., theta=1e-6):
        self.env = env
        self.G =  self.mark(G)
        self.V_upperb = self.upper_heuristic()
        self.V_lowerb = self.lower_heuristic()
        self.theta = theta
        self.gamma = gamma
        self.belman_error = 0.
    
    def FTVI(self, x=10, y = 1, s0 = 0):
        while True:
            oldV_low = self.V_lowerb[s0]
            
            for it in range(x):
                self.belman_error = 0.
                for s in range(env.nS):
                    ftvi.G.nodes[s]['visited'] = False      
                self.search(s0)
                if self.belman_error < self.theta:
                    return self.V_lowerb
            if oldV_low/self.V_lowerb[s0] > (100 - y)/100:
                break
        
        self.G = construct_Graph(self.env)
        self.TVI()
        return self.V_lowerb
    
    #Perform TVI
    def TVI(self):    
        #Check how many connected components (perhaps VI is better if it is small)
        if nx.number_strongly_connected_components(self.G) > 10:
            #Meta nodes graph
            G_scc = nx.condensation(self.G)
            scc_coupling = G_scc.graph["mapping"]
            top_sort = list(reversed(list(nx.topological_sort(G_scc))))
            
            for meta_node in top_sort:
                #Get real nodes in meta node
                nodes = [k for k,v in scc_coupling.items() if v == meta_node]
                
                #perform VI on the nodes
                self.VI_Gauss_Seidel(nodes)
    
    def VI_Gauss_Seidel(self, nodes):
        def update_value(state, V):
            A = defaultdict(lambda: 0.)
            for a in self.env.P[state]:
                for prob, n_state, reward, done in self.env.P[state][a]:
                    A[a] += prob*(reward + self.gamma*V[n_state])
            return A
            
        while True:
            delta = 0.
            for s in nodes:
                oldV = self.V_lowerb[s]
                self.V_lowerb[s] = min(update_value(s, self.V_lowerb).values())
                delta = max(delta, np.abs(self.V_lowerb[s] - oldV))

            if delta < self.theta:
                break
    
    #perform VI on subset of the environment
    def VI(self,nodes):
        def update_value(state, V):
            A = np.zeros(self.env.nA)
            for a in range(self.env.nA):
                for prob, n_state, reward, done in self.env.P[state][a]:
                    A[a] += prob*(reward + self.gamma*V[n_state])
            return A
        
        while True:
            delta = 0.
            for s in nodes:
                v_old = self.V_lowerb[s]
                self.V_lowerb[s] = np.max(update_value(s, self.V_lowerb))
                delta = max(delta, np.abs(self.V_lowerb[s] - v_old))
            
            if delta < self.theta:
                break
    
    def search(self, s):
        if self.G.nodes[s]['sink'] == True:
            return 
        
        self.G.nodes[s]['visited'] = True
        Q = defaultdict(lambda : 0.)
        
        for a in self.env.P[s]:
            for prob, n_state, reward, done in self.env.P[s][a]:
                Q[a] += prob*(reward + self.gamma*self.V_lowerb[n_state])        
        
        optimal_a = min(Q, key=Q.get)
        #print('optimal action :', optimal_a, ' for state', s)
        for prob, n_state, reward, done in self.env.P[s][optimal_a]:
            if self.G.nodes[n_state]['visited'] == False:
                self.search(n_state)
        
        self.belman_error =  max(self.belman_error, self.back_up(s))
    
    def back_up(self, s):
        Q_low = defaultdict(lambda: 0.)
        Q_upp = defaultdict(lambda: 0.)
        action_elimination = set()
        
        for a in self.env.P[s]:
            for prob, n_state, reward, done in self.env.P[s][a]:
                Q_low[a] += prob*(reward + self.gamma*self.V_lowerb[n_state])
                Q_upp[a] += prob*(reward + self.gamma*self.V_upperb[n_state])
                
            if Q_low[a] > self.V_upperb[s]:
                action_elimination.add(a)
                
        for a in action_elimination:
            del self.env.P[s][a]
            
        oldV_low = self.V_lowerb[s]
        self.V_lowerb[s] = min(Q_low.values())
        self.V_upperb[s] = min(Q_upp.values())
        return np.abs(self.V_lowerb[s] - oldV_low)
    
    def upper_heuristic(self):
        V_uppb = np.ones(env.nS)*0
        for s in range(self.env.nS):
            if self.G.nodes[s]['sink'] == True:
                V_uppb[s] = 0
        return V_uppb
    
    def lower_heuristic(self):
        V_lowb = -1*np.ones(env.nS)
        
        for s in range(self.env.nS):
            if self.G.nodes[s]['sink'] == True:
                V_lowb[s] = 0
        
        return V_lowb
    
    def mark(self, G):
        for s in G.nodes:
            G.nodes[s]['visited'] = False
        return G
                
    
    def draw_graph(self):
        nx.draw(self.G, with_labels=True)

MAP= [
    'SFFFFFFFHFFFFFFHFHFF',
    'FFFFFFFFFFFHFFFFFHFF',
    'FFHFFHFHFFFFFFHFFFFH',
    'FFHFFHFFFFFFFFHFFHFF',
    'FFFHFFFFFFFFFFFFFFHF',
    'FFFFHFFFFFHFFFFHFFFH',
    'HFFFFFFFFFFFFFFFHFFH',
    'FHFFFFFFFHFFFFFFFFFF',
    'FFHFFFFFFFHFFFFHFHFF',
    'FFHFHFFFFFFFHHFFFFFG']

env  = FrozenLakeEnv(desc=MAP)
env  = change_reward(env)
G_r = construct_Graph(env)
ftvi = FTVI(env, G_r)
#ftvi.FTVI()


#Convergence time of FTVI is around 34 seconds
#Convergence time of TVI  is around 34 seconds
print(len(G_r.edges) - len(ftvi.G.edges))

#start = timeit.default_timer()
#f_low = ftvi.TVI()
#stop = timeit.default_timer()
#print('Time: ', stop - start)

#G_r = construct_Graph(env)


    
#print(ftvi.belman_error)
for i in range(100):
    for s in range(env.nS):
        ftvi.G.nodes[s]['visited'] = False        
    for s in range(env.nS):
        ftvi.search(s)
#np.set_printoptions(suppress=True)
#print(ftvi.V_lowerb)
#G_r2 = construct_Graph(ftvi.env)
print(len(G_r.edges) - len(ftvi.G.edges))
#start = timeit.default_timer()
#stop = timeit.default_timer()
#print('Time: ', stop - start)