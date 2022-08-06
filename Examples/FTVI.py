from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import numpy as np
import timeit
import networkx as nx
import matplotlib.pyplot as plt

#Construct graph from the frozenlake 
#state if reachable from s0
#directed edge if transition with p(s2, s1) > 0 between states
def construct_Graph(env):
    G = nx.DiGraph()
    for s in range(env.nS):
        G.add_node(s, attr=env.P[s])
        for a in env.P[s]:
            for prob, n_s, reward, done in env.P[s][a]:
                if prob > 0:
                    G.add_edge(s,n_s)
    return G

#perform VI on subset of the environment
def VI(env, V, nodes, theta=1e-6, gamma=1.):
    def update_value(state, V):
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, n_state, reward, done in env.P[state][a]:
                A[a] += prob*(reward + gamma*V[n_state])
        return A
    while True:
        delta = 0.
        for s in nodes:
            v_old = V[s]
            V[s] = np.max(update_value(s, V))
            delta = max(delta, np.abs(V[s] - v_old))
            
        if delta < theta:
            break
    return V

#Perform TVI
def TVI(env, theta = 1e-6):
    #Build graph from env
    G_r = construct_Graph(env)
    
    #Check how many connected components (perhaps VI is better if it is small)
    if nx.number_strongly_connected_components(G_r) > 10:
        #Meta nodes graph
        G_scc = nx.condensation(G_r)
        scc_coupling = G_scc.graph["mapping"]
        top_sort = list(reversed(list(nx.topological_sort(G_scc))))
        
        V = np.zeros(env.nS)
        for meta_node in top_sort:
            #Get real nodes in meta node
            nodes = [k for k,v in scc_coupling.items() if v == meta_node]
            
            #perform VI on the nodes
            V = VI(env, V, nodes, theta)
            
    return V

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

env  =FrozenLakeEnv(desc=MAP)
start = timeit.default_timer()
V = TVI(env)  
stop = timeit.default_timer()


















#FVTI for later 

def search(s):
    return 0

def FTVI(env, x=5, y=3):
    V_lowerb = 0.
    while True:
        old_v = V_lowerb
        for it in range(x):
            for s in range(env.nS):
                print("add to graph and label unviseted")
            s = 0 #Init state is 0
            search(s)
        if old_v/V_lowerb > (100 - y)/100:
            break
            
    
    
    return 0