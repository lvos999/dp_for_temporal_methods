from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import numpy as np
import timeit
import networkx as nx
import matplotlib.pyplot as plt


def change_rewards(env):
    MAP_string =  env.desc.flatten()
    for s in range(env.nS):
        for a in env.P[s]:
            for count, val in enumerate(env.P[s][a]):
                if MAP_string[val[1]].decode("utf-8") == 'H':
                    env.P[s][a][count] = (val[0], val[1], -1., val[3])
    return env

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
def TVI(env, G_r, theta = 1e-6):    
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


#FVTI for later 
def search(env, belman_error, G, V_lowb, V_uppb,  s, gamma = 1.):
    #if G.nodes[s]['sink'] == True:
    if s == 199:
        backup_error, G, V_lowb, V_uppb = backup(env, G, V_lowb, V_uppb, s)
        return max(belman_error, backup_error), G, V_lowb, V_uppb
    
    G.nodes[s]['visited'] = True
    Q = np.zeros(len(env.P[s]))
    for a in env.P[s]:
        for prob, n_state, reward, done in env.P[s][a]:
            Q[a] += prob*(reward + gamma*V_lowb[n_state])
            
    greedy_a = np.argmin(Q)
    for prob, n_state, reward, done in env.P[s][greedy_a]:
        if G.nodes[n_state]['visited'] == False:
            search(env, belman_error, G, V_lowb, V_uppb, n_state, gamma)
    #print('checks')
    backup_error, G, V_lowb, V_uppb = backup(env, G, V_lowb, V_uppb, s)
    return max(belman_error, backup_error), G, V_lowb, V_uppb

#Make a back up for upper and lower bound of value functions
def backup(env, G, V_lowb, V_uppb,  s, gamma = 1.):
    Q_lowb = np.zeros(len(env.P[s]))
    Q_uppb = np.zeros(len(env.P[s]))

    for a in env.P[s]:
        for prob, n_state, reward, done in env.P[s][a]:
            Q_lowb[a] += prob*(reward + gamma*V_lowb[n_state])
            Q_uppb[a] += prob*(reward + gamma*V_uppb[n_state])
        if Q_lowb[a] > V_uppb[s]:
            for _, n_state, _, _  in env.P[s][a]:
                if G.has_edge(s, n_state):
                    G.remove_edge(s, n_state)
    
    oldV_l = V_lowb[s]
    V_lowb[s] = np.amin(Q_lowb)
    V_uppb[s] = np.amin(Q_uppb)
    return np.abs(V_lowb[s] - oldV_l), G, V_lowb, V_uppb

def lower_heuristic(env, G):
    V_lowb = np.zeros(env.nS)
    
    for s in range(env.nS):
        if G.nodes[s]['sink'] == True:
            V_lowb[s] = 0
    
    return V_lowb

def upper_heuristic(env, G):
    V_uppb = np.full(env.nS, np.inf)
    
    for s in range(env.nS):
        if G.nodes[s]['sink'] == True:
            V_uppb[s] = 0
    
    return V_uppb

def FTVI(env, G, x=100, y=3, theta=1e-6, gamma = 1.):
    #Keep track of both lower and upper bound of the optimal value function
    V_lowb = lower_heuristic(env, G)
    V_uppb = upper_heuristic(env, G)
    
    #Perform small heuristic search:
        #Stop if problems converges and return value function 
        #Stop if updates have little use and perform TVI
    while True:
        old_v = V_lowb[0]                     
        for it in range(x):
            belman_error = 0.
            for s in range(env.nS):             
                G.nodes[s]['visited'] = False
            s = 0                      
            
            belman_error, G, V_lowb, V_uppb = search(env, belman_error, G, V_lowb, V_uppb, s, gamma)
            if belman_error < theta:
                print(belman_error)
                return V_lowb
        if old_v/V_lowb[0] > (100 - y)/100:
            print('TVI')
            break
        
    #V_lowerb = TVI(env, G)                      #Call TVI algorithm on new graph
    return G

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
G_r = construct_Graph(env)

start = timeit.default_timer()
V = TVI(env, G_r)
stop = timeit.default_timer()
print('Time: ', stop - start)





