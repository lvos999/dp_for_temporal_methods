from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import numpy as np
import timeit

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
    'FFHFHFFFFFFFHHFFFFFG'
]

env = FrozenLakeEnv(desc=MAP)

def value_iteration(env, theta=1e-6, gamma=1.):
    
    def update_value(state, V):
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, n_state, reward, done in env.P[state][a]:
                A[a] += prob*(reward + gamma*V[n_state])
        return A
    
    V  = np.zeros(env.nS)
    while True:
        delta = 0.
        for s in range(env.nS):
            v_old = V[s]
            V[s] = np.max(update_value(s, V))
            delta = max(delta, np.abs(V[s] - v_old))
            
        if delta < theta:
            break
        
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        policy[s] = np.argmax(update_value(s, V))
    return policy, V

start = timeit.default_timer()
[pol, v]  = value_iteration(env, theta=1e-6, gamma=1.)
stop = timeit.default_timer()


print('Time: ', stop - start)  

#Standard VI takes around:
    #15 [sec]



#LEFT  = 0
#DOWN  = 1
#RIGHT = 2
#UP    = 3

#np.set_printoptions(precision=2)
#print("Value table")
#print(v.reshape(len(MAP_20x20),len(MAP_20x20[0])))
#print("policy")
#print(pol.reshape(len(MAP_20x20),len(MAP_20x20[0])))
    