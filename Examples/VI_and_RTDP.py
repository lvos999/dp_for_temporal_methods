from collections import defaultdict

import numpy as np
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv


MAP_20x20 = [
    'SFFFFFFFHFFFFFFHFHFF',
    'FFFFFFFFFFFHFFFFFHFF',
    'FFHFFHFHFFFFFFHFFFFH',
    'FFHFFHFFFFFFFFHFFHFF',
    'FFFHFFFFFFFFFFFFFFHF',
    'FFFFHFFFFFHFFFFHFFFH',
    'FFFFFFFHFHFFHFFFFFFF',
    'HFHFFFFFFFFFFHFFFFFF',
    'HFFFFFFFFHHFHFFHHFFF',
    'FFFFFFFFFHFHFFFFFFFF',
    'FFFFFFFFFFFFHFFFFFFH',
    'FFFFFFFHFFFFFFFFFFFH',
    'FFFFFFHFFFFFFFFFHHFF',
    'HFFHFFFHHFHFFFHHFFFF',
    'FFFFFFFFFHFHFFHHHFFF',
    'HFFFFFHFFFFFHFHFFFFF',
    'HFFFFFFFFFFFFFFFHFFH',
    'FHFFFFFFFHFFFFFFFFFF',
    'FFHFFFFFFFHFFFFHFHFF',
    'FFHFHFFFFFFFHHFFFFFG'
]

env = FrozenLakeEnv()

def q_from_v(env, V, s, gamma=1):
    q = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[s][a]:
            q[a] += prob * (reward + gamma * V[next_state])
    return q

def policy_improvement(env, V, gamma=1):
    policy = np.zeros([env.nS, env.nA]) / env.nA
    for s in range(env.nS):
        q = q_from_v(env, V, s, gamma)
        
        best_a = np.argwhere(q==np.max(q)).flatten()
        policy[s] = np.sum([np.eye(env.nA)[i] for i in best_a], axis=0)/len(best_a)
        
    return policy

#Value iteration on the frozen lake problem
def value_iteration(env, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            V[s] = max(q_from_v(env, V, s, gamma))
            delta =  max(delta, abs(V[s] - v))
        if delta < theta:
            break
    policy = policy_improvement(env, V, gamma)
    return policy, V

class RTDP:
    def __init__(self, env):
        self.env =  env
        self.it = 20_000
        self.gamma = 1
        self.goal_cost = -1
        self.hole_cost = .1
        self.default_cost = .1

        self.actions = [0, 1, 2, 3]
        
        self._init_cost()
        self.calc_policy()
    
    #Look at paper algo:
    def calc_policy(self):
        V = defaultdict(lambda: 0.)
        
        #Repeat trails 
        for i in range(self.it):
            if i%500 == 0:
                print(i)
                
            obs = self.env.reset()
            
           
            #Simulate the dp
            while True:
                #take greedy action 
                action = np.argmin([
                        self.cost[obs][a] + self.gamma * sum(
                            item[0] * V[item[1]] for item in self.env.P[obs][a]
                            )
                        for  a in self.actions
                    ])
                
                #update the value iteration
                V[obs] = self.cost[obs][action] + self.gamma *sum(
                        item[0] * V[item[1]]
                        for item in self.env.P[obs][action]
                    )
                
                #Simulate the next step
                obs, reward, done, _ = self.env.step(action)
                
            if done:
                cost_ = -1
                if reward == 0.0:
                    cost_ = 1
                V[obs] = cost_ + self.gamma * sum(
                    item[0] * V[item[1]]
                    for item in self.env.P[obs][action]
                    )
                break
        print(' ')
        print(V)
        self.V = V
    
    def _init_cost(self):
        cost = defaultdict(dict)
        for state in range(self.env.nS):
            for action in self.actions:
                if len(self.env.P[state][action]) == 1:
                    if self.env.P[state][action][0][2] == 1.0:
                        cost[state][action] = self.goal_cost
                    else:
                        cost[state][action] = self.hole_cost
                    continue

                done = self.env.P[state][action][1][3]
                if done and self.env.P[state][action][1][2] == 0.0:
                    cost[state][action] = self.hole_cost
                else:
                    cost[state][action] = self.default_cost

                if done and self.env.P[state][action][1][2] == 1.0:
                    cost[state][action] = self.goal_cost

        self.cost = cost
        
    def policy(self, state):
        return np.argmin([
            self.cost[state][a] + self.gamma *sum(item[0]*self.V[item[1]]
                                                  for item in self.env.P[state][a]) for a in self.actions ])
policy_vi, V_vi = value_iteration(env)

def evaluate(rtdp, env):
    rewards = 0.
    state = env.reset()
    done = False
    steps = 0
    while not done:
        state, reward, done, _ = env.step(rtdp.policy(state))
        rewards += reward
        steps += 1
        if steps > 1e4:
            break

    return rewards

def run():
    env = FrozenLakeEnv(desc=MAP_20x20)
    rtdp = RTDP(env)
    tot_rewards = 0.
    eval_iter = int(1e4)
    for i in range(eval_iter):
        if i % 100 == 0:
            print(i)
        tot_rewards += evaluate(rtdp, env)
    print(tot_rewards / eval_iter)
    
run()
# print the optimal policy
print("\nOptimal Policy (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3):")
print(policy_vi,"\n")