from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


def plot_values(V):
	# reshape value function
	V_sq = np.reshape(V, (4,4))

	# plot the state-value function
	fig = plt.figure(figsize=(6, 6))
	ax = fig.add_subplot(111)
	im = ax.imshow(V_sq, cmap='cool')
	for (j,i),label in np.ndenumerate(V_sq):
	    ax.text(i, j, np.round(label, 5), ha='center', va='center', fontsize=14)
	plt.tick_params(bottom='off', left='off', labelbottom='off', labelleft='off')
	plt.title('State-Value Function')
	plt.show()
    

env = FrozenLakeEnv()

print(env.observation_space)
print(env.action_space)

print(env.nS)
print(env.nA)

print(env.P[1][0][0])

#iterarive policy iteration
#Input:
#       1) Env  2) Policy 3) gamma 4) theta 
#Output:
#       1) Value function
def policy_evaluation(env, policy, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            Vs = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    Vs += action_prob*prob*(reward + gamma*V[next_state])
            delta = max(delta, np.abs(V[s] - Vs))
            V[s] = Vs
        if delta < theta:
            break
    return V

def q_from_v(env, V, s, gamma=1):
    q = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[s][a]:
            q[a] += prob * (reward + gamma * V[next_state])
    return q

random_policy = np.ones([env.nS, env.nA]) / env.nA# evaluate the policy 
V = policy_evaluation(env, random_policy)

Q = np.zeros([env.nS, env.nA])
for s in range(env.nS):
    Q[s] = q_from_v(env, V, s)
print("Action-Value Function:")
print(Q)