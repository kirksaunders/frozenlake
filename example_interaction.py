# Requirements: 
# python3.x
# gym (pip install gym)
import random
from time import sleep
import numpy as np

random.seed(0)

#from gym.envs.toy_text import FrozenLakeEnv
from frozen_lake import FrozenLakeEnv

MAX_ITERATIONS = 100

def evaluate(env, pi, num_eps=100):
    total_reward = 0
    for i in range(num_eps):
        state, done = env.reset(), False
        it = 0
        while not done and it < MAX_ITERATIONS:
            action = pi[state]
            state, reward, done, _ = env.step(action)
            total_reward += reward
            it += 1
    print("Total reward is {} out of {} episodes. Average reward per episode is {}".format(total_reward, num_eps, total_reward / num_eps))

def get_action_value(env, state, action, v, gamma):
    value = 0
    for probability, next_state, reward, _ in env.P[state][action]:
            value += probability * (reward + gamma*v[next_state])

    return value

def get_action_values(env, state, v, gamma):
    values = np.zeros(env.nA)
    for action in range(env.nA):
        values[action] = get_action_value(env, state, action, v, gamma)

    return values

def value_iteration(env, theta, gamma):
    # Initialize state value function to all 0's
    v = np.zeros(env.nS)

    # Perform value iteration
    while True:
        delta = 0
        for state in range(env.nS):
            new_v = np.max(get_action_values(env, state, v, gamma))
            delta = max(delta, abs(v[state] - new_v))
            v[state] = new_v

        if delta < theta:
            break

    # Determine optimal deterministic policy from value function
    pi = np.zeros(env.nS).astype(int)
    for state in range(env.nS):
        pi[state] = np.argmax(get_action_values(env, state, v, gamma))

    return (v, pi)

def policy_iteration(env, theta, gamma):
    # Initialize state value function to all 0's
    v = np.zeros(env.nS)
    v[env.nS-1] = 0

    # Initialize policy to always go left (not ideal, but should work anyways)
    pi = np.zeros(env.nS).astype(int)

    while True:
        # Perform policy evaluation
        while True:
            delta = 0
            for state in range(env.nS):
                new_v = get_action_value(env, state, pi[state], v, gamma)
                delta = max(delta, abs(v[state] - new_v))
                v[state] = new_v

            if delta < theta:
                break

        # Perform policy improvement
        stable = True
        for state in range(env.nS):
            best_action = np.argmax(get_action_values(env, state, v, gamma))
            if pi[state] != best_action:
                stable = False
            pi[state] = best_action

        if stable:
            break

    return (v, pi)

def monte_carlo(env, gamma, epsilon):
    q = np.random.rand(env.nS, env.nA)
    c = np.zeros((env.nS, env.nA))
    pi = np.zeros(env.nS).astype(int)

    iterations = 0
    while True:
        # Generate episode with equal probility stochastic policy
        s = []
        a = []
        r = []
        state, done = env.reset(), False
        it = 0
        while not done and it < MAX_ITERATIONS:
            s.append(state)
            if random.random() > epsilon:
                action = pi[state]
            else:
                action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            a.append(action)
            r.append(reward)
            it += 1

        g = 0.0
        w = 1.0
        for t in reversed(range(len(s))):
            g = gamma*g + r[t]
            c[s[t], a[t]] += w
            q[s[t], a[t]] += w/c[s[t], a[t]] * (g - q[s[t], a[t]])
            pi[s[t]] = np.argmax(q[s[t], :])

            if a[t] != pi[s[t]]:
                break
            w /= (1.0 - epsilon) + epsilon / env.nA

        iterations += 1
        if iterations % 10000 == 0:
            print("{} iterations of monte carlo control completed, evaluating policy".format(iterations))
            evaluate(env, pi)
            
def q_learning(env, gamma, alpha, epsilon):
    q = np.zeros((env.nS, env.nA))
    pi = np.zeros(env.nS).astype(int)

    iterations = 0
    while True:
        it = 0
        state, done = env.reset(), False
        while not done and it < MAX_ITERATIONS:
            if random.random() > epsilon:
                action = pi[state]
            else:
                action = env.action_space.sample()
            
            new_state, reward, done, _ = env.step(action)
            q[state, action] += alpha * (reward + gamma*np.max(q[new_state, :]) - q[state, action])
            pi[state] = np.argmax(q[state, :])
            state = new_state

        iterations += 1
        if iterations % 10000 == 0:
            print("{} iterations of q-learning completed, evaluating policy".format(iterations))
            evaluate(env, pi)

def main():
    # Stochastic
    env = FrozenLakeEnv(map_name="4x4", is_slippery=True)
    env.seed(0) #set env seed

    #v, pi = value_iteration(env, 1e-3, 1)
    #v, pi = policy_iteration(env, 1e-0, 1)

    #evaluate(env, pi)

    #monte_carlo(env, 1, 0.1)
    q_learning(env, 1, 0.01, 0.1)

    env.close()

if __name__=='__main__':
    main()