# Requirements: 
# python3.x
# gym (pip install gym)
import random
from time import sleep
import numpy as np

random.seed(0)

#from gym.envs.toy_text import FrozenLakeEnv
from frozen_lake import FrozenLakeEnv

MAX_ITERATIONS = 1000

def value_iteration(env, theta, gamma):
    # Initialize state value function to all 0's
    v = np.zeros(env.nS)

    # Perform value iteration
    while True:
        delta = 0
        for state in range(env.nS):
            old_v = v[state]
            new_v = None
            for action in range(env.nA):
                value = 0
                for p in env.P[state][action]:
                    value += p[0] * (p[2] + gamma*v[p[1]])
                new_v = value if new_v == None else max(new_v, value)
            v[state] = new_v
            delta = max(delta, abs(old_v - new_v))

        if delta < theta:
            break

    # Determine optimal deterministic policy from value function
    pi = np.zeros(env.nS).astype(int)
    for state in range(env.nS):
        best_action = None
        best_value = None
        for action in range(env.nA):
            value = 0
            for p in env.P[state][action]:
                value += p[0] * (p[2] + gamma*v[p[1]])
            if best_action == None or value > best_value:
                best_action = action
                best_value = value
        pi[state] = best_action

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
                old_v = v[state]
                new_v = 0
                for p in env.P[state][pi[state]]:
                    new_v += p[0] * (p[2] + gamma*v[p[1]])
                v[state] = new_v
                delta = max(delta, abs(old_v - new_v))

            if delta < theta:
                break

        # Perform policy improvement
        stable = True
        for state in range(env.nS):
            old_action = pi[state]
            best_action = None
            best_value = None
            for action in range(env.nA):
                value = 0
                for p in env.P[state][action]:
                    value += p[0] * (p[2] + gamma*v[p[1]])
                if best_action == None or value > best_value:
                    best_action = action
                    best_value = value
            pi[state] = best_action
            if old_action != best_action:
                stable = False

        if stable:
            break

    return (v, pi)

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
    print("Total reward is {} out of {} episodes ({}%)".format(total_reward, num_eps, (total_reward / num_eps)*100))

def monte_carlo(env, gamma):
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
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            a.append(action)
            r.append(reward)
            it += 1

        g = 0
        w = 1
        for t in reversed(range(len(s))):
            g = gamma*g + r[t]
            c[s[t], a[t]] += w
            q[s[t], a[t]] += w/c[s[t], a[t]] * (g - q[s[t], a[t]])
            pi[s[t]] = np.argmax(q[s[t], :])

            if a[t] != pi[s[t]]:
                break
            w *= env.nA

        iterations += 1
        if iterations % 10000 == 0:
            print("{} iterations of monte carlo control completed, evaluating policy".format(iterations))
            evaluate(env, pi)
            
def q_learning(env, gamma, alpha, epsilon):
    q = np.zeros((env.nS, env.nA))

    iterations = 0
    while True:
        it = 0
        state, done = env.reset(), False
        while not done and it < MAX_ITERATIONS:
            if random.random() > epsilon:
                action = np.argmax(q[state, :])
            else:
                action = np.random.choice(np.arange(0, env.nA))
            
            new_state, reward, done, _ = env.step(action)
            q[state, action] += alpha * (reward + gamma*np.max(q[new_state, :]) - q[state, action])
            state = new_state

        iterations += 1
        if iterations % 10000 == 0:
            print("{} iterations of q-learning completed, evaluating policy".format(iterations))
            pi = np.zeros(env.nS).astype(int)
            for i in range(env.nS):
                pi[i] = np.argmax(q[i, :])
            evaluate(env, pi)

def main():
    # Stochastic
    env = FrozenLakeEnv(map_name="4x4", is_slippery=True)
    env.seed(0) #set env seed

    v, pi = value_iteration(env, 1e-6, 0.999)
    #v, pi = policy_iteration(env, 1e-6, 0.999)

    evaluate(env, pi)

    #monte_carlo(env, 0.999)
    q_learning(env, 0.999, 0.1, 0.2)

    env.close()

if __name__=='__main__':
    main()