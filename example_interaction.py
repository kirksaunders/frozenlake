#Requirements: 
# python3.x
# gym (pip install gym)
import random
from time import sleep
import numpy as np

random.seed(0)

#This 'special' Frozen Lake allows access to transition dynamics
from FrozenLakeEnv import FrozenLakeEnv

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


def main():
    # Stochastic
    env = FrozenLakeEnv(map_name="4x4", is_slippery=True)
    env.seed(0) #set env seed

    v, pi = value_iteration(env, 1e-6, 1)
    #v, pi = policy_iteration(env, 1e-6, 1)

    np.set_printoptions(linewidth=1000)
    print("state value function:\n", np.reshape(v, (4,4)))
    print("optimal policy:\n", np.reshape(pi, (4,4)))

    # Interact with environment following optimal policy
    total_reward = 0
    for episode in range(10000):
        state = env.reset()
        for t in range(10000):
            #env.render()
            #sleep(1)
            action = pi[state]
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                #print("Episode {} finished after {} timesteps with reward {}".format(episode, t+1, reward))
                break
    print("Total reward: {}".format(total_reward))
    env.close()

    # State Layout
    # 0  1  2  3
    # 4  5  6  7
    # 8  9  10 11
    # 12 13 14 15

    #current_state = 14  # State from S_n=16 State space
    #action = 2  # Left action from A_n=4 Action space

    #print(env.P[current_state][action])

    # [(0.3333333333333333, 14, 0.0, False), #0.3333333333333333 chance action 2 in state 14 moves to state 14, receives reward 0, and does not terminate the episode
    # (0.3333333333333333, 15, 1.0, True), #0.3333333333333333 chance action 2 in state 14 moves to state 15, receives reward 1, and terminates the episode
    # (0.3333333333333333, 10, 0.0, False)] #0.3333333333333333 chance action 2 in state 14 moves to state 10, receives reward 0, and does not terminate the episode

    #env.reset()
    #env.render()

    #For more information on FrozenLake see: https://github.com/openai/gym/wiki/FrozenLake-v0
    #For more information on installation and interaction with openai gym see: http://gym.openai.com/docs/ 
    #let's randomly interact with the environment for 20 episodes
    #for i_episode in range(20):
    #    observation = env.reset()
    #    for t in range(100):
    #        env.render()
    #        sleep(1)
    #        action = env.action_space.sample()
    #        observation, reward, done, info = env.step(action)
    #        if done:
    #            print("Episode finished after {} timesteps".format(t+1))
    #            break
    #env.close()

if __name__=='__main__':
    main()