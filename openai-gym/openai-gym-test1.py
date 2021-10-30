import gym
env = gym.make('MsPacman-v0')
env.reset()
for t in range(10000):
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample()) # take a random action
    # if done: break
env.close()
