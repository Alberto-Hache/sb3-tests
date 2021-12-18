# Getting started with 'stablebaselines3'
# https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html

# A quick example of how to train, save and run A2C on a CartPole
# environment.

import gym

from stable_baselines3 import A2C

# Define the environment.
env = gym.make('CartPole-v1')

# Define and train the model.
model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# Save the model.
model.save('tmp/A2C-CarPole-v1')

# ...

# Load the model.
new_model = A2C.load('tmp/A2C-CarPole-v1')

# Run the trained model.
obs = env.reset()
for i in range(1000):
    action, _state = new_model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
