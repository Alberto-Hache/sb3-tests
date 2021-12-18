"""
RL with Atari games:

0. Install gym with atari AND the ROMs:
    > pip install gym[atari,accept-rom-license]
    ...
    > Successfully installed atari-py-0.2.6

1. Install Arcade Learning Environment (ALE)
(https://github.com/mgbellemare/Arcade-Learning-Environment)
    > pip install ale-py
    ...
    > Successfully installed ale-py-0.7.3 importlib-resources-5.4.0 zipp-3.6.0

"""

"""
NEXT STEPS (2. 3. 4.) ARE NO LONGER REQUIRED:

2. Download Roms.rar from the Atari 2600 VCS ROM Collection:
(http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html)

3. Extract the .rar file and copy all .bin files to roms/ in your project dir.

4. Import all supported ROMs by the ALE.
    > ale-import-roms roms/
    [SUPPORTED]         koolaid roms/Kool-Aid Man (Kool Aid Pitcher Man) (1983)...
    ...
    [NOT SUPPORTED]             roms/Targ (1983) (CBS Electronics - VSS) (80110)...
    Imported 110 / 1996 ROMs

"""

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=4, seed=0)
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)

model = A2C('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=5_000)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
