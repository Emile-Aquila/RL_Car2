import torch
from algo import Trainer
# import pybullet_envs
# import pybullet
import gym_donkeycar
import gym
from SAC import SAC
from env import MyEnv
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
exe_path = f"/home/emile/.local/lib/python3.9/site-packages/gym_donkeycar/DonkeySimLinux/donkey_sim.x86_64"
conf = {"exe_path": exe_path, "port": 9091}
env = gym.make("donkey-generated-track-v0", conf=conf)
env = MyEnv(env)


# ENV_ID = 'Pendulum-v0'
SEED = 0
REWARD_SCALE = 1.0
# NUM_STEPS = 5 * 10 ** 4
NUM_STEPS = 3 * 10 ** 6
# NUM_STEPS = 10**2
EVAL_INTERVAL = 2 * 10 ** 3

print("state shape {}".format(*env.observation_space))
print("action shape {}".format(env.action_space.shape))

algo = SAC(
    state_shape=env.observation_space,
    # state_shape=32,
    action_shape=env.action_space.shape,
    seed=SEED,
    reward_scale=REWARD_SCALE,
)

trainer = Trainer(
    env=env,
    # env_test=env_test,
    algo=algo,
    seed=SEED,
    num_steps=NUM_STEPS,
    eval_interval=EVAL_INTERVAL,
)
print("start train")
trainer.train()
