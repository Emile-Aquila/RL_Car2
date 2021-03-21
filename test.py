import os
import gym
import gym_donkeycar
import numpy as np
# SET UP ENVIRONMENT
# You can also launch the simulator separately
# in that case, you don't need to pass a `conf` object
exe_path = f"/home/emile/.local/lib/python3.9/site-packages/gym_donkeycar/DonkeySimLinux/donkey_sim.x86_64"
port = 9091

conf = { "exe_path" : exe_path, "port" : port }

#env = gym.make("donkey-generated-track-v0", conf=conf)
env = gym.make("donkey-generated-track-v0", conf=conf)

# PLAY
obs = env.reset()
for t in range(100):
	action = np.array([0.0, 0.5]) # drive straight with small speed
  	# execute the action
	obs, reward, done, info = env.step(action)
	#print("obs : {}, {}".format(obs, obs.shape))
	#print("rew : {}".format(reward))
	#print("info : {}".format(info))

# Exit the scene
env.close()
