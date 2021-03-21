from vae.vae import VAE
import numpy as np
import torch
import gym
from PIL import Image
import cv2
import os
import shutil


class MyEnv:
    def __init__(self, env_):
        self.env = env_
        self.dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.action_space = env_.action_space
        # self.observation_space = (1, 32)

        # vae
        self.vae = VAE()
        model_path = "./vae/vae.pth"
        self.vae.load_state_dict(torch.load(model_path))
        self.vae.to(self.dev)
        self._gen_id = 0  # 何回目のgenerateかを保持
        self._frames = []  # mp4生成用にframeを保存

    def step(self, action, show=False):
        n_state, rew, done, info = self.env.step(action)
        if show:
            self._frames.append(np.array(n_state))
        n_state = self.convert_state_vae(n_state)
        rew = self.change_rew(rew, info)
        if info["cte"] > 3.5:
            done = True
            rew = -1.0
        elif info["cte"] < -5.0:
            done = True
            rew = -1.0
        return n_state, rew, done, info

    def change_rew(self, rew, info):
        if info["speed"] < 0.0:
            return -0.6
        elif abs(info["cte"]) >= 2.0:
            return -1.0
        if rew > 0.0:
            rew /= 20.0
            if info["speed"] > 3.0:
                rew += info["speed"] / 30.0
        return rew

    def reset(self):
        state = self.env.reset()
        state = self.convert_state_vae(state)
        return state

    def seed(self, seed_):
        self.env.seed(seed_)

    def convert_state_to_tensor(self, state):  # state(array) -> np.array -> convert some -> tensor
        state_ = np.array(state).reshape((160, 120, 3))
        # print("state_ shape {}".format(state1.shape))
        state_ = state_[0:160, 40:120, :].reshape((1, 80, 160, 3))
        # print("state shape {}".format(state_.shape))
        # state_ = state_.reshape((1, 80, 160, 3))
        state_ = torch.from_numpy(state_).permute(0, 3, 1, 2).float().to(self.dev)
        state_ /= 255.0
        return state_

    def convert_state_vae(self, state):
        state_ = self.convert_state_to_tensor(state)
        state_, _, _ = self.vae.encode(state_)
        state_ = state_.clone().detach().cpu().numpy()[0]
        return state_

    def generate_mp4(self):
        # for mp4
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = 10
        video = cv2.VideoWriter('./mp4/output' + str(self._gen_id) + ".mp4", fourcc, fps, (120, 160))
        if not video.isOpened():
            print("can't be opened")
        # for path
        os.mkdir("./tmp")
        current_path = os.getcwd()  # 現在のディレクトリ
        # main procedure
        for idx, frame in enumerate(self._frames):
            fram = Image.fromarray(frame, "RGB")
            path = current_path + "/tmp/frame" + str(idx) + ".png"
            fram.save(path, 'PNG')
            img = cv2.imread(path)
            img = cv2.resize(img, (120, 160))
            if img is None:
                print("can't read")
                break
            video.write(img)
        video.release()
        shutil.rmtree("./tmp")
        self._frames.clear()
        self._gen_id += 1


