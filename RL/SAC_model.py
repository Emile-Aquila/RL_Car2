import torch
import torch.nn as nn
from algo import reparameterize
import gym_donkeycar
from env import MyEnv
import gym


class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.contiguous().view(inputs.size(0), -1)


def init_weights(m):
    if (type(m) is nn.Conv2d) or (type(m) is nn.Linear):
        nn.init.kaiming_normal_(m.weight)


class CriticNetwork2(nn.Module):
    def __init__(self):
        super().__init__()
        num = 64
        self.net1 = nn.Sequential(
            nn.Conv2d(3, num, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num, 64, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, 4, stride=2),
            nn.ReLU(inplace=True),
            Flatten(),  # torch.Size([1, 192])
            nn.Linear(192, 64),  # torch.Size([1, 64])
            # nn.Linear(64, 2 * 2),
        )
        self.net1_2 = nn.Sequential(
            nn.Conv2d(3, num, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num, 64, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, 4, stride=2),
            nn.ReLU(inplace=True),
            Flatten(),  # torch.Size([1, 192])
            nn.Linear(192, 64),  # torch.Size([1, 64])
            # nn.Linear(64, 2 * 2),
        )
        self.net2 = nn.Sequential(
            nn.Linear(2, 64)
        )
        self.net2_2 = nn.Sequential(
            nn.Linear(2, 64)
        )
        self.net3 = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.net3_2 = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.net1.apply(init_weights)
        self.net1_2.apply(init_weights)
        self.net2.apply(init_weights)
        self.net2_2.apply(init_weights)
        self.net3.apply(init_weights)
        self.net3_2.apply(init_weights)

    def forward(self, states, acts):
        s1, s2 = self.net1(states), self.net1_2(states)
        a1, a2 = self.net2(acts), self.net2_2(acts)
        t1 = torch.cat((s1, a1), dim=-1)
        t2 = torch.cat((s2, a2), dim=-1)
        out1, out2 = self.net3(t1), self.net3_2(t2)
        return out1, out2


class ActorNetwork2(nn.Module):
    def __init__(self):
        super().__init__()
        num = 64
        self.net = nn.Sequential(
            nn.Conv2d(3, num, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num, 64, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, 4, stride=2),
            nn.ReLU(inplace=True),
            Flatten(),  # torch.Size([1, 192])
            nn.Linear(192, 64),  # torch.Size([1, 64])
            nn.ReLU(inplace=True),
            nn.Linear(64, 2 * 2),
        )
        self.net.apply(init_weights)

    def forward(self, states):
        # print(states.shape)
        # print("net {}".format(self.net(states).shape))
        means, log_stds = self.net(states).chunk(2, dim=-1)
        # print("m,l {}, {}".format(means.shape, log_stds.shape))
        return means, log_stds

    def sample(self, inputs, deterministic=False):
        #  select action from inputs
        means, log_stds = self.forward(inputs)
        if deterministic:
            return torch.tanh(means)
        else:
            log_stds = torch.clip(log_stds, -20.0, 2.0)
            return reparameterize(means, log_stds)


class ActorNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        num = 256
        self.net = nn.Sequential(
            # nn.Linear(state_shape[0], num),
            nn.Linear(state_shape, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2 * action_shape[0]),
        )

    def forward(self, inputs):
        # calc means and log_stds
        means, log_stds = self.net(inputs).chunk(2, dim=-1)
        return means, log_stds

    def sample(self, inputs, deterministic=False):
        #  select action from inputs
        means, log_stds = self.forward(inputs)
        if deterministic:
            return torch.tanh(means)
        else:
            log_stds = torch.clip(log_stds, -20.0, 2.0)
            return reparameterize(means, log_stds)


class CriticNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        num = 256
        self.net1 = nn.Sequential(
            nn.Linear(state_shape + action_shape[0], num),
            nn.ReLU(inplace=True),
            nn.Linear(num, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.net2 = nn.Sequential(
            nn.Linear(state_shape + action_shape[0], num),
            nn.ReLU(inplace=True),
            nn.Linear(num, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, states, actions):
        inputs = torch.cat((states, actions), dim=-1)
        return self.net1(inputs), self.net2(inputs)

def main():
    exe_path = f"/home/emile/.local/lib/python3.9/site-packages/gym_donkeycar/DonkeySimLinux/donkey_sim.x86_64"
    conf = {"exe_path": exe_path, "port": 9091}
    env = gym.make("donkey-generated-track-v0", conf=conf)
    env = MyEnv(env)
    print("action space {}".format(env.action_space))
    # test = ActorNetwork2()
    # stat = env.reset()
    # stat = torch.Tensor(stat).permute(0, 3, 1, 2)
    # act = torch.zeros((1, 2))
    # test.forward(stat)
    # test2 = CriticNetwork2()
    # ans = test2.forward(stat, act)
    # print("ans {}".format(ans))


if __name__ == "__main__":
    main()
