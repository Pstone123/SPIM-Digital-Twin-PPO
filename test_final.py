import pickle
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

class ForwardModel(nn.Module):
    def __init__(self):
        super(ForwardModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )
    def forward(self, x):
        return self.net(x)

ckpt = torch.load('C:/Users/MaMak/forward_model.pth',
                   map_location='cpu', weights_only=False)
print('Keys in checkpoint:', list(ckpt.keys()))

fm = ForwardModel()
fm.load_state_dict(ckpt['model'])
fm.eval()

scaler_in  = ckpt['scaler_in']
scaler_out = ckpt['scaler_out']

print('Forward model and scalers loaded')

class SPIMForwardEnv(gym.Env):
    def __init__(self):
        super(SPIMForwardEnv, self).__init__()
        self.min_speed = 400.0
        self.max_speed = 1480.0
        self.max_steps = 200
        self.rng       = self.max_speed - self.min_speed
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32))
        self.observation_space = spaces.Box(
            low=-np.ones(7, dtype=np.float32),
            high=np.ones(7, dtype=np.float32))

    def _ns(self, speed):
        d = np.zeros((1,5)); d[0,1] = speed
        return scaler_in.transform(d)[0,1]

    def _ds(self, val):
        d = np.zeros((1,4)); d[0,0] = val
        return scaler_out.inverse_transform(d)[0,0]

    def _get_obs(self):
        sn = self.state[0]
        tn = self._ns(self.target_speed)
        return np.array([sn, tn, tn-sn,
                         self.state[1], self.state[2],
                         self.state[3], self.prev_action],
                        dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.target_speed = float(np.random.uniform(600, 1400))
        init_speed        = float(np.random.uniform(600, 1400))
        self.step_count   = 0
        self.prev_action  = 0.5
        self.prev_error   = abs(self.target_speed - init_speed)
        a = self.prev_action
        v = (0.4 + a*0.6) * 230
        row = scaler_in.transform([[v, init_speed, 2.0+a*3,
                                     1.0+a*2, 1.5+a*4]])[0]
        with torch.no_grad():
            self.state = fm(torch.FloatTensor(row).unsqueeze(0)).numpy()[0]
        return self._get_obs(), {}

    def step(self, action):
        a   = float(np.clip(action[0], 0.0, 1.0))
        ma  = 0.4 + a * 0.6
