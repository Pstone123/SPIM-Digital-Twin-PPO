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

fm = ForwardModel()
ckpt = torch.load('C:/Users/MaMak/forward_model.pth',
                   map_location='cpu', weights_only=False)
fm.load_state_dict(ckpt['model_state_dict'])
fm.eval()

with open('C:/Users/MaMak/scaler_in.pkl', 'rb') as f:
    scaler_in = pickle.load(f)
with open('C:/Users/MaMak/scaler_out.pkl', 'rb') as f:
    scaler_out = pickle.load(f)

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
        v   = ma * 230
        cs  = float(np.clip(self._ds(self.state[0]),
                            self.min_speed, self.max_speed))
        im  = float(scaler_out.inverse_transform(
                    [[0, self.state[1], 0, 0]])[0,1])
        ia  = float(scaler_out.inverse_transform(
                    [[0, 0, self.state[2], 0]])[0,2])
        tq  = float(scaler_out.inverse_transform(
                    [[0, 0, 0, self.state[3]]])[0,3])
        row = scaler_in.transform([[v, cs, im, ia, tq]])[0]
        with torch.no_grad():
            self.state = fm(torch.FloatTensor(row).unsqueeze(0)).numpy()[0]
        next_speed = float(np.clip(self._ds(self.state[0]),
                                    self.min_speed, self.max_speed))
        self.step_count += 1
        error    = abs(self.target_speed - next_speed)
        prev_err = self.prev_error
        reward   = -error / self.rng
        if error < 1.0:    reward += 2.0
        elif error < 20.0: reward += 1.0 / max(error, 0.1)
        elif error < 80.0: reward += 0.3 / max(error, 0.1)
        if error < prev_err: reward += 0.5
        else:                reward -= 0.1
        reward -= 0.05 * abs(a - self.prev_action)
        if self.step_count % 50 == 0:
            self.target_speed = float(np.random.uniform(600, 1400))
        self.prev_error  = error
        self.prev_action = a
        done = self.step_count >= self.max_steps
        return self._get_obs(), reward, done, False, {}

ppo = PPO.load('C:/Users/MaMak/ppo_spim_final')
print('PPO agent loaded')

print('\nEvaluating PPO...')
print('='*50)
test_targets = [500, 700, 900, 1100, 1300]
errors = []

for target in test_targets:
    env_eval = SPIMForwardEnv()
    obs, _ = env_eval.reset()
    env_eval.target_speed = float(target)
    obs = env_eval._get_obs()
    speeds = []
    for _ in range(200):
        action, _ = ppo.predict(obs, deterministic=True)
        obs, _, done, _, _ = env_eval.step(action)
        speed = float(np.clip(env_eval._ds(env_eval.state[0]),
                              env_eval.min_speed, env_eval.max_speed))
        speeds.append(speed)
        if done: break
    error = np.mean(np.abs(np.array(speeds[-50:]) - target))
    errors.append(error)
    print(f'Target {target:4d} RPM | Error: {error:6.2f} RPM')

print(f'\nMean error: {np.mean(errors):.2f} RPM')

print('\nAction analysis:')
for target in [600, 900, 1200]:
    env_eval = SPIMForwardEnv()
    obs, _ = env_eval.reset()
    env_eval.target_speed = float(target)
    obs = env_eval._get_obs()
    acts = []
    for _ in range(20):
        action, _ = ppo.predict(obs, deterministic=True)
        obs, _, _, _, _ = env_eval.step(action)
        acts.append(float(action[0]))
    avg = np.mean(acts)
    print(f'Target {target} RPM | Action: {avg:.4f} | Freq: {20+avg*30:.1f} Hz')
