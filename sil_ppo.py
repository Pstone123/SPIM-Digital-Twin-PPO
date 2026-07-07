import matlab.engine
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import time

print('Starting MATLAB Engine...')
eng = matlab.engine.start_matlab()
print('MATLAB started')

MODEL = 'SPIM_SIL_model'
eng.eval("cd('C:\\Users\\MaMak\\OneDrive\\Desktop\\SIL_model')", nargout=0)
eng.eval("load_system('SPIM_SIL_model')", nargout=0)

eng.eval("""
t = (0:0.00001:10)';
load_profile = zeros(size(t));
load_profile(t >= 0  & t < 2)  = 3;
load_profile(t >= 2  & t < 4)  = 5;
load_profile(t >= 4  & t < 6)  = 7;
load_profile(t >= 6  & t < 8)  = 5;
load_profile(t >= 8  & t <= 10) = 3;
load_ts = timeseries(load_profile, t);
assignin('base', 'load_ts', load_ts);
""", nargout=0)
print('Simulink ready')

class SPIMSILEnv(gym.Env):
    def __init__(self):
        super(SPIMSILEnv, self).__init__()
        self.min_speed = 400.0
        self.max_speed = 1480.0
        self.max_steps = 15
        self.step_dt   = 1.0
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32))
        self.observation_space = spaces.Box(
            low=np.array([-1.0,-1.0,-1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32))

    def _norm(self, speed):
        return (speed - self.min_speed) / (self.max_speed - self.min_speed) * 2 - 1

    def _get_obs(self):
        cn = self._norm(self.current_speed)
        tn = self._norm(self.target_speed)
        return np.array([cn, tn, cn-tn], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.target_speed  = float(np.random.uniform(600, 1400))
        self.current_speed = float(np.random.uniform(600, 1400))
        self.step_count    = 0
        self.prev_error    = abs(self.target_speed - self.current_speed)
        return self._get_obs(), {}

    def step(self, action):
        a  = float(np.clip(action[0], 0.0, 1.0))
        f  = 20 + a * 30
        ma = 0.4 + a * 0.6
        t_end = (self.step_count + 1) * self.step_dt
        eng.eval(f"""
t_step = (0:0.00001:{t_end})';
Vref_step = {ma} .* sin(2*pi*{f}.*t_step);
Vref_ts = timeseries(Vref_step, t_step);
assignin('base', 'Vref_ts', Vref_ts);
set_param('SPIM_SIL_model', 'StopTime', '{t_end}');
out = sim('SPIM_SIL_model');
""", nargout=0)
        speed_raw  = eng.eval("mean(out.wm_log(end-5000:end))", nargout=1)
        next_speed = float(np.clip(speed_raw, self.min_speed, self.max_speed))
        self.current_speed = next_speed
        self.step_count   += 1
        error       = abs(self.target_speed - self.current_speed)
        improvement = self.prev_error - error
        reward      = improvement / (self.max_speed - self.min_speed)
        if error < 10.0:    reward += 2.0
        elif error < 30.0:  reward += 0.5
        elif error < 80.0:  reward += 0.1
        if improvement < 0: reward -= 0.2
        self.prev_error = error
        done = self.step_count >= self.max_steps
        return self._get_obs(), reward, done, False, {}

print('\nTesting SIL environment...')
env_test = SPIMSILEnv()
obs, _ = env_test.reset()
print(f'Target:  {env_test.target_speed:.1f} RPM')
print(f'Current: {env_test.current_speed:.1f} RPM')
obs, r, done, _, _ = env_test.step(np.array([0.5]))
print(f'After action 0.5: speed={env_test.current_speed:.1f} RPM | reward={r:.4f}')
print('Environment test passed')

print('\nStarting PPO SIL training...')
print('Each step runs a real Simulink simulation')

env_train = Monitor(SPIMSILEnv())
ppo = PPO(
    policy        = 'MlpPolicy',
    env           = env_train,
    learning_rate = 3e-4,
    n_steps       = 128,
    batch_size    = 32,
    n_epochs      = 10,
    gamma         = 0.99,
    clip_range    = 0.2,
    verbose       = 1,
    seed          = 42,
    device        = 'cpu'
)

start = time.time()
ppo.learn(total_timesteps=1000)
elapsed = time.time() - start
print(f'\nTraining complete in {elapsed:.0f} seconds')
ppo.save('C:\\Users\\MaMak\\Desktop\\SPIM_SIL\\ppo_sil_agent')
print('PPO agent saved')

print('\nEvaluating PPO SIL agent...')
test_targets = [600, 800, 1000, 1200, 1400]
errors = []

for target in test_targets:
    env_eval = SPIMSILEnv()
    obs, _ = env_eval.reset()
    env_eval.target_speed  = float(target)
    env_eval.current_speed = float(target + np.random.uniform(-50,50))
    obs = env_eval._get_obs()
    for _ in range(10):
        action, _ = ppo.predict(obs, deterministic=True)
        obs, _, done, _, _ = env_eval.step(action)
        if done: break
    error = abs(env_eval.current_speed - target)
    errors.append(error)
    print(f'Target {target} RPM | Final: {env_eval.current_speed:.1f} RPM | Error: {error:.1f} RPM')

print(f'\nMean error: {np.mean(errors):.1f} RPM')
eng.quit()
print('SIL training complete')
