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
        self.min_speed  = 400.0
        self.max_speed  = 1480.0
        self.max_steps  = 20
        self.step_dt    = 0.2
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32))
        self.observation_space = spaces.Box(
            low=np.array([-1.0,-1.0,-1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32))

    def _norm_error(self, error):
        return np.clip(error / (self.max_speed - self.min_speed), -1.0, 1.0)

    def _get_obs(self):
        error  = self.target_speed - self.current_speed
        de_dt  = error - self.prev_error
        return np.array([
            self._norm_error(error),
            self._norm_error(self.prev_error),
            self._norm_error(de_dt)
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.target_speed  = float(np.random.uniform(600, 1400))
        self.current_speed = float(np.random.uniform(600, 1400))
        self.step_count    = 0
        self.prev_error    = self.target_speed - self.current_speed
        self.sim_time      = 0.0
        self.prev_action   = 0.5
        return self._get_obs(), {}

    def step(self, action):
        a     = float(np.clip(action[0], 0.0, 1.0))
        f     = 20 + a * 30
        ma    = 0.4 + a * 0.6
        t_end = self.sim_time + self.step_dt

        eng.eval(f"""
t_step = (0:0.00001:{t_end:.4f})';
Vref_step = {ma:.6f} .* sin(2*pi*{f:.6f}.*t_step);
Vref_ts = timeseries(Vref_step, t_step);
assignin('base', 'Vref_ts', Vref_ts);
set_param('{MODEL}', 'StopTime', '{t_end:.4f}');
out = sim('{MODEL}');
""", nargout=0)

        speed_raw  = eng.eval("mean(out.wm_log(max(1,end-2000):end))", nargout=1)
        next_speed = float(np.clip(speed_raw, self.min_speed, self.max_speed))
        self.sim_time      = t_end
        self.current_speed = next_speed
        self.step_count   += 1

        error     = self.target_speed - self.current_speed
        abs_error = abs(error)
        prev_abs  = abs(self.prev_error)
        rng       = self.max_speed - self.min_speed

        reward = -abs_error / rng

        if abs_error < 1.0:
            reward += 2.0
        elif abs_error < 20.0:
            reward += 1.0 / max(abs_error, 0.1)
        elif abs_error < 100.0:
            reward += 0.5 / max(abs_error, 0.1)

        if abs_error < prev_abs:
            reward += 0.5
        else:
            reward -= 0.2

        action_change = abs(a - self.prev_action)
        reward -= 0.1 * action_change
        self.prev_action = a

        too_far = abs_error > 0.65 * rng
        done    = (self.step_count >= self.max_steps) or too_far
        self.prev_error = error
        return self._get_obs(), reward, done, False, {}

print('\nTesting SIL environment...')
env_test = SPIMSILEnv()
obs, _ = env_test.reset()
print(f'Target:  {env_test.target_speed:.1f} RPM')
print(f'Current: {env_test.current_speed:.1f} RPM')
obs, r, done, _, _ = env_test.step(np.array([0.5]))
print(f'After action 0.5: speed={env_test.current_speed:.1f} RPM | reward={r:.4f}')
print('Environment test passed')

print('\nStarting PPO SIL v3 training...')
print('3,000 timesteps — approximately 6-8 hours overnight')

env_train = Monitor(SPIMSILEnv())
ppo = PPO(
    policy        = 'MlpPolicy',
    env           = env_train,
    learning_rate = 3e-4,
    n_steps       = 256,
    batch_size    = 64,
    n_epochs      = 10,
    gamma         = 0.99,
    clip_range    = 0.2,
    verbose       = 1,
    seed          = 42,
    device        = 'cpu'
)

start = time.time()
ppo.learn(total_timesteps=3000)
elapsed = time.time() - start
print(f'\nTraining complete in {elapsed:.0f} seconds')
ppo.save('C:\\Users\\MaMak\\Desktop\\SPIM_SIL\\ppo_sil_v3')
print('PPO v3 saved')

print('\nEvaluating PPO SIL v3...')
test_targets = [600, 800, 1000, 1200, 1400]
errors = []

for target in test_targets:
    env_eval = SPIMSILEnv()
    obs, _ = env_eval.reset()
    env_eval.target_speed  = float(target)
    env_eval.current_speed = float(target + np.random.uniform(-50,50))
    obs = env_eval._get_obs()
    for _ in range(20):
        action, _ = ppo.predict(obs, deterministic=True)
        obs, _, done, _, _ = env_eval.step(action)
        if done: break
    error = abs(env_eval.current_speed - target)
    errors.append(error)
    print(f'Target {target:4d} RPM | Final: {env_eval.current_speed:.1f} RPM | Error: {error:.1f} RPM')

print(f'\nMean error: {np.mean(errors):.1f} RPM')
eng.quit()
print('SIL v3 complete')
