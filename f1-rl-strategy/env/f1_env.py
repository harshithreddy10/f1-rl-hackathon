import gymnasium as gym
import numpy as np
import pickle
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.tyre_model import load_model, predict_lap_time

COMPOUNDS     = ["SOFT", "MEDIUM", "HARD"]
TOTAL_LAPS    = 78
PIT_TIME_LOSS = 22.0
PIT_COOLDOWN  = 3
MAX_TYRE_LIFE = {
    "SOFT":   25,
    "MEDIUM": 35,
    "HARD":   50,
}

class F1PitEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, weather="dry", render_mode=None):
        super().__init__()
        self.weather = weather
        self.render_mode = render_mode
        self.model, self.le = load_model()

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)
        self._reset_state()

    def _reset_state(self):
        self.lap            = 1
        self.compound       = "MEDIUM"
        self.tyre_life      = 1
        self.fuel_load      = 100.0
        self.total_time     = 0.0
        self.pit_count      = 0
        self.pit_cooldown   = 0
        self.lap_times      = []
        self.pit_laps       = []
        self.compounds_used = ["MEDIUM"]

    def _get_obs(self):
        weather_val = 1.0 if self.weather == "wet" else 0.0
        comp_idx    = COMPOUNDS.index(self.compound) / (len(COMPOUNDS) - 1)
        return np.array([
            self.lap / TOTAL_LAPS,
            min(self.tyre_life, 50) / 50.0,
            comp_idx,
            self.fuel_load / 100.0,
            35.0 / 60.0,
            weather_val,
            0.0,
            0.0,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"

        pit_this_lap  = False
        time_this_lap = 0.0

        if action == 0 or self.pit_cooldown > 0:
            pass
        else:
            new_compound      = COMPOUNDS[action - 1]
            pit_this_lap      = True
            time_this_lap    += PIT_TIME_LOSS
            # penalise pitting too early (before lap 15)
            if self.lap < 15:
                time_this_lap += (15 - self.lap) * 2.0
            # penalise pitting same compound
            if new_compound == self.compound:
                time_this_lap += 30.0
            self.compound     = new_compound
            self.tyre_life    = 1
            self.pit_count   += 1
            self.pit_cooldown = PIT_COOLDOWN
            self.pit_laps.append(self.lap)
            if new_compound not in self.compounds_used:
                self.compounds_used.append(new_compound)

        if self.pit_cooldown > 0:
            self.pit_cooldown -= 1

        lap_time = predict_lap_time(
            self.model, self.le,
            self.compound, self.tyre_life, self.lap, pit_this_lap
        )

        if self.weather == "wet":
            lap_time *= 1.15

        limit = MAX_TYRE_LIFE[self.compound]
        if self.tyre_life > limit:
            lap_time += (self.tyre_life - limit) * 0.8

        time_this_lap   += lap_time
        self.total_time += time_this_lap
        self.fuel_load  -= 100.0 / TOTAL_LAPS
        self.tyre_life  += 1
        self.lap_times.append(time_this_lap)
        self.lap        += 1

        terminated = self.lap > TOTAL_LAPS
        truncated  = False

        reward = -time_this_lap / 78.0

        if terminated:
            if self.pit_count == 0:
                reward -= 500.0
            elif len(set(self.compounds_used)) < 2:
                reward -= 300.0
            elif self.pit_count > 3:
                reward -= float(self.pit_count) * 10.0
            else:
                reward += 100.0

        if self.render_mode == "human":
            self._render()

        return self._get_obs(), reward, terminated, truncated, {
            "lap":        self.lap - 1,
            "lap_time":   time_this_lap,
            "total_time": self.total_time,
            "compound":   self.compound,
            "tyre_life":  self.tyre_life,
            "pit_count":  self.pit_count,
        }

    def _render(self):
        print(
            f"Lap {self.lap-1:2d}/{TOTAL_LAPS} | "
            f"{self.compound:6s} age={self.tyre_life-1:2d} | "
            f"Lap: {self.lap_times[-1]:.2f}s | "
            f"Total: {self.total_time:.1f}s"
        )

if __name__ == "__main__":
    env = F1PitEnv(weather="dry", render_mode="human")
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print()

    for lap in range(TOTAL_LAPS):
        if lap == 25:
            action = 3   # pit lap 25 → HARD
        elif lap == 60:
            action = 1   # pit lap 60 → SOFT
        else:
            action = 0   # stay out

        obs, reward, term, trunc, info = env.step(action)
        if term:
            break

    print(f"\n🏁 Total race time: {env.total_time:.1f}s")
    print(f"   Pit laps: {env.pit_laps}")
    print(f"   Compounds: {env.compounds_used}")