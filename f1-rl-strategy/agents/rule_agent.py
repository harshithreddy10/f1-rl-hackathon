import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from env.f1_env import F1PitEnv, TOTAL_LAPS, MAX_TYRE_LIFE

class RuleBasedAgent:
    def __init__(self):
        self.pit1_lap = 25
        self.pit2_lap = 60

    def reset(self):
        pass

    def act(self, obs, env: F1PitEnv) -> int:
        lap = env.lap

        if lap == self.pit1_lap:
            return 3   # pit lap 25 → HARD

        if lap == self.pit2_lap:
            return 1   # pit lap 60 → SOFT

        return 0       # stay out