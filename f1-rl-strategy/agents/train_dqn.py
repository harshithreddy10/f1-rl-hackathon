import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np

from env.f1_env import F1PitEnv

MODEL_PATH = "models/dqn_f1_agent"
LOG_DIR    = "models/dqn_logs/"
os.makedirs(LOG_DIR, exist_ok=True)


class ProgressCallback(BaseCallback):
    def __init__(self, print_every=10000):
        super().__init__()
        self.print_every = print_every

    def _on_step(self) -> bool:
        if self.n_calls % self.print_every == 0:
            print(f"  Step {self.n_calls:>7d} | episodes={self.n_calls // 78:>4d}")
        return True


def make_env():
    return Monitor(F1PitEnv(weather="dry"))


def train(total_timesteps=500_000):
    print(f"🚀 Training DQN for {total_timesteps:,} steps...")
    print("   ETA: ~45-60 mins on CPU\n")

    env = make_vec_env(make_env, n_envs=1)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        verbose=0,
    )

    eval_env = Monitor(F1PitEnv(weather="dry"))
    eval_cb  = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_PATH + "_best",
        log_path=LOG_DIR,
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[ProgressCallback(), eval_cb],
        progress_bar=True,
    )

    model.save(MODEL_PATH)
    print(f"\n✅ Model saved → {MODEL_PATH}.zip")
    return model


def evaluate(model_path=MODEL_PATH + "_best/best_model", weather="dry", n_episodes=10):
    model = DQN.load(model_path)
    env   = F1PitEnv(weather=weather)
    times = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(int(action))
            if term or trunc:
                break
        times.append(env.total_time)
        print(f"  Episode {ep+1}: {env.total_time:.1f}s | pits={env.pit_laps} | compounds={env.compounds_used}")

    avg = np.mean(times)
    print(f"\n📊 Average race time: {avg:.1f}s")
    print(f"   Baseline (2-stop):  6226.6s")
    print(f"   Time saved:         {6226.6 - avg:+.1f}s")
    return avg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--steps", type=int, default=200_000)
    args = parser.parse_args()

    if args.eval_only:
        evaluate()
    else:
        train(args.steps)
        print("\n📊 Evaluating trained agent...")
        evaluate()