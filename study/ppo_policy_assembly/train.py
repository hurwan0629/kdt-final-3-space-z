"""프로젝트 루트에서 실행: python study/ppo_policy_assembly/train.py"""
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import TargetEnv
from policy import SmallHybridPolicy


def learning_rate(progress_remaining):
    """SB3가 호출하는 schedule: 1(학습 시작) -> 0(학습 종료)."""
    return 3e-4 * (0.25 + 0.75 * progress_remaining)


class ProgressCallback(BaseCallback):
    def _on_step(self):
        if self.num_timesteps % 512 == 0:
            lr = self.model.policy.optimizer.param_groups[0]["lr"]
            print(f"[step {self.num_timesteps:4d}] optimizer lr={lr:.7f}")
        return True


def main():
    # 각 람다가 별도의 Gymnasium Env와 Monitor를 만든다. (서브프로세스 아님)
    env = DummyVecEnv([lambda: Monitor(TargetEnv()) for _ in range(2)])
    try:
        model = PPO(
            SmallHybridPolicy,   # Policy 클래스 전달; SB3가 인스턴스를 생성
            env,
            learning_rate=learning_rate,  # 숫자 3e-4로 바꾸면 고정 LR
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            gamma=0.98,
            gae_lambda=0.95,
            seed=42,
            device="cpu",
            verbose=0,
        )

        print("policy type:", type(model.policy).__name__)
        print("action_net:", model.policy.action_net)
        print("optimizer:", type(model.policy.optimizer).__name__)
        print("initial log_std:", model.policy.log_std.detach().tolist())

        obs = env.reset()  # (2, 4)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=model.device)
            actions, values, log_probs = model.policy(obs_tensor)
        print("obs:", tuple(obs_tensor.shape))
        print("actions:", tuple(actions.shape), actions.tolist())
        print("values:", tuple(values.shape))
        print("log_probs:", tuple(log_probs.shape))

        # 학습: forward로 수집 -> evaluate_actions로 재평가 -> backward/update
        model.learn(total_timesteps=2048, callback=ProgressCallback())

        print("trained log_std:", model.policy.log_std.detach().tolist())
        print("final optimizer lr:", model.policy.optimizer.param_groups[0]["lr"])
        obs = env.reset()
        actions, _ = model.predict(obs, deterministic=True)
        print("predict deterministic:", actions.tolist())
    finally:
        env.close()


if __name__ == "__main__":
    main()
