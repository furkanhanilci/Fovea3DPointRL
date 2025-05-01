import yaml
import wandb
from sb3_contrib.common.wandb_callback import WandbCallback
from stable_baselines3 import PPO
from envs.lidar_fovea_env import LidarFoveaEnv

def main():
    with open('configs/ppo_fovea.yaml') as f:
        ppo_cfg = yaml.safe_load(f)

    run = wandb.init(
        project="Fovea3DPointRL",
        config=ppo_cfg,
        sync_tensorboard=True,
        save_code=True,
    )

    env = LidarFoveaEnv(
        det_cfg='configs/pv_rcnn_fovea.yaml',
        ckpt='checkpoints/pv_rcnn.pth'
    )

    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=ppo_cfg['learning_rate'],
        n_steps=ppo_cfg['n_steps'],
        batch_size=ppo_cfg['batch_size'],
        gamma=ppo_cfg['gamma'],
        gae_lambda=ppo_cfg['gae_lambda'],
        clip_range=ppo_cfg['clip_range'],
        ent_coef=ppo_cfg.get('ent_coef', 0.01),
        vf_coef=ppo_cfg.get('vf_coef', 0.5),
        max_grad_norm=ppo_cfg.get('max_grad_norm', 0.5),
        verbose=1,
        tensorboard_log="./runs"
    )

    model.learn(
        total_timesteps=2_000_000,
        callback=WandbCallback(
            model_save_path="models/",
            verbose=2,
        )
    )

    run.finish()

if __name__ == "__main__":
    main()