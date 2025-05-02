# train.py
#!/usr/bin/env python3
import argparse
import os
import yaml
import wandb
from sb3_contrib.common.wandb_callback import WandbCallback
from stable_baselines3 import PPO
from envs.lidar_fovea_env import LidarFoveaEnv

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--det_cfg',          default='configs/pv_rcnn_fovea.yaml')
    p.add_argument('--ckpt',             default='checkpoints/pv_rcnn.pth')
    p.add_argument('--ppo_cfg',          default='configs/ppo_fovea.yaml')
    p.add_argument('--timesteps',  type=int, default=2000000)
    p.add_argument('--model_dir',        default='models')
    p.add_argument('--model_name',       default='ppo_fovea.zip')
    p.add_argument('--wandb_project',    default='Fovea3DPointRL')
    p.add_argument('--tensorboard_log',  default='runs')
    args = p.parse_args()

    with open(args.ppo_cfg) as f:
        ppo_cfg = yaml.safe_load(f)

    run = wandb.init(
        project=args.wandb_project,
        config=ppo_cfg,
        sync_tensorboard=True,
        save_code=True,
    )

    env = LidarFoveaEnv(det_cfg=args.det_cfg, ckpt=args.ckpt)

    model = PPO(
        'MlpPolicy', env,
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
        tensorboard_log=args.tensorboard_log
    )

    os.makedirs(args.model_dir, exist_ok=True)
    model.learn(
        total_timesteps=args.timesteps,
        callback=WandbCallback(model_save_path=args.model_dir, verbose=2)
    )
    model.save(os.path.join(args.model_dir, args.model_name))
    run.finish()

if __name__ == '__main__':
    main()