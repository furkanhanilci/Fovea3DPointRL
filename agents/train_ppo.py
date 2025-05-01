import yaml
from stable_baselines3 import PPO
from envs.lidar_fovea_env import LidarFoveaEnv


def main():
    # Load PPO hyperparameters
    with open('configs/ppo_fovea.yaml') as f:
        ppo_cfg = yaml.safe_load(f)
    # Create environment
    env = LidarFoveaEnv(
        det_cfg='configs/pv_rcnn_fovea.yaml',
        ckpt='checkpoints/pv_rcnn.pth'
    )
    # Instantiate PPO agent
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
        tensorboard_log='./runs'
    )
    # Train agent
    model.learn(total_timesteps=2_000_000)
    # Save model
    model.save('ppo_fovea.zip')


if __name__ == '__main__':
    main()