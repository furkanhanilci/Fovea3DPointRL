# evaluate.py
#!/usr/bin/env python3
import argparse
import time
import wandb
from stable_baselines3 import PPO
from envs.lidar_fovea_env import LidarFoveaEnv

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',    required=True)
    p.add_argument('--det_cfg',       default='configs/pv_rcnn_fovea.yaml')
    p.add_argument('--ckpt',          default='checkpoints/pv_rcnn.pth')
    p.add_argument('--wandb_project', default='Fovea3DPointRL')
    args = p.parse_args()

    run = wandb.init(project=f"{args.wandb_project}-eval", job_type="evaluation")

    env   = LidarFoveaEnv(det_cfg=args.det_cfg, ckpt=args.ckpt)
    model = PPO.load(args.model_path)

    obs, info = env.reset()
    done      = False
    start     = time.time()
    frames    = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        frames += 1

    elapsed = time.time() - start
    fps     = frames / elapsed

    run.log({"eval_fps": fps, "eval_frames": frames})
    print(f"{frames} frames in {elapsed:.2f}s → {fps:.2f} FPS")

    run.finish()

if __name__ == '__main__':
    main()
