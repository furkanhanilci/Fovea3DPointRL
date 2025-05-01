import argparse
from stable_baselines3 import PPO
from envs.lidar_fovea_env import LidarFoveaEnv
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--det_cfg', type=str, default='configs/pv_rcnn_fovea.yaml')
    parser.add_argument('--ckpt', type=str, default='checkpoints/pv_rcnn.pth')
    args = parser.parse_args()

    env = LidarFoveaEnv(det_cfg=args.det_cfg, ckpt=args.ckpt)
    model = PPO.load(args.model_path)

    obs, info = env.reset()
    done = False
    start = time.time()
    frame_count = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        frame_count += 1
    elapsed = time.time() - start
    fps = frame_count / elapsed
    print(f"Evaluation done: {frame_count} frames in {elapsed:.2f}s ({fps:.2f} FPS)")