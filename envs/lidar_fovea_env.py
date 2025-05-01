import gymnasium as gym
import torch
import numpy as np
from lidar_gym.envs.lidar_env import LidarEnv
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from .crop_ops import crop_points_o3d


def compute_reward(pred_dicts, roi_pts, full_pts, alpha=3.0, beta=1.0):
    # Extract IoU values from predictions
    iou_vals = pred_dicts[0].get('pred_boxes_iou', [])
    max_iou = iou_vals.max().item() if len(iou_vals) else 0.0
    # Cost term = fraction of points kept
    cost = roi_pts.shape[0] / full_pts.shape[0]
    return alpha * max_iou - beta * cost


class LidarFoveaEnv(LidarEnv):
    def __init__(self, det_cfg: str, ckpt: str, cube_len: float = 20.0):
        super().__init__()
        # Load PV-RCNN model configuration and weights
        cfg_from_yaml_file(det_cfg, cfg)
        self.det = build_network(model_cfg=cfg.MODEL, num_class=cfg.NUM_CLASS)
        self.det.load_params_from_file(ckpt, logger=None, to_cpu=False)
        self.det.cuda().eval()
        # Action: [cx, cy, cz, half_side]
        self.action_space = gym.spaces.Box(
            low=np.array([0, -40, -3, 1], dtype=np.float32),
            high=np.array([70.4, 40, 1, cube_len/2], dtype=np.float32),
        )
        # Observation space inherited
        self.observation_space = self.dataset_env.observation_space
        self.full_pts = None

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.full_pts = info['points']  # store full point cloud
        return obs, info

    @torch.no_grad()
    def _inference(self, pts: np.ndarray):
        input_dict = {'points': torch.tensor(pts, device='cuda:0')}
        load_data_to_gpu(input_dict)
        pred_dicts, _ = self.det.forward(input_dict)
        return pred_dicts

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        center = action[:3]
        half = float(action[3])
        # Crop ROI
        roi_pts = crop_points_o3d(info['points'], center, half)
        # Detect
        pred_dicts = self._inference(roi_pts)
        # Compute reward
        reward = compute_reward(pred_dicts, roi_pts, self.full_pts)
        return obs, reward, terminated, truncated, info