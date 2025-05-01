# Fovea3DPointRL

Fovea3DPointRL is a research prototype that brings together foveated attention, deep reinforcement learning, and LiDAR-based 3D object detection in a single PyTorch pipeline. In this system, an RL agent first examines a low-resolution overview of the entire point cloud and decides a high-resolution “fovea” region to crop. Only the points inside that region are sent, via a GPU-accelerated Open3D crop, to a PV-RCNN detector which then identifies objects. The agent receives a reward balancing detection accuracy (measured by IoU or mAP) against computational cost, allowing it to learn policies that yield 30–50 % faster inference with minimal drop in detection performance on standard benchmarks.

## Repository Structure

Fovea3DPointRL/
├── agents/                 # RL training scripts (Stable-Baselines3 PPO baseline)
├── envs/
│   ├── lidar_fovea_env.py  # Gymnasium wrapper embedding PV-RCNN
│   └── crop_ops.py         # Open3D GPU ROI utilities
├── third_party/            # Git submodules: OpenPCDet, Lidar-gym, SB3
├── configs/                # pv_rcnn_fovea.yaml, ppo_fovea.yaml, etc.
└── docs/                   # Installation notes, dataset prep, benchmarks, figures

