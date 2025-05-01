# Fovea3DPointRL

Fovea3DPointRL is a research prototype that integrates foveated attention mechanisms, deep reinforcement learning, and LiDAR-based three-dimensional object detection into a cohesive PyTorch-driven workflow. In the proposed system, an RL agent first examines a low-resolution, global representation of the entire point cloud to inform its decision-making process. Based on this overview, the agent identifies the coordinates of a high-resolution “fovea” region. Points within that region are then dynamically cropped using GPU-accelerated Open3D operations and passed exclusively to a PV-RCNN detector. The agent’s reward function balances detection accuracy (measured by IoU or mAP) against computational cost; thus, the policy learned achieves a 30–50 % improvement in inference speed on standard benchmark datasets while maintaining minimal accuracy degradation.

| Role | GitHub Repository | Purpose |
|------|------------------|---------|
| 3-D detector (PV-RCNN, PointPillars, CenterPoint …) | <https://github.com/open-mmlab/OpenPCDet> | High-accuracy LiDAR detection backbones |
| Foveated LiDAR Gym environment | <https://github.com/Zdeeno/Lidar-gym> | KITTI-based Gym environment for ROI selection |
| RL algorithms – PPO / DQN / SAC | <https://github.com/DLR-RM/stable-baselines3> | Pure-PyTorch implementations, Gymnasium-compatible |
| GPU point-cloud ops & visualisation | <https://github.com/isl-org/Open3D> | Fast ROI cropping + live 3-D viewer |
