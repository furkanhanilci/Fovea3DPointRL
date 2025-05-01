# Fovea3DPointRL

Fovea3DPointRL is a research prototype that brings together foveated attention, deep reinforcement learning, and LiDAR-based 3D object detection in a single PyTorch pipeline. In this system, an RL agent first examines a low-resolution overview of the entire point cloud and decides a high-resolution “fovea” region to crop. Only the points inside that region are sent, via a GPU-accelerated Open3D crop, to a PV-RCNN detector which then identifies objects. The agent receives a reward balancing detection accuracy (measured by IoU or mAP) against computational cost, allowing it to learn policies that yield 30–50 % faster inference with minimal drop in detection performance on standard benchmarks.

| Role | GitHub Repository | Purpose |
|------|------------------|---------|
| 3-D detector (PV-RCNN, PointPillars, CenterPoint …) | <https://github.com/open-mmlab/OpenPCDet> | High-accuracy LiDAR detection backbones |
| Foveated LiDAR Gym environment | <https://github.com/Zdeeno/Lidar-gym> | KITTI-based Gym environment for ROI selection |
| RL algorithms – PPO / DQN / SAC | <https://github.com/DLR-RM/stable-baselines3> | Pure-PyTorch implementations, Gymnasium-compatible |
| GPU point-cloud ops & visualisation | <https://github.com/isl-org/Open3D> | Fast ROI cropping + live 3-D viewer |
