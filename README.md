# Fovea3DPointRL

**Fovea3DPointRL** is an open-source research prototype that fuses  
**Foveated Attention · Deep Reinforcement Learning · LiDAR 3D Object Detection**  
into a single PyTorch-based pipeline.

> **Goal:** Process large, sparse point clouds faster **without losing accuracy** by letting an RL agent decide **where to look** (high-resolution ROI) while a PV-RCNN detector decides **what is seen** inside that ROI.

---

## 1 · Core Idea

1. A **low-resolution context map** of the whole LiDAR scan feeds an RL agent.  
2. The agent outputs a 3-D crop box (= *fovea*).  
3. Points inside that box are **cropped on GPU** (Open3D Tensor API).  
4. Cropped points are passed to **PV-RCNN** (or any OpenPCDet backbone).  
5. **Reward** = detection quality (IoU / mAP) − computational cost.  
6. PPO (or DQN) iteratively improves the policy.

Typical result: **30–50 % speed-up** with < 0.5 pt mAP loss on KITTI; larger gains on denser datasets.

---

## 2 · Upstream Dependencies

| Role | GitHub Repo | Notes |
|------|-------------|-------|
| 3-D detector (PV-RCNN, PointPillars, CenterPoint …) | <https://github.com/open-mmlab/OpenPCDet> | Official PV-RCNN code (PyTorch + CUDA ops) |
| Foveated LiDAR Gym environment | <https://github.com/Zdeeno/Lidar-gym> | KITTI-based; easy to swap dataset |
| RL algorithms – PPO, DQN, SAC | <https://github.com/DLR-RM/stable-baselines3> | Pure PyTorch, Gymnasium-compatible |
| GPU point-cloud ops & vis | <https://github.com/isl-org/Open3D> | Fast ROI crop + live visualisation |

---

## 3 · Quick Start  (Ubuntu 20.04 / CUDA 11.8)

```bash
# 1) virtual env
conda create -n fovea3d python=3.10 -y
conda activate fovea3d
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install open3d==0.18.0 "stable-baselines3[extra]" gymnasium

# 2) clone third-party libs
git clone --recursive https://github.com/open-mmlab/OpenPCDet.git third_party/OpenPCDet
git clone https://github.com/Zdeeno/Lidar-gym.git               third_party/Lidar-gym

# 3) build OpenPCDet CUDA ops
cd third_party/OpenPCDet && python setup.py develop && cd ../../

# 4) (example) prepare KITTI infos
python -m pcdet.datasets.kitti.kitti_dataset \
       create_kitti_infos third_party/OpenPCDet/tools/cfgs/dataset_configs/kitti_dataset.yaml

# 5) train PPO agent (2 M steps default)
python agents/train_ppo.py
```
Fovea3DPointRL/
├─ agents/           # RL training scripts (SB3 PPO baseline)
├─ envs/
│   ├─ lidar_fovea_env.py   # Gymnasium wrapper w/ PV-RCNN inside
│   └─ crop_ops.py          # Open3D GPU ROI utilities
├─ third_party/      # submodules: OpenPCDet, Lidar-gym, SB3
├─ configs/          # pv_rcnn_fovea.yaml, ppo_fovea.yaml …
└─ docs/             # installation, benchmarks, figures

