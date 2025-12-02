# 🧬 Protein-Ligand Binding Site VQ-VAE

本项目实现了一个**端到端的蛋白-配体结合位点离散表示学习流水线**，基于 GCPNet 图神经网络和 VQ-VAE 架构。

## 项目简介

从 3432 个 PDB 复合物出发，通过识别蛋白-配体结合位点（≤4.0Å），构建三张图（蛋白图、配体图、相互作用图），使用完整版 GCPNet 提取 641 维边级融合特征，最终通过 VQ-VAE 学习 4096 个离散码本（binding codes）。

**核心特性**：完整版 GCPNet 编码器（6层）、三图联合建模、端到端训练、混合精度、HDF5 数据流。

## 快速开始

1. **数据准备**：运行 `feature extraction/full_pipeline.py` 生成 HDF5 数据
2. **模型训练**：打开 `end_to_end_vqvae_training.ipynb` 进行端到端训练
3. **推理使用**：使用 `inference_encode.py` 或 `inference_embed.py` 进行推理

---

## 📁 项目结构

```
notebook-lab/
├── vqvae.py                               # VQ-VAE 模型实现
├── inference_encode.py                    # 推理：单独编码三张图
├── inference_embed.py                     # 推理：批量生成 embeddings
├── test_vqvae_training.py                 # 单元测试脚本
│
├── end_to_end_vqvae_training.ipynb        # ⭐ 端到端训练主流程
│
├── feature extraction/
│   ├── full_pipeline.py                   # ⭐ 完整流水线 (PDB→HDF5)
│   ├── debug_pipeline.py                  # 调试版流水线
│   └── pdb_complex_analysis.py            # PDB 分析工具
│
├── config_gcpnet_encoder.yaml             # GCPNet 编码器配置
├── config_vqvae.yaml                      # VQ-VAE 训练配置
│
├── gcpnet/                                # GCPNet 模块
│   ├── features/                          # 特征提取器
│   ├── models/graph_encoders/             # 图编码器
│   ├── geometry.py                        # 3D 几何变换
│   └── heads.py                           # 输出头
│
├── complex-20251129T063258Z-1-001/        # 3432 个 PDB 文件
│
├── improtant data/                        # HDF5 输出目录
│   ├── binding_sites.h5                   # 接触信息
│   ├── binding_embeddings_*.h5            # 三图 embeddings
│   ├── binding_edge_features.h5           # 边级局部特征
│   └── binding_edge_features_fused.h5     # ⭐ 最终融合特征 (641维)
│
├── checkpoints/vqvae_end_to_end/          # 训练 checkpoint
│
├── data_analyzer/                         # PDB 分析工具
├── gcpnet_README.md                       # GCPNet 详细文档
└── README.md                              # 本文件
```

## 环境依赖

```bash
pip install torch torch-geometric biopython pandas numpy omegaconf pyyaml h5py
pip install x-transformers vector-quantize-pytorch ndlinear
```

## 数据统计

- **PDB 文件**: 3,432 个
- **接触记录**: 25,626 条 (≤4.0Å)
- **样本数**: 3,139 个
- **边数**: 13,798 条
- **特征维度**: 641 维
- **VQ 码本**: 4,096 codes

