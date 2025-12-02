## notebook-lab：VQ-VAE + GCPNet 端到端流水线

本仓库包含一个基于 **GCPNet 图编码器 + Transformer VQ-VAE** 的端到端流水线，用于对蛋白–配体结合位点的局部环境进行离散化表示学习（binding codes）。

下面的 README 总结了主要脚本、数据流程和训练方式，便于快速上手和回顾。

---

## 1. 总体架构概览

- **目标**：
  - 对蛋白–配体结合位点的局部环境进行离散化表示学习
  - 学到可用于分析和下游任务的离散 code（codebook indices）

- **输入**：
  - PDB 复合物（蛋白 + 配体），位于 `complex-20251129T063258Z-1-001/complex/*.pdb`

- **中间表示**：
  - 使用 GCPNet 对三张图进行编码：
    - 蛋白图（Protein graph）
    - 配体图（Ligand graph）
    - 相互作用图（Protein–Ligand interaction graph）
  - 同时提取局部边级几何特征（距离、方向、alpha/kappa/dihedral 等）

- **输出**：
  - 一个高维边级融合特征矩阵（例如 641 维）
  - VQ-VAE 学到的离散 codebook 以及重建特征（用于评估与下游任务）

- **实现拆分为两大部分**：
  - 特征提取（离线脚本）：`feature extraction/full_pipeline.py`
  - 端到端训练（Notebook）：`end_to_end_vqvae_training.ipynb`

- **核心模型文件**：
  - `vqvae.py`
    - 定义 `VQVAETransformer`，负责 Transformer 编码 + Vector Quantization
    - 支持 TikTok 压缩、Residual VQ、多层 codebook、正交正则等
  - `gcpnet/`
    - 包含 GCPNet 图编码器、几何特征提取、各类层与 head
    - 用于蛋白/配体/相互作用三种图的表示学习

---

## 2. 特征提取流水线（`feature extraction/full_pipeline.py`）

### 2.1 功能与输入输出

**功能总览**：

- 从 PDB 复合物中识别蛋白–配体接触残基（binding sites）
- 基于接触残基构建三张图：
  - 蛋白图（Protein graph）
  - 配体图（Ligand graph）
  - 相互作用图（Protein–Ligand interaction graph）
- 使用完整版 GCPNet 编码器获取三张图的 embedding
- 提取局部边级几何特征（含 alpha/kappa/二面角等）
- 将 embedding 与局部边级特征进行融合，生成最终 **边级融合特征矩阵**
- 所有结果以 HDF5 格式保存到 `improtant data/` 目录

**输入目录**：

- PDB 复合物：`complex-20251129T063258Z-1-001/complex/*.pdb`

**主要输出 HDF5 文件**（均位于 `improtant data/`）：

- `binding_sites.h5`：
  - 蛋白–配体接触残基信息（PDB id、链、残基号、距离等）
- `binding_embeddings_protein.h5`：
  - 蛋白图级 embedding（每个样本一个向量）
- `binding_embeddings_ligand.h5`：
  - 配体图级 embedding
- `binding_embeddings_interaction.h5`：
  - 相互作用图级 embedding
- `binding_edge_features.h5`：
  - 边级局部几何特征（不含图级 embedding）
- `binding_edge_features_fused.h5`：
  - 最终用于 VQ-VAE 训练的融合边特征
  - `features`：形状约为 `(N_edges, 641)`，由以下部分拼接：
    - 纯边级局部特征
    - 蛋白图 embedding
    - 配体图 embedding
    - 相互作用图 embedding
  - `graph_index`：指示每条边属于哪一个样本/图
  - 其他元信息：`pdb_id`, `ligand_resname` 等

### 2.2 PDB 复合物分析（接触识别）

- 入口函数：`analyze_all_pdbs(pdb_dir: Path) -> pd.DataFrame`
- 核心步骤：
  - `split_protein_and_ligands(structure)`：
    - 将结构中残基划分为蛋白残基和小分子配体残基
    - 过滤掉水、金属离子等不关心的 HETATM
  - `compute_contacts_for_structure(pdb_path: Path)`：
    - 枚举所有蛋白残基–配体残基对，计算最小原子–原子距离
    - 若小于 `DIST_CUTOFF = 4.0 Å` 则记为一条接触记录
  - 接触信息通过 `save_binding_sites_to_h5(df, BINDING_SITES_H5)` 写入 HDF5

### 2.3 三图构建与 GCPNet 编码

**蛋白图构建与编码**：

- `build_pyg_data_for_group(...)`：
  - 根据 binding site 分组，从 PDB 构建蛋白图
  - 节点：氨基酸/原子
  - 边：KNN 或距离约束
- 使用 `ProteinFeaturiser`（`gcpnet.features.factory`）：
  - 生成标量特征（氨基酸类型、B-factor、二级结构等）
  - 生成向量特征（坐标、方向、几何量等）
- `encode_protein_graph(batch)`：
  - 调用 `GCPNetModel`，返回：
    - `node_embedding`
    - `graph_embedding`（图级向量，用于融合到边特征）

**配体图构建与编码**：

- `build_ligand_graph_from_pdb(...)`：
  - 从 PDB 中抽取配体残基，构建小分子图（节点为原子，边为化学键/距离邻居）
- `encode_ligand_graph(ligand_data_list)`：
  - 使用 GCPNet 架构（替代原先的 Simple MLP）
  - 返回每个样本的图级 embedding

**相互作用图构建与编码**：

- `build_interaction_graph(protein_data, ligand_data)`：
  - 基于空间邻近，在蛋白原子与配体原子之间建立“相互作用边”
  - 节点带有角色编码（蛋白/配体）
- `encode_interaction_graph(inter_data_list)` / `encode_interaction_graph_nodes(...)`：
  - 图级 embedding：用于全局语义
  - 节点级 embedding：可用于后续边级特征拼接

### 2.4 边级局部特征与融合

- 边级局部特征提取：`compute_and_save_edge_features(...)`
  - 对每一条“蛋白–配体接触边”构建局部几何特征：
    - 距离、方向向量
    - 局部坐标系相关量：alpha / kappa / dihedral angles 等
    - 氨基酸类型、原子类型 one-hot 或嵌入
- 特征融合与最终 HDF5 输出：`fuse_and_save_edge_features(...)`（名称以实际代码为准）
  - 对每条边，将以下部分进行串联（cat）：
    - 纯局部几何特征向量
    - 对应样本的蛋白图 embedding
    - 对应样本的配体图 embedding
    - 对应样本的相互作用图 embedding
  - 保存为 `binding_edge_features_fused.h5`，并在 attrs 中记录：
    - `feature_dim`
    - `protein_emb_dim`
    - `ligand_emb_dim`
    - `interaction_emb_dim`
    - `edge_feature_dim`

---

## 3. 端到端 VQ-VAE 训练（`end_to_end_vqvae_training.ipynb`）

### 3.1 数据与环境配置

- 基准路径：
  - `BASE_DIR = c:/Users/Administrator/Desktop/IGEM/stage1/notebook-lab`
- 输入数据：
  - `H5_DATA_PATH = BASE_DIR / 'improtant data' / 'binding_edge_features_fused.h5'`
- Checkpoint 目录：
  - `CHECKPOINT_DIR = BASE_DIR / 'checkpoints' / 'vqvae_end_to_end'`
- 配置文件：
  - `config_vqvae.yaml`：Transformer + VQ-VAE 配置
  - `config_gcpnet_encoder.yaml`：GCPNet 模型与特征提取配置
- Notebook 中提供：
  - HDF5 数据存在性检查
  - 自动运行 `feature extraction/full_pipeline.py` 生成数据的单元格
  - 如果 `binding_edge_features_fused.h5` 不存在，将通过 `importlib` 加载并执行 `full_pipeline.main()`

### 3.2 数据集定义：`EdgeFeatureDataset`

- 基于 HDF5 构造 PyTorch `Dataset`：
  - 加载：
    - `features`：`(N_edges, feature_dim)`
    - `graph_index`：`(N_edges,)`
    - `pdb_id`, `ligand_resname` 等
  - 每个 `__getitem__` 返回一个“样本图”的边特征矩阵：
    - 固定长度 `max_edges_per_sample = 512`
    - 超过则截断，不足则用 0 padding，并提供 `mask`
  - 输出：
    - `padded_features`：`(max_edges, feature_dim)`
    - `mask`：`(max_edges,)`，1 为有效边，0 为 padding
- 使用 `DataLoader` 封装为 batch：
  - `edge_feats`：`(B, L, feature_dim)`，目前 `feature_dim = 641`
  - `mask`：`(B, L)`

### 3.3 模型结构与配置

Notebook 内组装了一个端到端结构：

- **GCPNet Encoder + Featuriser**（来自 `gcpnet` 包）
  - 在端到端训练模式下，GCPNet 参数默认是可训练的，梯度可以回传

- **Feature Projector**
  - 将 641 维边级特征投影到 VQ-VAE 输入维度（通常 128 维）：
  - 形状变化：`(B, L, 641) → (B, L, 128)`

- **VQ-VAE 模型（`VQVAETransformer`，见 `vqvae.py`）**
  - Encoder：NdLinear 或 Conv1d + Transformer
  - Vector Quantizer：
    - 普通 VQ (`VectorQuantize`)
    - Residual VQ (`ResidualVQ`) + TikTok token 压缩
    - 正交正则化（Orthogonal Regularization）
  - Decoder：由外部传入的解码器，将量化后的 code 重建回原始特征空间
  - 输入输出接口（简化）：
    - 输入：`(B, L, D_in)` + `mask`
    - 输出：`(decoder_output, indices, vq_loss, codebook_usage_info, ...)`

- **损失函数**
  - 重建损失（reconstruction loss）：
    - 在原始边级特征空间计算 MSE：`MSE(decoder_output, edge_feats)`（对有效 mask 位置）
  - VQ 损失：
    - commitment loss + codebook 更新等
  - 总损失：
    - `loss_total = loss_recon + 0.1 * loss_vq`（权重可在 `compute_total_loss` 中调节）

- **优化与调度器**
  - 对 GCPNet + Featuriser + FeatureProjector + VQ-VAE 使用分组学习率
  - 使用 `GradScaler + autocast()` 实现混合精度训练
  - 使用 `scheduler.step(...)` 实现学习率调度（如 cosine decay）

### 3.4 训练流程概览

1. **单 batch 维度检查**：
   - 在正式训练前，取一个 batch 做前向传播：
   - 检查 `decoder_output.shape` 是否与 `edge_feats.shape` 一致
   - 若不一致则抛出 `RuntimeError('Decoder output dimension mismatch')`

2. **正式训练循环**：
   - `for epoch in 1..NUM_EPOCHS`：
     - 对每个 batch：
       - `edge_feats, mask → GPU`
       - `projected_feats = feature_projector(edge_feats)`
       - `outputs = vqvae_model(projected_feats, mask, nan_mask)`
       - `total_loss, loss_dict = compute_total_loss(outputs, edge_feats, mask, vq_weight=0.1)`
       - 反向传播 + `clip_grad_norm_` + `optimizer.step()`
     - 收集所有 batch 的 VQ code indices，计算 codebook 使用率：
       - `unique_codes = torch.unique(indices)`
       - `codebook_usage = len(unique_codes) / codebook_size`
     - 将 `total_loss / recon_loss / vq_loss / codebook_usage` 记录到 `train_history`

3. **Checkpoint 保存**：
   - 每 `SAVE_INTERVAL` 个 epoch（或最后一个 epoch）保存：
     - `gcpnet_encoder.state_dict()`
     - `featuriser.state_dict()`
     - `feature_projector.state_dict()`
     - `vqvae_model.state_dict()`
     - `optimizer`, `scheduler` 状态
     - `train_history`
     - `config`：将 VQ-VAE 与 GCPNet 的配置 dict 一并存入



---

## 4. 项目文件索引

```text
notebook-lab/
├── README.md                              # 本文件
├── .gitignore                             # Git 忽略文件配置
├── gcpnet_README.md                       # GCPNet 模块详细说明
├── config_gcpnet_encoder.yaml             # GCPNet 编码器配置文件
├── config_vqvae.yaml                      # VQ-VAE 训练配置文件
├── vqvae.py                               # VQ-VAE 模型实现
│
├── complex-20251129T063258Z-1-001/        # 数据集（3432 个 PDB 文件）
│   └── complex/
│       ├── 0.pdb
│       ├── 1.pdb
│       └── ... (3432 PDB files)
│
├── feature extraction/                    # 特征提取脚本目录
│   ├── full_pipeline.py                   # ⭐ 完整流水线（PDB 分析 → 边级特征融合）
│   └── pdb_complex_analysis.py            # PDB 复合物分析脚本
│
├── 📓 核心 Notebooks
│   ├── binding_edge_codebook.ipynb        # ⭐⭐⭐ 主流程：Edge 码本 + 完整 VQ-VAE 训练
│   └── PDB_complex_analysis.ipynb         # 结合位点识别
│
├── 📊 输出数据
│   ├── binding_sites.csv                  # 蛋白-配体接触记录（25,626 条）
│   ├── binding_embeddings.csv             # 简单特征嵌入（23 维）
│   ├── binding_embeddings_protein.csv     # 蛋白图 embedding（3,139 样本）
│   ├── binding_embeddings_ligand.csv      # 配体图 embedding
│   ├── binding_embeddings_interaction.csv # 相互作用图 embedding
│   ├── binding_edge_codes.csv             # Edge 离散码本索引
│   └── improtant data/                    # HDF5 数据目录（需运行 full_pipeline.py 生成）
│       ├── binding_sites.h5               # 接触信息（HDF5）
│       ├── binding_embeddings_protein.h5  # 蛋白 embedding（HDF5）
│       ├── binding_embeddings_ligand.h5   # 配体 embedding（HDF5）
│       ├── binding_embeddings_interaction.h5  # 相互作用 embedding（HDF5）
│       ├── binding_edge_features.h5       # 边级局部特征（HDF5）
│       ├── binding_edge_features_fused.h5 # ⭐ 最终融合特征（用于 VQ-VAE）
│       └── binding_edge_features_fused.csv # 融合特征（CSV 版本，13,798 条边）
│
├── checkpoints/                           # 训练 checkpoint 目录（训练后生成）
│   └── vqvae_edge_features/
│       ├── epoch_5.pth
│       ├── epoch_10.pth
│       └── ...
│
├── gcpnet/                                # GCPNet 模块（特征提取 + 图编码器）
│   ├── features/
│   │   ├── factory.py                     # ProteinFeaturiser（主要接口）
│   │   ├── node_features.py               # 节点特征（氨基酸 one-hot、主链角度等）
│   │   ├── edge_features.py               # 边特征（距离、向量）
│   │   ├── representation.py              # 坐标表示（CA/CA+CB 等）
│   │   └── ...
│   ├── models/
│   │   ├── base.py                        # 预训练模型加载接口
│   │   ├── graph_encoders/
│   │   │   ├── gcpnet.py                  # GCPNet 主编码器
│   │   │   ├── components/                # 网络组件（径向基、消息传递层等）
│   │   │   └── layers/
│   │   └── utils.py
│   ├── geometry.py                        # 3D 几何变换（刚体、旋转矩阵）
│   ├── heads.py                           # 输出头（回归、分类）
│   └── ...
│
├── data_analyzer/                         # PDB 分析工具
│   ├── pdb_complex_analyzer.py            # 批量分析脚本
│   ├── PDB复合物分析指南.md
│   └── README_PDB分析.md
│
├── inference_encode.py                    # 推理脚本 1：单独编码蛋白/配体/相互作用图
└── inference_embed.py                     # 推理脚本 2：批量生成 binding embeddings