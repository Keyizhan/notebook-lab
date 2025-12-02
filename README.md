# Protein-Ligand Binding Site Analysis & VQ-VAE Training Pipeline

本项目提供从 PDB 复合物分析到 VQ-VAE 离散码本训练的完整流水线，包括：

1. 批量分析蛋白-配体复合物结构，识别关键结合位点
2. 使用 GCPNet 图神经网络提取三图（蛋白口袋图、配体图、相互作用图）的几何嵌入特征
3. 构建边级融合特征并训练 VQ-VAE 离散码本

---

## 📋 项目概览

### 主要功能

1. **PDB 复合物批量解析**：自动识别蛋白-配体接触界面的关键残基
2. **三图构建与特征提取**：
   - 蛋白 binding 残基图（Cα 节点 + KNN 图）
   - 配体原子图（原子节点 + KNN 图）
   - 蛋白-配体相互作用图（跨模态边）
3. **GCPNet 编码**：使用预配置的 GCPNet 模型对三类图分别编码并拼接，生成高维嵌入向量
4. **边级特征融合**：提取相互作用图的边级局部特征并融合三图 embedding
5. **VQ-VAE 离散码本训练**：
   - Edge 级几何码本（简化版）
   - 完整 VQ-VAE 训练（Transformer + Vector Quantizer + Geometric Decoder）

### 适用场景

- 蛋白-配体结合位点识别
- 药物-靶标相互作用预测
- 虚拟筛选与药物设计
- 结构生物学数据挖掘
- 蛋白质结构生成与压缩
- 离散表示学习

---

## 📁 目录结构

```
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

```

---

## 🗂️ 数据说明

### 输入数据

#### 1. PDB 复合物数据集

- **路径**：`complex-20251129T063258Z-1-001/complex/`
- **数量**：3432 个 PDB 文件
- **内容**：蛋白-配体复合物结构（包含 `ATOM` 和 `HETATM` 记录）
- **命名**：按整数编号（0.pdb, 1.pdb, ..., 3431.pdb）

### 输出数据

#### 1. `binding_sites.csv`（25,626 条记录）

由 `PDB_complex_analysis.ipynb` 或 `full_pipeline.py` 生成，记录所有蛋白残基与配体的空间接触关系。

| 列名              | 说明                  | 示例    |
| ----------------- | --------------------- | ------- |
| `pdb_id`          | PDB 文件编号          | `0`     |
| `protein_chain`   | 蛋白链 ID             | `A`     |
| `protein_resnum`  | 残基序号              | `7`     |
| `protein_icode`   | 插入码                | ` `     |
| `protein_resname` | 残基名称              | `VAL`   |
| `ligand_resname`  | 配体名称              | `FAD`   |
| `ligand_chain`    | 配体链 ID             | `B`     |
| `ligand_resnum`   | 配体残基号            | `1`     |
| `ligand_icode`    | 配体插入码            | ` `     |
| `min_distance`    | 最小原子-原子距离 (Å) | `3.055` |

**用途**：

- 标记关键结合位点（distance ≤ 4.0 Å）
- 构建蛋白 binding 残基子图的节点掩码

#### 2. 边级融合特征（257 维）

由 `full_pipeline.py` 生成，包含边级局部特征和三图 embedding 的融合。

**特征维度拆分**：

- `feat_0` ~ `feat_127`：源节点（蛋白残基）的 GCPNet 嵌入（128 维）
- `feat_128` ~ `feat_255`：目标节点（配体原子）的 GCPNet 嵌入（128 维）
- `feat_256`：边距离特征（1 维）

**用途**：

- VQ-VAE 训练的输入特征
- 边级离散表示学习

---

## 🚀 使用流程

### 环境依赖

```bash
# Python 3.8+
pip install torch torch-geometric biopython pandas numpy omegaconf pyyaml h5py
pip install x-transformers vector-quantize-pytorch ndlinear  # VQ-VAE 训练
pip install graphein  # 如需使用 Graphein 的角度计算功能
```

### 🔥 快速开始：完整流水线

#### 方式 1：一键运行完整 pipeline（推荐）

```bash
cd "feature extraction"
python full_pipeline.py
```

**输出**：

- `improtant data/binding_sites.h5` - 蛋白-配体接触信息
- `improtant data/binding_embeddings_*.h5` - 三图 embedding（蛋白、配体、相互作用）
- `improtant data/binding_edge_features.h5` - 边级局部特征
- `improtant data/binding_edge_features_fused.h5` - **最终融合特征（用于 VQ-VAE 训练）**

**处理流程**：

1. 分析 3432 个 PDB 文件 → 识别 25,626 条接触记录
2. 构建三张图并用 GCPNet 编码 → 生成 3,139 个样本的 embedding
3. 提取 13,798 条边的局部特征
4. 融合四个文件生成最终的边级特征矩阵（257 维）

**预计时间**：10-30 分钟（取决于机器性能）

#### 方式 2：分步运行（调试用）

### Step 1：识别结合位点

运行 `PDB_complex_analysis.ipynb`：

1. 设置路径参数（PDB 目录、输出 CSV 路径）
2. 配置距离阈值（默认 4.0 Å）和忽略的 HET 残基（水分子、离子等）
3. 批量解析 PDB，计算残基-配体最小距离
4. 导出 `binding_sites.csv`

**关键代码单元**：

```python
# 设置参数
DIST_CUTOFF = 4.0
IGNORED_HET = {"HOH", "WAT", "NA", "K", "CL", ...}

# 运行分析
analyze_all_pdbs(PDB_DIR, OUTPUT_CSV)
```

### Step 2：VQ-VAE 离散码本训练

运行 `binding_edge_codebook.ipynb`：

**Part 1：Edge 级几何码本（Cells 1-9）**

- 读取 `binding_edge_features_fused.csv`（13,798 条边 × 257 维）
- 使用简单 MLP 将边特征映射到 VQ 空间（128 维）
- 训练 VQ 码本（4096 个 codes）
- 导出 `binding_edge_codes.csv`

**Part 2：完整 VQ-VAE 训练（Cells 10-17）**

- 读取 `improtant data/binding_edge_features_fused.h5`
- 使用 FeatureProjector 将 257 维投影到 128 维
- 完整 VQ-VAE 架构：
  - GCPNet encoder → Transformer encoder → Vector Quantizer → Geometric Decoder
  - 多任务损失：MSE + backbone distance/direction + next-token prediction + VQ loss
- 保存 checkpoint 到 `checkpoints/vqvae_edge_features/`

**关键代码**：

```python
# Part 1: Edge 码本训练
edge_encoder = EdgeToVQSpace(257, 128)
vq_layer = model.vector_quantizer
# 训练并导出 edge_code

# Part 2: 完整 VQ-VAE 训练
feature_projector = FeatureProjector(257, 128)
full_vqvae = VQVAETransformer(configs, decoder, logger)
# 训练并保存 checkpoint
```

---

## 📊 关键配置文件

### `config_gcpnet_encoder.yaml`

GCPNet 编码器的完整配置，包括：

- **特征提取器**：
  - 节点标量特征：氨基酸 one-hot、序列位置编码、主链角度（α、κ、二面角）
  - 节点向量特征：backbone 方向
  - 边特征：距离、归一化向量

- **编码器结构**：
  - 6 层 GCP（Geometric-Complete Pairwise）消息传递
  - 节点标量/向量维度：128/16
  - 边标量/向量维度：32/4
  - 径向基函数：8 个高斯基（r_max=10.0 Å）
  - 激活函数：SiLU
  - Pooling：sum

### `config_vqvae.yaml`

VQ-VAE 训练的完整配置，包括：

- **模型结构**：
  - GCPNet encoder（预训练）
  - Transformer encoder：8 层，1024 维
  - Vector Quantizer：4096 codes，128 维，EMA 更新
  - TikTok 压缩：8 倍压缩因子
  - Geometric Decoder：重建 backbone 坐标

- **训练设置**：
  - Batch size：4-8（根据 GPU 内存）
  - Learning rate：1e-4
  - Optimizer：AdamW
  - Mixed precision：FP16
  - Max length：512（序列长度）

- **损失函数**：
  - MSE loss（重建损失）
  - VQ loss（码本损失，权重 0.1）
  - Backbone distance/direction loss（几何约束）
  - Next-token prediction loss（自回归）

---

## 📖 详细文档

- **`gcpnet_README.md`**：gcpnet 模块的完整 API 说明与即插即用指南
- **`data_analyzer/PDB复合物分析指南.md`**：PDB 解析与接触分析的详细步骤
- **`binding_edge_codebook.ipynb`**：⭐ 主流程文档，包含 Edge 码本和完整 VQ-VAE 训练的详细说明
- **Notebook 内嵌文档**：每个 Cell 都有中文注释和 Markdown 说明

## 🎯 核心文件说明

### ⭐⭐⭐ `binding_edge_codebook.ipynb`

**最重要的 notebook**，包含两套独立的 VQ 码本训练流程：

#### Part 1：Edge 级几何码本（Cells 1-9）

- **目标**：为蛋白-配体结合边建立离散码本
- **输入**：`binding_edge_features_fused.csv`（13,798 条边 × 257 维）
- **输出**：`binding_edge_codes.csv`（每条边的离散 code）
- **用途**：下游边级离散表示

#### Part 2：完整 VQ-VAE 训练（Cells 10-17）

- **目标**：完整实现 `vqvae.py` 的蛋白质结构生成模型
- **输入**：`improtant data/binding_edge_features_fused.h5`
- **输出**：`checkpoints/vqvae_edge_features/epoch_*.pth`
- **架构**：GCPNet + Transformer + VQ + Geometric Decoder
- **用途**：结构生成、压缩、离散表示学习

### ⭐⭐ `feature extraction/full_pipeline.py`

**完整的自动化流水线脚本**：

- 一键完成从 PDB 分析到边级特征融合的全流程
- 输出 HDF5 格式数据（高效、压缩、支持大规模数据）
- 处理 3432 个 PDB 文件，生成 13,798 条边的融合特征

---

## 🔧 扩展与自定义

### 修改距离阈值

在 `PDB_complex_analysis.ipynb` 或 `full_pipeline.py` 中修改：

```python
DIST_CUTOFF = 5.0  # 例如改为 5 Å
```

### 更换 GCPNet 配置

编辑 `config_gcpnet_encoder.yaml`，调整：

- `num_layers`：网络深度
- `emb_dim`：节点嵌入维度
- `r_max`、`num_rbf`：径向基参数

### 添加新的特征

在 `ProteinFeaturiser` 中启用更多特征：

```python
featuriser = ProteinFeaturiser(
    scalar_node_features=["amino_acid_one_hot", "dihedrals", "alpha", "kappa"],
    vector_node_features=["orientation"],
    ...
)
```

---

## 📈 数据规模统计

| 数据项      | 数量/维度   | 说明                                |
| ----------- | ----------- | ----------------------------------- |
| PDB 文件    | 3,432 个    | 蛋白-配体复合物                     |
| 接触记录    | 25,626 条   | 距离 ≤ 4.0 Å 的残基-配体对          |
| 样本数      | 3,139 个    | (pdb_id, ligand) 组合               |
| 边数        | 13,798 条   | 相互作用图的蛋白-配体边             |
| 边特征维度  | 257 维      | [h_src(128) + h_dst(128) + dist(1)] |
| VQ 码本大小 | 4,096 codes | 离散码本容量                        |
| VQ 空间维度 | 128 维      | 量化后的特征维度                    |

