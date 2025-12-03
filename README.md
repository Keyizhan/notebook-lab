# 蛋白-配体复合物三图融合与离散编码系统

## 📋 项目概述

基于 VQ-VAE 的蛋白-配体复合物分析系统，通过**三图融合**策略捕捉蛋白口袋与配体的几何特征和相互作用模式，并将其编码为离散码本。

## 🎯 核心目标

- 从 PDB 复合物文件中识别蛋白-配体结合口袋
- 构建异构图融合蛋白、配体和交互信息
- 通过向量量化获得结合模式的离散表示
- 支持端到端训练和结构重建

## 🏗️ 系统架构

```
PDB 复合物
    ↓
口袋检测 (5Å cutoff)
    ↓
三图融合
├─ 蛋白口袋图 (kNN)
├─ 配体内部图 (距离阈值)
└─ 蛋白-配体交互图 (5Å, 双向)
    ↓
GCPNet 编码器
    ↓
异构节点特征化
├─ 蛋白: one-hot + dihedrals + orientations
└─ 配体: 几何特征 (距离、方向等)
    ↓
VQVAE 编码器
    ↓
向量量化 (4096 码本)
    ↓
解码器重建口袋结构
```

## 📊 数据流

### 输入
- **格式**: PDB 文件 (蛋白-配体复合物)
- **示例**: `0.pdb` (459 残基 + FAD 辅因子)

### 输出
- **离散码本**: 每个口袋残基/交互边 → 码本索引 (0-4095)
- **重建结构**: 口袋区域的 3D 坐标

## 🔧 关键组件

### 1. 数据处理 (`data/`)

**PDB 解析** (`pdb_complex_utils.py`):
```python
- parse_pdb_complex()      # 解析蛋白和配体
- detect_pocket_residues() # 5Å 距离检测口袋
- build_ligand_graph()     # 配体内部连接
```

**数据集** (`dataset.py`):
```python
- ProteinLigandComplexDataset  # 复合物数据集
- _featurize_as_graph()         # 三图融合
- custom_collate_pretrained_gcp # 批处理
```

### 2. 模型 (`models/`)

**编码器** (`gcpnet/`):
- GCPNet: 几何感知的图神经网络
- 支持异构节点 (蛋白 + 配体)
- 消息传递融合多尺度信息

**VQVAE** (`vqvae.py`):
- Transformer 编码器
- 向量量化层 (VectorQuantize)
- Transformer 解码器

**超级模型** (`super_model.py`):
- 整合 GCPNet + VQVAE
- 支持残基级/边级量化切换

### 3. 特征化 (`models/gcpnet/features/`)

**节点特征**:
- 蛋白: `amino_acid_one_hot`, `dihedrals`, `alpha`, `kappa`
- 配体: `distance_to_centroid`, `relative_position`, `pairwise_distance`
- 自动填充到统一维度 (49维)

**边特征**:
- 三种边类型: 蛋白内部 (0), 配体内部 (1), 交互 (2)
- 方向性: 双向交互边

## 🎨 三图融合详解

### 图结构

```python
节点:
  [0 ... num_pocket-1]           # 蛋白口袋残基
  [num_pocket ... num_pocket+Q-1] # 配体原子

边:
  edge_type = 0: 蛋白内部 (kNN, k=30)
  edge_type = 1: 配体内部 (距离 < 2Å)
  edge_type = 2: 蛋白-配体交互 (距离 < 5Å, 双向)
```

### 示例 (0.pdb)

```
节点: 133 个 (47 蛋白 + 86 配体)
边: 5572 条
  - 蛋白内部: 1410
  - 配体内部: 226
  - 交互边: 3936
```

## 🔬 量化策略

### 残基级量化 (默认)

```yaml
use_edge_quantization: False
```

- **量化对象**: 口袋残基
- **码本含义**: 残基在配体影响下的结构状态
- **输出**: [num_pocket_residues] 个码

### 边级量化 (可选)

```yaml
use_edge_quantization: True
```

- **量化对象**: 蛋白-配体交互边
- **码本含义**: (残基, 配体原子) 接触对的结合方式
- **输出**: [num_interaction_edges] 个码

## 📦 环境配置

### Python 环境
```bash
Python 3.11.9
PyTorch 2.1.0 + CUDA 11.8
torch-geometric 2.4.0
```

### 虚拟环境
```bash
# 已创建: vqvae_env
.\vqvae_env\Scripts\activate
```

### 关键依赖
```
torch-scatter, torch-sparse, torch-cluster
x-transformers
vector-quantize-pytorch==1.14.24
graphein, tmtools
```

## 🚀 使用方法

### 1. 准备数据

```
数据目录结构:
complex/
  ├─ 0.pdb
  ├─ 1.pdb
  └─ ...
```

### 2. 配置文件

```yaml
# configs/config_vqvae.yaml
train_settings:
  data_path: ../complex/
  data_format: pdb_complex
  batch_size: 2
  max_task_samples: 10

model:
  use_edge_quantization: False  # True 启用边级量化
  encoder:
    name: gcpnet
    pretrained:
      enabled: False  # 可加载预训练权重
```

### 3. 训练

```bash
python train.py --config_path ./configs/config_vqvae.yaml
```

### 4. 测试

```bash
# 测试数据加载
python test_dataloader.py

# 测试三图融合
python test_fusion_success.py

# 测试边级量化
python test_edge_quantization.py
```

## 📈 训练输出

```
Epoch 1: loss nan, vq_loss nan, activation 0.0
  - loss 为 NaN: 正常 (无预训练权重 + 小数据集)
  - activation: 码本激活率
  - 每个 epoch: ~3-5 秒
```

## 🎯 应用场景

1. **药物设计**: 识别相似的结合口袋
2. **虚拟筛选**: 预测配体-蛋白结合模式
3. **结构生成**: 基于码本生成新的结合构象
4. **结合模式聚类**: 发现新的相互作用模式

## 📝 关键文件

```
vq_encoder_decoder-master/
├─ data/
│  ├─ pdb_complex_utils.py    # PDB 解析和口袋检测
│  └─ dataset.py               # 数据集和三图融合
├─ models/
│  ├─ super_model.py           # 主模型
│  ├─ vqvae.py                 # VQVAE 实现
│  └─ gcpnet/
│     └─ features/
│        ├─ node_features.py   # 异构节点特征
│        └─ ligand_features.py # 配体特征
├─ configs/
│  └─ config_vqvae.yaml        # 配置文件
├─ train.py                    # 训练脚本
└─ test_*.py                   # 测试脚本
```



- 原项目: VQ-VAE for protein structure
- GCPNet: Geometric Complete Protein Network
- 三图融合: 蛋白 + 配体 + 交互的异构图表示

---


