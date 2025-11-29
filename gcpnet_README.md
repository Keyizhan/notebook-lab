# gcpnet 目录脚本说明与即插即用指南

本文件对 `gcpnet/` 目录内的所有脚本逐一解释，重点说明：

- 每个脚本/模块的**主要功能**
- 关键类与函数的**核心逻辑**
- 你在项目中如何**即插即用**（特别是结合蛋白结构 + 关键结合位点的任务）

目录结构如下（当前精简版本）：

- `__init__.py`
- `constants.py`
- `geometry.py`
- `heads.py`
- `typecheck.py`
- `types.py`
- `features/`
  - `__init__.py`
  - `factory.py`
  - `node_features.py`
  - `edge_features.py`
  - `representation.py`
  - `sequence_features.py`
  - `utils.py`
- `layers/`
  - `__init__.py`
  - `structure_proj.py`
- `models/`
  - `__init__.py`（空）
  - `base.py`
  - `utils.py`
  - `graph_encoders/`（GCPNet 具体编码器定义，未在此文档展开）

---

## 顶层模块

### `__init__.py`

**功能：**

- 导入并关闭 Graphein 库的 verbose 输出：
  - `from graphein import verbose; verbose(False)`
- 尝试从已安装的 `gcpnet` 包中读取版本号，读取不到则回退到 `"0.0.0"`。

**即插即用建议：**

- 使用时只需在代码中 `import gcpnet` 即可，不需要修改此文件。
- 导入后可以通过 `gcpnet.__version__` 查看版本号（如果你以后打包为 pip 包）。

---

### `constants.py`

**功能：**

- 定义项目路径常量：
  - `SRC_PATH = Path(__file__).parent` → `gcpnet` 源码所在目录。
  - `PROJECT_PATH = SRC_PATH.parent` → 项目根目录。

**即插即用建议：**

- 当你需要在代码中用**相对路径**加载本项目的配置文件或模型权重时，可以：

  ```python
  from gcpnet.constants import PROJECT_PATH
  config_path = PROJECT_PATH / "configs" / "encoder.yaml"
  ```

---

### `typecheck.py`

**功能：**

- 控制是否启用运行时类型检查（基于 `jaxtyping` + `beartype`）。
- 逻辑：
  - 环境变量 `GCPNET_ENABLE_RUNTIME_TYPECHECK` 为 `"1"/"true"/"yes"` 且两个依赖都安装时：
    - `jaxtyped` 装饰器和 `typechecker` 使用真实版本 → 执行时会做类型检查。
  - 否则：
    - `jaxtyped`、`typechecker` 都是空装饰器/空函数 → 不做额外检查，性能更好。

**即插即用建议：**

- 你在使用 `gcpnet` 的函数/类时不需要关心这个文件，它只影响装饰器行为。
- 调试阶段如果想**更严格检查输入类型**，可以在运行环境中设置：

  ```bash
  export GCPNET_ENABLE_RUNTIME_TYPECHECK=1
  ```

---

### `types.py`

**功能：**

- 集中定义项目中常用的类型别名和特征名称枚举：

  - 激活函数类型 `ActivationType`：
    - 允许值：`"relu"`, `"elu"`, `"leaky_relu"`, `"tanh"`, `"sigmoid"`, `"none"`, `"silu"`, `"swish"`。

  - 编码器输出类型 `EncoderOutput = NewType("EncoderOutput", Dict[str, torch.Tensor])`。

  - 特征名称：
    - `ScalarNodeFeature`：
      - `"amino_acid_one_hot"`
      - `"alpha"`
      - `"kappa"`
      - `"dihedrals"`
      - `"sequence_positional_encoding"`
    - `VectorNodeFeature`：
      - `"orientation"`
    - `ScalarEdgeFeature`：
      - `"edge_distance"`
    - `VectorEdgeFeature`：
      - `"edge_vectors"`

  - `OrientationTensor`：节点方向张量的 NewType，形状为 `(n_nodes, 2, 3)`。

**即插即用建议：**

- 在配置或代码中指定特征时，
  请使用这些字符串常量作为合法选项，避免拼写错误：

  ```python
  scalar_node_features = ["amino_acid_one_hot", "dihedrals"]
  vector_node_features = ["orientation"]
  scalar_edge_features = ["edge_distance"]
  vector_edge_features = ["edge_vectors"]
  ```

---

## 几何核心

### `geometry.py`

**功能：**

- 提供 3D 几何变换的核心类，用于实现 E(3) 等变/不变操作：

  - `fp32_autocast_context`：
    - 一个上下文管理器，在 GPU + AMP 环境下强制几何计算使用 `float32`，
      避免半精度导致的数值不稳定。

  - `Rotation` 协议：
    - 抽象接口，约定旋转需要提供：
      - `identity`、`random`、`as_matrix`、`compose`、`apply`、`invert` 等方法。

  - `RotationMatrix(Rotation)`：
    - 具体的旋转矩阵实现，内部存储张量形状为 `(..., 3, 3)`。
    - 提供：
      - `identity(shape)`：构造单位旋转；
      - `random(shape)`：随机正交基；
      - `compose`/`apply`/`invert`：组合变换、应用到点、求逆；
      - `from_graham_schmidt`：用 Gram-Schmidt 正交化一组向量，生成旋转矩阵。

  - `Affine3D`：
    - 表示三维仿射变换（平移 + 旋转）：
      - 属性：`trans: Tensor(..., 3)`，`rot: Rotation`。
      - 方法：
        - `identity(shape_or_affine, rotation_type=RotationMatrix)`：生成单位刚体；
        - `compose(other)`：组合两次空间变换；
        - `apply(p)`：作用在坐标 `p` 上；
        - `invert()`：求逆刚体；
        - `mask(mask_tensor, with_zero=False)`：按掩码选择性应用/重置变换；
        - `from_graham_schmidt(neg_x_axis, origin, xy_plane)`：从局部几何构造刚体。

**即插即用建议：**

- 绝大多数情况下你不需要直接改这个文件。
- 如果你想自定义**结构预测头**或对中间特征做几何变换，可以：

  - 使用 `Affine3D.identity` 为每个残基建立局部坐标系；
  - 通过 `Affine3D.compose` 堆叠多个预测更新；
  - 用 `Affine3D.apply` 把刚体变换作用在一个模板主链坐标上（参考 `layers/structure_proj.py` 的用法）。

---

## 输出头

### `heads.py`

**功能：**

- 提供通用的网络输出头模块：

  1. `PairwisePredictionHead`
     - 输入：
       - 节点嵌入张量 `x` (`[batch, n_nodes, embed_dim]`)，
       - 可选 pairwise 额外特征 `pairwise`。
     - 过程：
       - 先用 `downproject` 降维；
       - 将投影后的向量拆成 `q`、`k` 两部分；
       - 构造 pairwise 特征：`prod = q_i * k_j` 和 `diff = q_i - k_j`；
       - 与 `pairwise`（若存在）拼接后，经过 `linear1 + GELU + LayerNorm + linear2` 输出到 `n_bins` 维。
     - 应用：适合做**残基-残基对**的预测，如距离分布、接触图等。

  2. `RegressionHead`
     - 输入：节点或图级别的特征 `features`；
     - 结构：`Linear(embed_dim → embed_dim)` + GELU + LayerNorm + `Linear(embed_dim → output_dim)`；
     - 用途：回归/简单分类任务（例如图级打分或节点级标量预测）。

**即插即用建议：**

- 对于“关键结合位点识别”任务：
  - 你可以在 GCPNet 编码器输出的节点 embedding 上加 `RegressionHead(embed_dim, 1)` 或自定义 `nn.Linear(embed_dim, 1)`，
    然后用 `sigmoid + BCE` 做二分类（0/1 是否为 binding residue）。
  - `PairwisePredictionHead` 更适合预测残基-残基关系（例如界面接触矩阵），
    如果你之后想做蛋白-配体的 pairwise 距离预测，可以使用它。

---

## 特征模块 `features/`

### `features/factory.py` – `ProteinFeaturiser`

**功能：**

- 这是整个项目最重要的“特征工厂”：
  - 输入：Graphein 的 `ProteinBatch` 或 PyG 的 `Batch`（至少包含 `coords`、`residue_type`、`edge_index` 等字段）。
  - 输出：同一个 Batch，但填充好了：
    - `x`：标量节点特征；
    - `pos`：节点位置（此精简版本中是 Cα 坐标）；
    - `edge_index` + `edge_type`：图结构；
    - `edge_attr`：标量边特征（例如距离）；
    - `x_vector_attr`、`edge_vector_attr`：向量特征。

核心参数：

- `representation`: 结构表示方式。精简版中实际只支持 `"CA"`（见 `representation.py`）。
- `scalar_node_features`: 节点标量特征列表（见 `types.ScalarNodeFeature`）。
- `vector_node_features`: 节点向量特征列表（如 `"orientation"`）。
- `edge_types`: 边类型列表，精简实现仅支持类似 `"knn_k"` 的 KNN 图。
- `scalar_edge_features`: 边标量特征名称列表（如 `"edge_distance"`）。
- `vector_edge_features`: 边向量特征名称列表（如 `"edge_vectors"`）。

`forward` 主要流程：

1. **标量节点特征**：
   - 如果包含 `"sequence_positional_encoding"`，使用 `PositionalEncoding` 写入 `batch.seq_pos → batch.x`；
   - 调用 `compute_scalar_node_features`（见 `node_features.py`）
     - 组合 one-hot 氨基酸、主链角度等特征，并与 positional encoding 拼接。

2. **结构表示转换**：
   - `batch = transform_representation(batch, self.representation)`；
   - 当前只支持 `"CA"`：将 `batch.pos` 设置为 `batch.coords[:, 1, :]`（Cα 坐标）。

3. **向量节点特征**：
   - 若 `vector_node_features` 非空，调用 `compute_vector_node_features`，
     在 `batch.x_vector_attr` 中写入每个节点的方向等向量特征。

4. **图边构建**：
   - 若 `edge_types` 非空，调用 `_compute_edges`：
     - 当前实现只检查 edge_types 是否为单个 `"knn_k"` 风格字符串；
     - 实际上这里重用已有的 `batch.edge_index`，若 `edge_type` 为空则填 0。

5. **边标量/向量特征**：
   - `compute_scalar_edge_features`：基于 `pos` 和 `edge_index` 计算边距离；
   - `compute_vector_edge_features`：计算归一化的边向量，写入 `edge_vector_attr`。

**即插即用建议：**

- 当你已经有了一个 `ProteinBatch`（例如使用 Graphein 从 PDB 构建）：

  ```python
  from gcpnet.features.factory import ProteinFeaturiser

  featuriser = ProteinFeaturiser(
      representation="CA",
      scalar_node_features=["amino_acid_one_hot", "dihedrals"],
      vector_node_features=["orientation"],
      edge_types=["knn_16"],
      scalar_edge_features=["edge_distance"],
      vector_edge_features=["edge_vectors"],
  )

  batch = featuriser(batch)  # batch 现在带有所有几何与序列特征
  ```

- 在你的 binding_sites 场景中：
  - 把“是否为关键结合位点”作为 `batch.y`（标签）或额外节点特征（例如附加到 `batch.x`）；
  - 让 GCPNet 的 encoder 在这个几何丰富的图上做节点分类即可。

---

### `features/node_features.py`

**功能：**

- 计算节点级标量和向量特征。

1. `compute_scalar_node_features(x, node_features)`：
   - 支持特征：
     - `"amino_acid_one_hot"`：
       - 调用 `sequence_features.amino_acid_one_hot(x, num_classes=23)`，
         从 `x.residue_type` 生成 one-hot。
     - `"alpha"`, `"kappa"`, `"dihedrals"`：
       - 使用 Graphein 的角度函数，对主链坐标 `x.coords` + 批次索引 `x.batch` 计算主链几何角度，
         并做嵌入（如 `rad=True, embed=True`）。
     - `"sequence_positional_encoding"`：
       - 在这里被跳过（由 `ProteinFeaturiser` 处理）。
   - 返回：大小为 `(N, F)` 的张量，F 是所有选定特征拼接后的总维度。

2. `compute_vector_node_features(x, vector_features)`：
   - 当前支持：
     - `"orientation"`：通过 `orientations(...)` 函数计算每个节点 backbone 的前向/后向方向向量。
   - 会在 `x.x_vector_attr` 中存储一个形状 `(N, 2, 3)` 的方向张量。

3. `orientations(X, coords_slice_index, ca_idx=1)`：
   - 从 `coords` 中取出 Cα 坐标；
   - 为每个节点构造：
     - `forward` 向量（当前节点到下一个残基），
     - `backward` 向量（当前节点到上一个残基），
   - 对端点进行零填充处理，然后归一化成单位向量。

**即插即用建议：**

- 你通常不需要直接调用这些函数，只需要在 `ProteinFeaturiser` 中配置 `scalar_node_features` / `vector_node_features` 列表。
- 确保：
  - `batch.coords` 和 `batch.batch` 合法；
  - `batch.residue_type` 已被设置（例如由 Graphein 从 PDB 序列解析）。

---

### `features/edge_features.py`

**功能：**

- 计算边的标量和向量特征：

1. `compute_scalar_edge_features(x, features)`：
   - 仅支持 `"edge_distance"`：
     - 使用 `_edge_distance(pos, edge_index)` 计算边的欧式距离，返回 `(num_edges, 1)` 张量。

2. `compute_vector_edge_features(x, features)`：
   - 仅支持 `"edge_vectors"`：
     - 根据 `x.pos` 和 `x.edge_index` 计算 `pos[i] - pos[j]`，归一化后存到 `x.edge_vector_attr`，
       形状一般为 `(num_edges, 1, 3)`。

**即插即用建议：**

- 要使用这些特征，保证：
  - `batch.pos` 已由 `transform_representation` 设置为 Cα 坐标；
  - `batch.edge_index` 已构图；
  - 然后在 `ProteinFeaturiser` 中添加：

  ```python
  scalar_edge_features=["edge_distance"],
  vector_edge_features=["edge_vectors"],
  ```

---

### `features/representation.py`

**功能：**

- 控制从原始 `coords` 到图节点坐标 `pos` 的表示方式。
- 精简版中只支持：

  ```python
  transform_representation(batch, representation_type: Literal["CA"])
  ```

  - 当 `representation_type == "CA"` 时：
    - 将 `batch.pos` 设置为 `batch.coords[:, 1, :]`（默认假设 index 1 是 Cα 原子）。
  - 如果传入其它字符串会抛异常（防止错误配置）。

**即插即用建议：**

- 使用 Graphein 构建 `coords` 时，保持默认原子顺序（N, CA, C, ...），则 index 1 就是 CA。
- 直接在 `ProteinFeaturiser` 里设置 `representation="CA"` 即可。

---

### `features/sequence_features.py`

**功能：**

- 提供基于序列的节点特征：

  - `amino_acid_one_hot(x, num_classes=23)`：
    - 从 `x.residue_type` 生成 one-hot，类别数默认为 23（包含标准 20 氨基酸 + 额外类别）。

**即插即用建议：**

- 确保构图时为每个节点设置了 `residue_type`（整数编码）。
- 在 `ProteinFeaturiser` 中加入 `"amino_acid_one_hot"` 即可使用。

---

### `features/utils.py`

**功能：**

- `_normalize(tensor, dim=-1)`：
  - 安全地归一化张量，在分母为 0 时用 0 替代，避免 NaN/Inf。

**即插即用建议：**

- 主要在内部被节点/边特征计算调用，你一般不需要直接使用。

---

## 结构预测层 `layers/`

### `layers/structure_proj.py`

**功能：**

- 定义一个结构预测头 `Dim6RotStructureHead`，从节点 latent 表征预测 backbone 刚体变换和坐标：

  - `BB_COORDINATES`：
    - 固定的 3 个 backbone 原子（N, CA, C）的局部坐标模板（3×3）。

  - `Dim6RotStructureHead`：

    - 初始化参数：
      - `input_dim`：encoder 输出的 embedding 维度；
      - `trans_scale_factor`：平移缩放因子（默认 10.0）；
      - `predict_torsion_angles`：是否额外预测侧链扭转角（精简版只保留接口）。

    - `forward(x, affine, affine_mask, **kwargs)`：
      - 若 `affine` 为 `None`：
        - 用 `Affine3D.identity` 初始化单位刚体；
      - 否则从输入继承初始刚体；
      - 通过全连接 + GELU + LayerNorm 得到投影输出：
        - 分割为 `trans`、`vec_x`、`vec_y`（以及可选的 torsion）；
      - 对向量归一化，使用 `Affine3D.from_graham_schmidt` 将其转换为增量刚体 `update`；
      - `rigids = rigids.compose(update.mask(affine_mask))` 更新刚体；
      - 将 `BB_COORDINATES` 应用在刚体上，得到预测坐标 `pred_xyz`。

**即插即用建议：**

- 若你的任务只是“特征提取 + 残基级分类”（例如识别 binding 残基），可以暂时不使用这个结构头。
- 若你未来想根据 encoder 输出重建/精修 backbone 结构，则可以接入 `Dim6RotStructureHead` 作为输出层。

---

## 模型模块 `models/`

### `models/base.py`

**功能：**

- 提供**配置驱动**的 encoder 构建与预训练加载接口。

关键组件：

1. 动态导入工具：

   - `_import_from_string(path: str)`：
     - 从字符串如 `"gcpnet.features.factory.ProteinFeaturiser"` 加载类对象。

   - `instantiate_module(spec: Mapping[str, Any]) -> Optional[nn.Module]`：
     - `spec["module"]` 指定完整路径，`spec["kwargs"]` 是初始化参数；
     - 返回对应实例。

2. 配置加载：

   - `load_encoder_config(path)`：
     - 用 YAML 读入 encoder 配置，并保证返回的是一个 Mapping。

3. `EncoderComponents` dataclass：

   - 封装：
     - `featuriser`：例如 `ProteinFeaturiser`；
     - `encoder`：例如 GCPNet 主体；
     - `task_transform`：可选的任务级 transform（对 batch 做额外处理）。

4. `PretrainedEncoder(nn.Module)`：

   - 封装了 `featuriser + encoder (+ task_transform)`：

     - `featurise(batch)`：
       - 先跑 `self.featuriser`，再可选 `self.task_transform`。

     - `forward(batch)`：
       - `batch = self.featurise(batch)`；
       - 返回 `self.encoder(batch)` 的结果（一个 `EncoderOutput` 字典）。

5. `_build_components(cfg)`：

   - 从 config（一般是一个 YAML dict）中读取：
     - `cfg["features"]`：featuriser 的 spec；
     - `cfg["task"]["transform"]`：任务变换的 spec（可选）；
     - `cfg["encoder"]`：编码器本体的 spec；
   - 对其中嵌套的 `module_cfg` / `model_cfg` / `layer_cfg` 做 `SimpleNamespace` 包装，方便属性访问。

6. 预训练加载：

   - `_load_checkpoint(path, map_location)`：
     - 支持不同风格的 checkpoint（`state_dict`/`model_state_dict`/直接是 dict）。

   - `_apply_checkpoint(components, state_dict, strict)`：
     - 从 state_dict 中抽取 `encoder.`、`featuriser.`、`task_transform.` 前缀对应的子权重并加载。

   - `instantiate_encoder(config_path)`：
     - 返回 `(EncoderComponents, cfg)`，不加载权重。

   - `load_pretrained_encoder(config_path, checkpoint_path=None, ...)`：
     - 一步创建并加载预训练 encoder，返回 `PretrainedEncoder` 实例。

**即插即用建议：**

- 如果你有一份官方/自己写的 encoder 配置（YAML），可以这样直接读取并使用：

  ```python
  from gcpnet.models.base import load_pretrained_encoder

  encoder = load_pretrained_encoder(
      config_path="path/to/encoder.yaml",
      checkpoint_path="path/to/weights.ckpt",  # 可选
      map_location="cuda"  # 或 "cpu"
  )

  # batch: 你用 Graphein + ProteinFeaturiser 生成的 Batch/ProteinBatch
  out = encoder(batch)  # out 是一个 dict[str, Tensor]，包含节点/图 embedding 等
  ```

- 对你的任务：
  - 你可以把 `binding_sites.csv` 得到的残基标签 `y` 附到 batch 上，
  - 用 `PretrainedEncoder` 抽取节点 embedding，
  - 再接一个简单的节点分类头（例如 `RegressionHead` 或 `nn.Linear`）。

---

### `models/utils.py`

**功能：**

- 各种在 GCPNet 内部使用的通用工具函数：

1. 聚合函数：

   - `get_aggregation(aggregation: str)`：
     - 返回一个 pooling 函数：`sum`/`mean`/`max`，
     - 使用 `torch_scatter` 对 `Batch` 或 `batch_index` 做图级聚合。

2. 激活函数选择：

   - `get_activations(act_name: ActivationType, return_functional=False)`：
     - 根据字符串返回 `nn.Module` 或 functional（`torch.nn.functional` 中的对应函数）。

3. 几何中心化/反中心化：

   - `centralize(batch, key, batch_index, node_mask=None)`：
     - 对某个键（如 `coords` 或 `pos`）按图做中心化，返回 `(centroid, centered)`。
   - `decentralize(batch, key, batch_index, entities_centroid, node_mask=None)`：
     - 将中心化后的量恢复到原始坐标系。

4. 边局部坐标：

   - `localize(pos, edge_index, norm_pos_diff=True, node_mask=None)`：
     - 根据节点坐标构造每条边的 3×3 局部正交基（差分向量、叉积向量、垂直向量），
     - 作为边级几何特征，支持等变网络。

5. 其它：

   - `safe_norm(x, dim=-1, eps=1e-8, keepdim=False, sqrt=True)`：安全范数；
   - `is_identity(obj)`：判断模块/函数是否是恒等映射。

**即插即用建议：**

- 如果你自己写新的 GNN layer 或 head，想使用和 GCPNet 一致的几何处理方式，可以直接调用这些函数。
- 对普通“拿 encoder 当特征提取器”的使用场景，不必直接操作这里。

---

## `models/graph_encoders/`

此目录下包含：

- `gcpnet.py`：GCPNet 主编码器实现（图网络的主体），依赖前述的 `models.utils`、`features`、`geometry` 等模块；
- `components/` 与子 `layers/`：内部组件和层结构。

> 由于你当前的需求主要是：**把蛋白-配体 binding 数据接到一个已经存在的 GCPNet 编码器上**，
> 你通常不需要修改这里的实现，只需要：

1. 用 Graphein 构建 `ProteinBatch`；
2. 用 `ProteinFeaturiser` 生成带几何特征的 Batch；
3. 用 `load_pretrained_encoder` 或手动实例化 GCPNet encoder；
4. 接一个简单的节点级分类 head 预测关键残基。

如果后续你需要对 `gcpnet.py` 里的网络结构做进一步解读或修改，可以单独再展开该文件。

---

## 总结：如何在你当前项目中即插即用 gcpnet

1. **数据准备（PDB + binding_sites）**：

   - 利用已有脚本/Notebook，从 PDB 构建 `ProteinBatch`：
     - 至少包含：`coords`、`residue_type`、`batch`、`edge_index`、`seq_pos`（可选）。
   - 使用 `binding_sites.csv` 给每个残基一个标签：`is_binding_residue`（0/1），存到 `batch.y` 或单独的张量中。

2. **特征提取（gcpnet.features）**：

   - 初始化 `ProteinFeaturiser`，配置所需的节点/边特征：

     ```python
     featuriser = ProteinFeaturiser(
         representation="CA",
         scalar_node_features=["amino_acid_one_hot", "dihedrals"],
         vector_node_features=["orientation"],
         edge_types=["knn_16"],
         scalar_edge_features=["edge_distance"],
         vector_edge_features=["edge_vectors"],
     )

     batch = featuriser(batch)
     ```

3. **编码 + 任务头（gcpnet.models + heads）**：

   - 使用 `load_pretrained_encoder` 加载一个 GCPNet encoder（如有对应 config & checkpoint），或手动实例化：

     ```python
     from gcpnet.models.base import load_pretrained_encoder

     encoder = load_pretrained_encoder("path/to/encoder.yaml", map_location="cuda")

     # encoder(batch) 返回一个 dict，包括节点/图 embedding
     output = encoder(batch)
     node_repr = output["node_embeddings"]  # 具体 key 依实际实现为准
     ```

   - 再用 `RegressionHead` 或自定义 `nn.Linear` 对 `node_repr` 做节点级别的关键位点预测。

4. **保持空间不变性/等变性**：

   - 所有几何操作（coords → pos、localize、edge_vectors、orientation 等）
     均由 `features` 与 `models.utils` 里的函数保证旋转/平移下的正确性；
   - 你只需要把 binding 标签作为**附加的标量监督信号**传入网络即可。
