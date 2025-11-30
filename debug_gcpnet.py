"""
调试 GCPNet 维度匹配问题
"""
import sys
from pathlib import Path
import torch
from torch_geometric.data import Data, Batch

# 添加路径
BASE_DIR = Path(r"c:\Users\Administrator\Desktop\IGEM\stage1\notebook-lab")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from omegaconf import OmegaConf
from types import SimpleNamespace
from gcpnet.features.factory import ProteinFeaturiser
from gcpnet.models.graph_encoders.gcpnet import GCPNetModel

print("=" * 60)
print("步骤 1: 检测 ProteinFeaturiser 实际输出维度")
print("=" * 60)

# 创建简化的 featuriser
featuriser_test = ProteinFeaturiser(
    representation="CA",
    scalar_node_features=["amino_acid_one_hot"],
    vector_node_features=[],
    edge_types=["knn_16"],
    scalar_edge_features=["edge_distance"],
    vector_edge_features=["edge_vectors"],
)

# 创建测试数据
test_data = Data()
test_data.pos = torch.randn(10, 3)
test_data.residue_type = torch.randint(0, 20, (10,))
test_data.edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
test_data.y = torch.zeros(10, dtype=torch.long)

test_batch = Batch.from_data_list([test_data])
test_batch.coords = torch.stack([torch.zeros_like(test_batch.pos), test_batch.pos], dim=1)
test_batch.seq_pos = torch.arange(test_batch.pos.size(0), dtype=torch.long)

# 运行 featuriser
test_batch = featuriser_test(test_batch)

# 检查输出维度
h_dim = test_batch.x.size(-1)
chi_dim = test_batch.x_vector_attr.size(1) if hasattr(test_batch, 'x_vector_attr') and test_batch.x_vector_attr is not None else 0
e_dim = test_batch.edge_attr.size(-1) if hasattr(test_batch, 'edge_attr') and test_batch.edge_attr is not None else 0
xi_dim = test_batch.edge_vector_attr.size(1) if hasattr(test_batch, 'edge_vector_attr') and test_batch.edge_vector_attr is not None else 0

print(f"✓ Featuriser 实际输出维度:")
print(f"  - 节点标量特征 (h): {h_dim}")
print(f"  - 节点向量特征 (chi): {chi_dim}")
print(f"  - 边标量特征 (e): {e_dim}")
print(f"  - 边向量特征 (xi): {xi_dim}")

print("\n" + "=" * 60)
print("步骤 2: 检查 YAML 配置文件中的默认维度")
print("=" * 60)

CFG_PATH = BASE_DIR / "config_gcpnet_encoder.yaml"
cfg = OmegaConf.load(str(CFG_PATH))
model_cfg = cfg.encoder.kwargs.model_cfg

print(f"配置文件中的默认维度:")
print(f"  - h_input_dim: {model_cfg.h_input_dim}")
print(f"  - chi_input_dim: {model_cfg.chi_input_dim}")
print(f"  - e_input_dim: {model_cfg.e_input_dim}")
print(f"  - xi_input_dim: {model_cfg.xi_input_dim}")

print("\n" + "=" * 60)
print("步骤 3: 修正配置并初始化模型")
print("=" * 60)

# 转换配置
enc_kwargs_dict = OmegaConf.to_container(cfg.encoder.kwargs, resolve=True)

# 更新维度以匹配实际输出
print(f"\n更新配置维度以匹配 featuriser 输出...")
print(f"⚠️  注意：GCPNet 会自动将边特征 (1维) + RBF展开 (8维) = 9维")
print(f"   所以 e_input_dim 应保持为 9，不要修改为 1！")

# 修改节点标量和向量特征维度，边特征保持不变
enc_kwargs_dict['model_cfg']['h_input_dim'] = h_dim
enc_kwargs_dict['model_cfg']['chi_input_dim'] = chi_dim  # 也需要修改为 0
# e_input_dim 保持为 9（1 + 8 RBF）
# xi_input_dim 保持为 1

print(f"✓ 更新后的配置维度:")
print(f"  - h_input_dim: {enc_kwargs_dict['model_cfg']['h_input_dim']} (修改: 49 → {h_dim})")
print(f"  - chi_input_dim: {enc_kwargs_dict['model_cfg']['chi_input_dim']} (修改: 2 → {chi_dim})")
print(f"  - e_input_dim: {enc_kwargs_dict['model_cfg']['e_input_dim']} (保持不变: 1+8 RBF)")
print(f"  - xi_input_dim: {enc_kwargs_dict['model_cfg']['xi_input_dim']} (保持不变)")

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

enc_kwargs = {
    'num_layers': enc_kwargs_dict['num_layers'],
    'emb_dim': enc_kwargs_dict['emb_dim'],
    'node_s_emb_dim': enc_kwargs_dict['node_s_emb_dim'],
    'node_v_emb_dim': enc_kwargs_dict['node_v_emb_dim'],
    'edge_s_emb_dim': enc_kwargs_dict['edge_s_emb_dim'],
    'edge_v_emb_dim': enc_kwargs_dict['edge_v_emb_dim'],
    'r_max': enc_kwargs_dict['r_max'],
    'num_rbf': enc_kwargs_dict['num_rbf'],
    'activation': enc_kwargs_dict['activation'],
    'pool': enc_kwargs_dict['pool'],
    'module_cfg': dict_to_namespace(enc_kwargs_dict['module_cfg']),
    'model_cfg': dict_to_namespace(enc_kwargs_dict['model_cfg']),
    'layer_cfg': dict_to_namespace(enc_kwargs_dict['layer_cfg']),
}

print(f"\n初始化 GCPNet 模型...")
try:
    test_encoder = GCPNetModel(**enc_kwargs).eval()
    print(f"✓ 模型初始化成功！")
except Exception as e:
    print(f"✗ 模型初始化失败: {e}")
    raise

print("\n" + "=" * 60)
print("步骤 4: 测试前向传播")
print("=" * 60)

# 确保 batch 有所有必需的属性
if not hasattr(test_batch, "x_vector_attr") or test_batch.x_vector_attr is None:
    test_batch.x_vector_attr = torch.zeros(test_batch.x.size(0), 0, 3, device=test_batch.x.device)
if not hasattr(test_batch, "edge_vector_attr") or test_batch.edge_vector_attr is None:
    test_batch.edge_vector_attr = torch.zeros(test_batch.edge_index.size(1), 0, 3, device=test_batch.x.device)

print(f"测试 batch 的属性:")
print(f"  - x shape: {test_batch.x.shape}")
print(f"  - x_vector_attr shape: {test_batch.x_vector_attr.shape}")
print(f"  - edge_attr shape: {test_batch.edge_attr.shape}")
print(f"  - edge_vector_attr shape: {test_batch.edge_vector_attr.shape}")

try:
    with torch.no_grad():
        output = test_encoder(test_batch)
    print(f"✓ 前向传播成功！")
    print(f"  - node_embedding shape: {output['node_embedding'].shape}")
    print(f"  - graph_embedding shape: {output['graph_embedding'].shape}")
except Exception as e:
    print(f"✗ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n" + "=" * 60)
print("调试完成！所有测试通过。")
print("=" * 60)
