"""
调试脚本：测试前 10 个数据的处理流程
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path
import torch
import pandas as pd
from torch_geometric.data import Data, Batch

# 添加路径
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from gcpnet.models.graph_encoders.gcpnet import GCPNetModel
from gcpnet.features.factory import ProteinFeaturiser
from omegaconf import OmegaConf

# 路径配置
PDB_DIR = BASE_DIR / "complex-20251129T063258Z-1-001" / "complex"
BINDING_SITES_CSV = BASE_DIR / "binding_sites.csv"

# 加载配置
gcpnet_cfg_path = BASE_DIR / "config_gcpnet_encoder.yaml"
gcpnet_configs = OmegaConf.load(str(gcpnet_cfg_path))

print("=" * 70)
print("调试脚本：测试 GCPNet featuriser")
print("=" * 70)

# 1. 初始化 featuriser（使用简化配置，避免需要多原子的特征）
print("\n[Step 1] 初始化 ProteinFeaturiser...")
featuriser = ProteinFeaturiser(
    representation="CA",
    scalar_node_features=["amino_acid_one_hot", "sequence_positional_encoding"],  # 23 + 16 = 39 维
    vector_node_features=["orientation"],  # 2 维
    edge_types=["knn_16"],
    scalar_edge_features=["edge_distance"],  # 9 维
    vector_edge_features=["edge_vectors"],  # 1 维
)
featuriser.eval()
print(f"  ✓ Featuriser 创建成功（简化配置，CA-only）")

# 2. 初始化 GCPNet encoder
print("\n[Step 2] 初始化 GCPNet encoder...")
gcpnet_kwargs = gcpnet_configs.encoder.kwargs

# 修改输入维度以匹配简化的特征集
gcpnet_kwargs['model_cfg']['h_input_dim'] = 39  # amino_acid_one_hot (23) + seq_pos_enc (16)

print(f"  期望输入维度:")
print(f"    - h_input_dim: {gcpnet_kwargs['model_cfg']['h_input_dim']} (修改为 39)")
print(f"    - chi_input_dim: {gcpnet_kwargs['model_cfg']['chi_input_dim']}")
print(f"    - e_input_dim: {gcpnet_kwargs['model_cfg']['e_input_dim']}")
print(f"    - xi_input_dim: {gcpnet_kwargs['model_cfg']['xi_input_dim']}")

gcpnet_encoder = GCPNetModel(**gcpnet_kwargs)
gcpnet_encoder.eval()
print(f"  ✓ GCPNet encoder 创建成功")

# 3. 加载 binding sites 数据
print("\n[Step 3] 加载 binding sites 数据...")
df = pd.read_csv(BINDING_SITES_CSV)
print(f"  总记录数: {len(df)}")

# 按 (pdb_id, ligand_resname, ligand_chain, ligand_resnum) 分组
grouped = df.groupby(['pdb_id', 'ligand_resname', 'ligand_chain', 'ligand_resnum'])
binding_groups = [(key, list(group.itertuples(index=False))) for key, group in grouped]
print(f"  分组数: {len(binding_groups)}")

# 4. 测试前 10 个数据
print("\n[Step 4] 测试前 10 个数据...")
print("=" * 70)

from full_pipeline import build_pyg_data_for_group, to_batch_for_featuriser

test_groups = binding_groups[:10]
success_count = 0
failed_groups = []

for i, (group_key, group_rows) in enumerate(test_groups):
    pdb_id, lig_resname, lig_chain, lig_resnum = group_key
    print(f"\n[{i+1}/10] 测试 {pdb_id} - {lig_resname} {lig_chain}:{lig_resnum}")
    
    try:
        # 构建蛋白图
        protein_data = build_pyg_data_for_group(
            PDB_DIR, group_key, group_rows, protein_chains=None
        )
        
        if protein_data is None:
            print(f"  ✗ 无法构建蛋白图（PDB 文件不存在或无有效残基）")
            failed_groups.append((group_key, "无法构建蛋白图"))
            continue
        
        print(f"  ✓ 蛋白图构建成功")
        print(f"    - 节点数: {protein_data.pos.shape[0]}")
        print(f"    - 边数: {protein_data.edge_index.shape[1]}")
        print(f"    - coords shape: {protein_data.coords.shape}")
        print(f"    - residue_type shape: {protein_data.residue_type.shape}")
        print(f"    - seq_pos shape: {protein_data.seq_pos.shape}")
        
        # 转换为 batch
        batch = to_batch_for_featuriser([protein_data])
        print(f"  ✓ Batch 创建成功")
        print(f"    - batch.pos shape: {batch.pos.shape}")
        print(f"    - batch.coords shape: {batch.coords.shape}")
        print(f"    - batch.residue_type shape: {batch.residue_type.shape}")
        print(f"    - batch.seq_pos shape: {batch.seq_pos.shape}")
        print(f"    - batch._slice_dict keys: {list(batch._slice_dict.keys())}")
        
        # 应用 featuriser
        with torch.no_grad():
            batch_featurised = featuriser(batch)
        
        print(f"  ✓ Featuriser 应用成功")
        
        # 检查特征维度
        if hasattr(batch_featurised, 'x'):
            print(f"    - x (标量特征) shape: {batch_featurised.x.shape}")
        if hasattr(batch_featurised, 'x_vector'):
            print(f"    - x_vector (向量特征) shape: {batch_featurised.x_vector.shape}")
        if hasattr(batch_featurised, 'edge_attr'):
            print(f"    - edge_attr (边标量特征) shape: {batch_featurised.edge_attr.shape}")
        if hasattr(batch_featurised, 'edge_attr_vector'):
            print(f"    - edge_attr_vector (边向量特征) shape: {batch_featurised.edge_attr_vector.shape}")
        
        # 应用 GCPNet encoder
        with torch.no_grad():
            h_nodes = gcpnet_encoder(batch_featurised)
        
        print(f"  ✓ GCPNet encoder 应用成功")
        # GCPNet 返回字典，包含 'graph', 'node', 'edge' 等键
        if isinstance(h_nodes, dict):
            print(f"    - 输出类型: dict")
            for key, val in h_nodes.items():
                if hasattr(val, 'shape'):
                    print(f"      - {key}: {val.shape}")
                else:
                    print(f"      - {key}: {type(val)}")
        else:
            print(f"    - 输出 shape: {h_nodes.shape}")
        
        success_count += 1
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        failed_groups.append((group_key, str(e)))

# 5. 总结
print("\n" + "=" * 70)
print("测试总结")
print("=" * 70)
print(f"成功: {success_count}/10")
print(f"失败: {len(failed_groups)}/10")

if failed_groups:
    print("\n失败的组:")
    for group_key, error in failed_groups:
        print(f"  - {group_key}: {error}")

if success_count == 10:
    print("\n✓ 所有测试通过！可以运行完整 pipeline")
else:
    print("\n✗ 存在失败案例，需要修复")
