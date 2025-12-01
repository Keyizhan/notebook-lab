"""
GCPNet 结合位点三图特征提取完整流水线

功能：
1. 从 binding_sites.csv 加载蛋白-配体接触信息
2. 为每个 (PDB, 配体) 样本构建三张图：
   - 蛋白 binding 残基图（Cα 节点）
   - 配体原子图
   - 蛋白-配体相互作用图
3. 使用 GCPNet 对三张图分别编码
4. 提取边级局部特征
5. 融合四个文件生成最终的边级特征矩阵

输出文件：
- binding_embeddings_protein.csv
- binding_embeddings_ligand.csv
- binding_embeddings_interaction.csv
- binding_edge_features_gcpnet.csv
- improtant data/binding_edge_features_fused.csv
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import math
import numpy as np
import pandas as pd

from Bio.PDB import PDBParser

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

# 添加项目根目录到 sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from gcpnet.features.factory import ProteinFeaturiser
from gcpnet.models.graph_encoders.gcpnet import GCPNetModel

from omegaconf import OmegaConf

# ============================================================
# 全局配置
# ============================================================

PDB_DIR = BASE_DIR / "complex-20251129T063258Z-1-001" / "complex"
BINDING_CSV = BASE_DIR / "binding_sites.csv"
OUTPUT_DIR = BASE_DIR / "improtant data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 输出文件路径
TRIPLET_PROTEIN_EMBEDDINGS_CSV = BASE_DIR / "binding_embeddings_protein.csv"
TRIPLET_LIGAND_EMBEDDINGS_CSV = BASE_DIR / "binding_embeddings_ligand.csv"
TRIPLET_INTER_EMBEDDINGS_CSV = BASE_DIR / "binding_embeddings_interaction.csv"
EDGE_FEATURES_CSV = BASE_DIR / "binding_edge_features_gcpnet.csv"
FUSED_EDGE_FEATURES_CSV = OUTPUT_DIR / "binding_edge_features_fused.csv"

# 图构建参数
K_NEIGHBORS = 16
DIST_CUTOFF = 4.0

# 氨基酸编码
AA3_TO_ID = {
    "ALA": 0, "CYS": 1, "ASP": 2, "GLU": 3, "PHE": 4,
    "GLY": 5, "HIS": 6, "ILE": 7, "LYS": 8, "LEU": 9,
    "MET": 10, "ASN": 11, "PRO": 12, "GLN": 13, "ARG": 14,
    "SER": 15, "THR": 16, "VAL": 17, "TRP": 18, "TYR": 19,
}
UNKNOWN_AA_ID = len(AA3_TO_ID)

# 原子类型编码（简化版）
ATOM_TYPE_TO_ID = {
    "C": 0, "N": 1, "O": 2, "S": 3, "P": 4,
    "F": 5, "CL": 6, "BR": 7, "I": 8,
}
UNKNOWN_ATOM_ID = len(ATOM_TYPE_TO_ID)

parser = PDBParser(QUIET=True)

# ============================================================
# Part 1: 数据加载
# ============================================================

def load_binding_sites(csv_path: Path) -> Dict[Tuple[str, str, str, int], List]:
    """按 (pdb_id, ligand_resname, ligand_chain, ligand_resnum) 分组 binding 记录。"""
    df = pd.read_csv(csv_path)
    groups = defaultdict(list)
    
    for _, row in df.iterrows():
        key = (
            str(row["pdb_id"]),
            str(row["ligand_resname"]),
            str(row["ligand_chain"]),
            int(row["ligand_resnum"]),
        )
        groups[key].append(row)
    
    print(f"共 {len(df)} 条 binding 记录，{len(groups)} 个 (pdb, ligand) 组合。")
    return groups


# ============================================================
# Part 2: 蛋白图构建
# ============================================================

def build_ca_graph_from_pdb(pdb_path: Path, protein_chains: Optional[List[str]] = None):
    """从 PDB 构建 Cα 图。"""
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = next(structure.get_models())
    
    ca_coords = []
    chain_ids = []
    resnums = []
    res_types = []
    
    for chain in model:
        chain_id = chain.id
        if protein_chains and chain_id not in protein_chains:
            continue
        
        for residue in chain:
            hetfield = residue.id[0]
            if hetfield.strip():
                continue
            
            resname = residue.get_resname().strip()
            if "CA" not in residue:
                continue
            ca = residue["CA"]
            
            ca_coords.append(ca.coord)
            chain_ids.append(chain_id)
            resnums.append(residue.id[1])
            res_types.append(AA3_TO_ID.get(resname, UNKNOWN_AA_ID))
    
    if not ca_coords:
        return [], [], None, None, None
    
    coords = torch.tensor(ca_coords, dtype=torch.float32)
    residue_type_ids = torch.tensor(res_types, dtype=torch.long)
    
    # KNN 图
    N = coords.shape[0]
    if N == 1:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        dist_mat = torch.cdist(coords, coords)
        knn = min(K_NEIGHBORS, N - 1)
        _, knn_idx = torch.topk(-dist_mat, k=knn + 1, dim=-1)
        
        rows, cols = [], []
        for i in range(N):
            for j in knn_idx[i].tolist():
                if i == j:
                    continue
                rows.append(i)
                cols.append(j)
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
    
    return chain_ids, resnums, coords, residue_type_ids, edge_index


def build_pyg_data_for_group(
    pdb_dir: Path,
    group_key: Tuple[str, str, str, int],
    group_rows: List,
    protein_chains: Optional[List[str]] = None,
) -> Optional[Data]:
    """对一个 (pdb, ligand) group：构建 Cα 图并标记 binding 残基。"""
    pdb_id, lig_resname, lig_chain, lig_resnum = group_key
    pdb_path = pdb_dir / f"{pdb_id}.pdb"
    if not pdb_path.exists():
        return None
    
    node_chain_ids, node_resnums, coords, res_type_ids, edge_index = build_ca_graph_from_pdb(
        pdb_path, protein_chains=protein_chains
    )
    if coords is None:
        return None
    
    N = coords.shape[0]
    index_map = {
        (cid, int(rnum)): i
        for i, (cid, rnum) in enumerate(zip(node_chain_ids, node_resnums))
    }
    
    # 标记 binding 残基
    y = torch.zeros(N, dtype=torch.long)
    for row in group_rows:
        cid = str(row["protein_chain"])
        rnum = int(row["protein_resnum"])
        idx = index_map.get((cid, rnum))
        if idx is not None:
            y[idx] = 1
    
    data = Data()
    data.pos = coords
    data.residue_type = res_type_ids
    data.edge_index = edge_index
    data.y = y
    data.pdb_id = pdb_id
    data.ligand_resname = lig_resname
    data.ligand_chain = lig_chain
    data.ligand_resnum = int(lig_resnum)
    
    return data


def to_batch_for_featuriser(data_list: List[Data]) -> Batch:
    """合并为 Batch，并填充 coords / seq_pos 字段供 ProteinFeaturiser 使用。"""
    batch = Batch.from_data_list(data_list)
    pos = batch.pos
    zeros = torch.zeros_like(pos)
    coords = torch.stack([zeros, pos], dim=1)
    batch.coords = coords
    batch.seq_pos = torch.arange(pos.size(0), dtype=torch.long)
    return batch


# ============================================================
# Part 3: 配体图构建
# ============================================================

def build_ligand_graph_from_pdb(
    pdb_path: Path,
    lig_resname: str,
    lig_chain: str,
    lig_resnum: int,
) -> Optional[Data]:
    """从 PDB 构建配体原子图。"""
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = next(structure.get_models())
    
    target_chain = None
    for chain in model:
        if chain.id == lig_chain:
            target_chain = chain
            break
    
    if target_chain is None:
        return None
    
    target_residue = None
    for residue in target_chain:
        if residue.get_resname().strip() == lig_resname and residue.id[1] == lig_resnum:
            target_residue = residue
            break
    
    if target_residue is None:
        return None
    
    atom_coords = []
    atom_types = []
    
    for atom in target_residue.get_atoms():
        atom_coords.append(atom.coord)
        element = atom.element.strip().upper()
        atom_types.append(ATOM_TYPE_TO_ID.get(element, UNKNOWN_ATOM_ID))
    
    if not atom_coords:
        return None
    
    coords = torch.tensor(atom_coords, dtype=torch.float32)
    atom_type_ids = torch.tensor(atom_types, dtype=torch.long)
    
    # KNN 图
    N = coords.shape[0]
    if N == 1:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        dist_mat = torch.cdist(coords, coords)
        knn = min(K_NEIGHBORS, N - 1)
        _, knn_idx = torch.topk(-dist_mat, k=knn + 1, dim=-1)
        
        rows, cols = [], []
        for i in range(N):
            for j in knn_idx[i].tolist():
                if i == j:
                    continue
                rows.append(i)
                cols.append(j)
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
    
    data = Data()
    data.pos = coords
    data.atom_type = atom_type_ids
    data.edge_index = edge_index
    
    return data


# ============================================================
# Part 4: 相互作用图构建
# ============================================================

def build_interaction_graph(protein_data: Data, ligand_data: Data) -> Data:
    """构建蛋白-配体相互作用图（双向边）。"""
    prot_coords = protein_data.pos
    lig_coords = ligand_data.pos
    
    N_prot = prot_coords.shape[0]
    N_lig = lig_coords.shape[0]
    
    # 计算距离矩阵
    dist_mat = torch.cdist(prot_coords, lig_coords)
    
    # 找出距离 <= DIST_CUTOFF 的边
    mask = (dist_mat <= DIST_CUTOFF)
    prot_idx, lig_idx = torch.where(mask)
    
    # 构建双向边
    edge_index = torch.stack([
        torch.cat([prot_idx, lig_idx + N_prot]),
        torch.cat([lig_idx + N_prot, prot_idx])
    ], dim=0)
    
    # 合并节点坐标
    pos = torch.cat([prot_coords, lig_coords], dim=0)
    
    # 节点角色标记：0=蛋白，1=配体
    node_role = torch.cat([
        torch.zeros(N_prot, dtype=torch.long),
        torch.ones(N_lig, dtype=torch.long)
    ])
    
    data = Data()
    data.pos = pos
    data.edge_index = edge_index
    data.node_role = node_role
    data.num_protein_nodes = N_prot
    data.num_ligand_nodes = N_lig
    
    return data


# ============================================================
# Part 5: GCPNet 编码器
# ============================================================

# 初始化 ProteinFeaturiser
featuriser = ProteinFeaturiser(
    representation="CA",
    scalar_node_features=["amino_acid_one_hot"],
    vector_node_features=[],
    edge_types=["knn_16"],
    scalar_edge_features=["edge_distance"],
    vector_edge_features=["edge_vectors"],
)

# 加载 GCPNet 配置并初始化模型
gcpnet_cfg_path = BASE_DIR / "config_gcpnet_encoder.yaml"
gcpnet_configs = OmegaConf.load(str(gcpnet_cfg_path))
gcpnet_kwargs = gcpnet_configs.encoder.kwargs
gcpnet_encoder = GCPNetModel(**gcpnet_kwargs)
gcpnet_encoder.eval()

# 简单的配体/相互作用图编码器（MLP）
class SimpleLigandEncoder(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=64, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, atom_type_onehot):
        return self.net(atom_type_onehot)

ligand_encoder = SimpleLigandEncoder()
ligand_encoder.eval()

interaction_encoder = SimpleLigandEncoder(in_dim=1, hidden_dim=64, out_dim=128)
interaction_encoder.eval()


def encode_protein_graph(protein_batch: Batch) -> torch.Tensor:
    """使用 ProteinFeaturiser + GCPNet 编码蛋白图。"""
    with torch.no_grad():
        # 先用 featuriser 提取特征
        protein_batch = featuriser(protein_batch)
        
        # 再用 GCPNet encoder
        h_nodes = gcpnet_encoder(protein_batch)
        
        # 对每个图的 binding 节点做 pooling
        graph_idx = protein_batch.batch
        y = protein_batch.y
        
        h_list = []
        num_graphs = int(graph_idx.max().item()) + 1
        for g in range(num_graphs):
            mask_g = (graph_idx == g)
            h_g = h_nodes[mask_g]
            y_g = y[mask_g]
            
            mask_binding = (y_g > 0)
            if mask_binding.any():
                h_pooled = h_g[mask_binding].mean(dim=0)
            else:
                h_pooled = h_g.mean(dim=0)
            h_list.append(h_pooled)
        
        return torch.stack(h_list, dim=0)


def encode_ligand_graph(ligand_data_list: List[Data]) -> torch.Tensor:
    """编码配体图（简单 MLP）。"""
    with torch.no_grad():
        h_list = []
        for data in ligand_data_list:
            atom_type = data.atom_type
            atom_type_onehot = torch.nn.functional.one_hot(
                atom_type, num_classes=UNKNOWN_ATOM_ID + 1
            ).float()
            h_atoms = ligand_encoder(atom_type_onehot)
            h_pooled = h_atoms.mean(dim=0)
            h_list.append(h_pooled)
        return torch.stack(h_list, dim=0)


def encode_interaction_graph(inter_data_list: List[Data]) -> torch.Tensor:
    """编码相互作用图（简单 MLP）。"""
    with torch.no_grad():
        h_list = []
        for data in inter_data_list:
            node_role = data.node_role.unsqueeze(-1).float()
            h_nodes = interaction_encoder(node_role)
            h_pooled = h_nodes.mean(dim=0)
            h_list.append(h_pooled)
        return torch.stack(h_list, dim=0)


# ============================================================
# Part 6: 边级局部特征提取
# ============================================================

def encode_interaction_graph_nodes(inter_data_list: List[Data]) -> List[torch.Tensor]:
    """对相互作用图的节点进行编码，返回每个图的节点 embedding。"""
    with torch.no_grad():
        h_nodes_list = []
        for data in inter_data_list:
            node_role = data.node_role.unsqueeze(-1).float()
            h_nodes = interaction_encoder(node_role)
            h_nodes_list.append(h_nodes)
        return h_nodes_list


def compute_interaction_edge_features(
    pdb_dir: Path,
    binding_groups: dict,
    max_groups: Optional[int] = None,
    protein_chains: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """计算边级局部特征。"""
    keys = list(binding_groups.keys())
    if max_groups is not None:
        keys = keys[:max_groups]
    
    protein_data_list = []
    ligand_data_list = []
    inter_data_list = []
    meta_list = []
    
    for key in keys:
        group_rows = binding_groups[key]
        pdb_id, lig_resname, lig_chain, lig_resnum = key
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        
        protein_data = build_pyg_data_for_group(pdb_dir, key, group_rows, protein_chains)
        if protein_data is None:
            continue
        
        ligand_data = build_ligand_graph_from_pdb(pdb_path, lig_resname, lig_chain, lig_resnum)
        if ligand_data is None:
            continue
        
        inter_data = build_interaction_graph(protein_data, ligand_data)
        
        protein_data_list.append(protein_data)
        ligand_data_list.append(ligand_data)
        inter_data_list.append(inter_data)
        meta_list.append(key)
    
    if not inter_data_list:
        print("没有成功构建的相互作用图样本。")
        return None
    
    # 编码相互作用图节点
    h_nodes_list = encode_interaction_graph_nodes(inter_data_list)
    
    # 提取边级特征
    edge_records = []
    for g, (data, h_nodes, key) in enumerate(zip(inter_data_list, h_nodes_list, meta_list)):
        pdb_id, lig_resname, lig_chain, lig_resnum = key
        edge_index = data.edge_index
        pos = data.pos
        node_role = data.node_role
        N_prot = data.num_protein_nodes
        
        for e in range(edge_index.shape[1]):
            src, dst = edge_index[:, e].tolist()
            
            # 只保留蛋白->配体的边
            if node_role[src] == 0 and node_role[dst] == 1:
                h_src = h_nodes[src].cpu().numpy()
                h_dst = h_nodes[dst].cpu().numpy()
                dist = torch.norm(pos[src] - pos[dst]).item()
                
                feat_vec = np.concatenate([h_src, h_dst, [dist]])
                
                edge_records.append({
                    "pdb_id": pdb_id,
                    "ligand_resname": lig_resname,
                    "ligand_chain": lig_chain,
                    "ligand_resnum": lig_resnum,
                    "graph_index": g,
                    "src_index": src,
                    "dst_index": dst - N_prot,
                    "src_role": "protein",
                    "dst_role": "ligand",
                    **{f"feat_{i}": feat_vec[i] for i in range(len(feat_vec))}
                })
    
    if not edge_records:
        print("未提取到任何边级特征。")
        return None
    
    df_edge = pd.DataFrame(edge_records)
    df_edge.to_csv(EDGE_FEATURES_CSV, index=False)
    print(f"已保存 {len(df_edge)} 条边级局部特征到 {EDGE_FEATURES_CSV}")
    
    return df_edge


# ============================================================
# Part 7: 三图 embedding 计算
# ============================================================

def compute_triplet_embeddings(
    pdb_dir: Path,
    binding_groups: dict,
    max_groups: Optional[int] = None,
    protein_chains: Optional[List[str]] = None,
):
    """计算三图 embedding 并分别保存。"""
    keys = list(binding_groups.keys())
    if max_groups is not None:
        keys = keys[:max_groups]
    
    protein_data_list = []
    ligand_data_list = []
    inter_data_list = []
    meta_list = []
    
    print(f"\n[Triplet Embeddings] 处理 {len(keys)} 个样本...")
    for i, key in enumerate(keys, 1):
        if i % 100 == 0:
            print(f"  进度: {i}/{len(keys)}", end="\r")
        
        group_rows = binding_groups[key]
        pdb_id, lig_resname, lig_chain, lig_resnum = key
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        
        protein_data = build_pyg_data_for_group(pdb_dir, key, group_rows, protein_chains)
        if protein_data is None:
            continue
        
        ligand_data = build_ligand_graph_from_pdb(pdb_path, lig_resname, lig_chain, lig_resnum)
        if ligand_data is None:
            continue
        
        inter_data = build_interaction_graph(protein_data, ligand_data)
        
        protein_data_list.append(protein_data)
        ligand_data_list.append(ligand_data)
        inter_data_list.append(inter_data)
        meta_list.append(key)
    
    print(f"\n  成功构建 {len(protein_data_list)} 个样本")
    
    if not protein_data_list:
        print("没有成功构建的样本。")
        return None
    
    # 1) 蛋白图编码
    print("\n[Triplet Embeddings] 编码蛋白图...")
    protein_batch = to_batch_for_featuriser(protein_data_list)
    h_protein = encode_protein_graph(protein_batch)
    
    # 2) 配体图编码
    print("[Triplet Embeddings] 编码配体图...")
    h_ligand = encode_ligand_graph(ligand_data_list)
    
    # 3) 相互作用图编码
    print("[Triplet Embeddings] 编码相互作用图...")
    h_inter = encode_interaction_graph(inter_data_list)
    
    # 转换为 numpy
    H_protein = h_protein.detach().cpu().numpy()
    H_ligand = h_ligand.detach().cpu().numpy()
    H_inter = h_inter.detach().cpu().numpy()
    
    # 构建元信息
    records = []
    for (pdb_id, lig_resname, lig_chain, lig_resnum) in meta_list:
        records.append({
            "pdb_id": pdb_id,
            "ligand_resname": lig_resname,
            "ligand_chain": lig_chain,
            "ligand_resnum": lig_resnum,
        })
    df_meta = pd.DataFrame(records)
    
    # 构建三个 embedding DataFrame
    feat_cols_protein = [f"feat_{i}" for i in range(H_protein.shape[1])]
    df_feat_protein = pd.DataFrame(H_protein, columns=feat_cols_protein)
    df_protein = pd.concat([df_meta, df_feat_protein], axis=1)
    
    feat_cols_ligand = [f"feat_{i}" for i in range(H_ligand.shape[1])]
    df_feat_ligand = pd.DataFrame(H_ligand, columns=feat_cols_ligand)
    df_ligand = pd.concat([df_meta, df_feat_ligand], axis=1)
    
    feat_cols_inter = [f"feat_{i}" for i in range(H_inter.shape[1])]
    df_feat_inter = pd.DataFrame(H_inter, columns=feat_cols_inter)
    df_inter = pd.concat([df_meta, df_feat_inter], axis=1)
    
    # 保存
    df_protein.to_csv(TRIPLET_PROTEIN_EMBEDDINGS_CSV, index=False)
    df_ligand.to_csv(TRIPLET_LIGAND_EMBEDDINGS_CSV, index=False)
    df_inter.to_csv(TRIPLET_INTER_EMBEDDINGS_CSV, index=False)
    
    print(f"\n已保存 {len(df_protein)} 条蛋白 embedding 到 {TRIPLET_PROTEIN_EMBEDDINGS_CSV}")
    print(f"已保存 {len(df_ligand)} 条配体 embedding 到 {TRIPLET_LIGAND_EMBEDDINGS_CSV}")
    print(f"已保存 {len(df_inter)} 条相互作用 embedding 到 {TRIPLET_INTER_EMBEDDINGS_CSV}")
    
    return df_protein, df_ligand, df_inter


# ============================================================
# Part 8: 四文件融合
# ============================================================

def fuse_edge_and_graph_level_features() -> Optional[pd.DataFrame]:
    """将边级局部特征与三路图级 embedding 融合成一个边级特征矩阵。"""
    print("\n[Fusion] 融合四个文件...")
    
    if not EDGE_FEATURES_CSV.exists():
        print(f"找不到边级特征文件: {EDGE_FEATURES_CSV}")
        return None
    
    if not TRIPLET_PROTEIN_EMBEDDINGS_CSV.exists() or \
       not TRIPLET_LIGAND_EMBEDDINGS_CSV.exists() or \
       not TRIPLET_INTER_EMBEDDINGS_CSV.exists():
        print("缺少图级 embedding 文件，请先运行三图编码。")
        return None
    
    df_edge = pd.read_csv(EDGE_FEATURES_CSV)
    df_prot = pd.read_csv(TRIPLET_PROTEIN_EMBEDDINGS_CSV)
    df_lig = pd.read_csv(TRIPLET_LIGAND_EMBEDDINGS_CSV)
    df_inter = pd.read_csv(TRIPLET_INTER_EMBEDDINGS_CSV)
    
    keys = ["pdb_id", "ligand_resname", "ligand_chain", "ligand_resnum"]
    
    # 给图级 embedding 的特征列加前缀
    df_prot = df_prot.rename(columns={c: f"prot_{c}" for c in df_prot.columns if c.startswith("feat_")})
    df_lig = df_lig.rename(columns={c: f"lig_{c}" for c in df_lig.columns if c.startswith("feat_")})
    df_inter = df_inter.rename(columns={c: f"inter_{c}" for c in df_inter.columns if c.startswith("feat_")})
    
    # 按样本主键 merge
    df_fused = df_edge.merge(df_prot, on=keys, how="left") \
                      .merge(df_lig, on=keys, how="left") \
                      .merge(df_inter, on=keys, how="left")
    
    df_fused.to_csv(FUSED_EDGE_FEATURES_CSV, index=False)
    print(f"已保存融合后的边级局部特征到 {FUSED_EDGE_FEATURES_CSV}")
    print(f"  总特征维度: {len([c for c in df_fused.columns if c.startswith('feat_') or c.startswith('prot_') or c.startswith('lig_') or c.startswith('inter_')])}")
    
    return df_fused


# ============================================================
# Main 函数
# ============================================================

def main():
    """完整流水线：全量处理所有数据。"""
    print("=" * 60)
    print("GCPNet 结合位点三图特征提取完整流水线")
    print("=" * 60)
    
    # 1. 加载 binding_sites.csv
    if not BINDING_CSV.exists():
        raise FileNotFoundError(f"binding_sites.csv 未找到：{BINDING_CSV}")
    
    binding_groups = load_binding_sites(BINDING_CSV)
    
    # 2. 计算三图 embedding（全量）
    print("\n" + "=" * 60)
    print("Step 1/3: 计算三图 embedding")
    print("=" * 60)
    compute_triplet_embeddings(PDB_DIR, binding_groups, max_groups=None)
    
    # 3. 计算边级局部特征（全量）
    print("\n" + "=" * 60)
    print("Step 2/3: 计算边级局部特征")
    print("=" * 60)
    compute_interaction_edge_features(PDB_DIR, binding_groups, max_groups=None)
    
    # 4. 融合四个文件
    print("\n" + "=" * 60)
    print("Step 3/3: 融合四个文件")
    print("=" * 60)
    fuse_edge_and_graph_level_features()
    
    print("\n" + "=" * 60)
    print("✓ 完整流水线执行完成！")
    print("=" * 60)
    print(f"\n输出文件：")
    print(f"  1. {TRIPLET_PROTEIN_EMBEDDINGS_CSV}")
    print(f"  2. {TRIPLET_LIGAND_EMBEDDINGS_CSV}")
    print(f"  3. {TRIPLET_INTER_EMBEDDINGS_CSV}")
    print(f"  4. {EDGE_FEATURES_CSV}")
    print(f"  5. {FUSED_EDGE_FEATURES_CSV}")


if __name__ == "__main__":
    main()
