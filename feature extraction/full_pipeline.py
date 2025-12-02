"""
完整的蛋白-配体结合位点分析与特征提取流水线

功能：
1. 分析 PDB 复合物，识别蛋白-配体接触残基
2. 构建三张图（蛋白、配体、相互作用）
3. 使用 GCPNet 提取特征
4. 生成边级局部特征并融合
5. 所有输出保存为 HDF5 格式

模型架构升级：
- 蛋白图编码器：完整版 GCPNet（6层，128维标量 + 16维向量特征）
- 配体图编码器：GCPNet 架构（替代原 SimpleLigandEncoder MLP）
- 相互作用图编码器：GCPNet 架构（替代原 SimpleLigandEncoder MLP）
- 特征提取：使用完整的 ProteinFeaturiser，包含 alpha、kappa、dihedrals 等几何特征
- 配置来源：config_gcpnet_encoder.yaml

输出文件（HDF5）：
- binding_sites.h5                    # 蛋白-配体接触信息
- binding_embeddings_protein.h5       # 蛋白图级 embedding
- binding_embeddings_ligand.h5        # 配体图级 embedding
- binding_embeddings_interaction.h5   # 相互作用图级 embedding
- binding_edge_features.h5            # 边级局部特征
- binding_edge_features_fused.h5      # 最终融合的边级特征矩阵
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import math

import numpy as np
import pandas as pd
import h5py

from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa

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
OUTPUT_DIR = BASE_DIR / "improtant data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 输出文件路径（HDF5）
BINDING_SITES_H5 = OUTPUT_DIR / "binding_sites.h5"
PROTEIN_EMBEDDINGS_H5 = OUTPUT_DIR / "binding_embeddings_protein.h5"
LIGAND_EMBEDDINGS_H5 = OUTPUT_DIR / "binding_embeddings_ligand.h5"
INTERACTION_EMBEDDINGS_H5 = OUTPUT_DIR / "binding_embeddings_interaction.h5"
EDGE_FEATURES_H5 = OUTPUT_DIR / "binding_edge_features.h5"
FUSED_EDGE_FEATURES_H5 = OUTPUT_DIR / "binding_edge_features_fused.h5"

# PDB 分析参数
DIST_CUTOFF = 4.0
IGNORED_HET = {
    "HOH", "WAT",  # 水
    "NA", "K", "CL", "MG", "CA", "ZN", "FE", "MN", "CU",
    "SO4", "PO4", "IOD", "GOL", "PEG",
}
RECEPTOR_CHAINS: List[str] = []  # 空列表表示分析所有蛋白链

# 图构建参数
K_NEIGHBORS = 16

# 氨基酸编码
AA3_TO_ID = {
    "ALA": 0, "CYS": 1, "ASP": 2, "GLU": 3, "PHE": 4,
    "GLY": 5, "HIS": 6, "ILE": 7, "LYS": 8, "LEU": 9,
    "MET": 10, "ASN": 11, "PRO": 12, "GLN": 13, "ARG": 14,
    "SER": 15, "THR": 16, "VAL": 17, "TRP": 18, "TYR": 19,
}
UNKNOWN_AA_ID = len(AA3_TO_ID)

# 原子类型编码
ATOM_TYPE_TO_ID = {
    "C": 0, "N": 1, "O": 2, "S": 3, "P": 4,
    "F": 5, "CL": 6, "BR": 7, "I": 8,
}
UNKNOWN_ATOM_ID = len(ATOM_TYPE_TO_ID)

parser = PDBParser(QUIET=True)


# ============================================================
# Part 1: PDB 复合物分析（蛋白-配体接触识别）
# ============================================================

def split_protein_and_ligands(structure):
    """将结构中的原子分为蛋白残基和小分子配体残基。"""
    protein_residues = []
    ligand_residues = []
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            chain_is_receptor = (not RECEPTOR_CHAINS or chain_id in RECEPTOR_CHAINS)
            
            for residue in chain:
                resname = residue.get_resname().strip()
                hetfield = residue.id[0]
                
                if is_aa(residue, standard=True) and chain_is_receptor:
                    protein_residues.append((chain_id, residue))
                elif hetfield.startswith("H") and resname not in IGNORED_HET:
                    ligand_residues.append((chain_id, residue))
    
    return protein_residues, ligand_residues


def compute_contacts_for_structure(pdb_path: Path) -> List[dict]:
    """对单个 PDB 文件计算蛋白残基与配体残基的最小原子-原子距离。"""
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    protein_residues, ligand_residues = split_protein_and_ligands(structure)
    
    if not protein_residues or not ligand_residues:
        return []
    
    all_atoms = [atom for atom in structure.get_atoms()]
    _ = NeighborSearch(all_atoms)
    
    records = []
    
    for prot_chain_id, prot_res in protein_residues:
        prot_atoms = list(prot_res.get_atoms())
        
        for lig_chain_id, lig_res in ligand_residues:
            lig_atoms = list(lig_res.get_atoms())
            min_dist = math.inf
            
            for pa in prot_atoms:
                for la in lig_atoms:
                    d = pa - la
                    if d < min_dist:
                        min_dist = d
            
            if min_dist <= DIST_CUTOFF:
                res_id = prot_res.id[1]
                icode = prot_res.id[2].strip() or ""
                lig_res_id = lig_res.id[1]
                lig_icode = lig_res.id[2].strip() or ""
                
                record = {
                    "pdb_id": pdb_path.stem,
                    "protein_chain": prot_chain_id,
                    "protein_resnum": res_id,
                    "protein_icode": icode,
                    "protein_resname": prot_res.get_resname().strip(),
                    "ligand_resname": lig_res.get_resname().strip(),
                    "ligand_chain": lig_chain_id,
                    "ligand_resnum": lig_res_id,
                    "ligand_icode": lig_icode,
                    "min_distance": round(float(min_dist), 3),
                }
                records.append(record)
    
    return records


def analyze_all_pdbs(pdb_dir: Path) -> pd.DataFrame:
    """遍历目录下所有 .pdb 文件并返回 binding sites DataFrame。"""
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    print(f"\n[PDB Analysis] Found {len(pdb_files)} PDB files under {pdb_dir}")
    
    all_records = []
    for i, pdb_path in enumerate(pdb_files, 1):
        if i % 100 == 0 or i == len(pdb_files):
            print(f"  进度: [{i}/{len(pdb_files)}] Processing {pdb_path.name} ...", end="\r")
        try:
            recs = compute_contacts_for_structure(pdb_path)
            all_records.extend(recs)
        except Exception as e:
            print(f"\n  Error processing {pdb_path.name}: {e}")
    
    print(f"\n[PDB Analysis] Total contact records: {len(all_records)}")
    
    if not all_records:
        print("[PDB Analysis] No contacts found. Check parameters.")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_records)
    return df


def save_binding_sites_to_h5(df: pd.DataFrame, output_path: Path):
    """将 binding sites 保存为 HDF5 格式。"""
    with h5py.File(output_path, 'w') as f:
        # 保存字符串列
        for col in ['pdb_id', 'protein_chain', 'protein_icode', 'protein_resname', 
                    'ligand_resname', 'ligand_chain', 'ligand_icode']:
            f.create_dataset(col, data=df[col].astype('S'))
        
        # 保存数值列
        f.create_dataset('protein_resnum', data=df['protein_resnum'].values)
        f.create_dataset('ligand_resnum', data=df['ligand_resnum'].values)
        f.create_dataset('min_distance', data=df['min_distance'].values)
        
        f.attrs['num_records'] = len(df)
        f.attrs['description'] = 'Protein-ligand binding site contact records'
    
    print(f"[PDB Analysis] Saved {len(df)} records to {output_path}")


def load_binding_sites_from_h5(h5_path: Path) -> pd.DataFrame:
    """从 HDF5 加载 binding sites。"""
    with h5py.File(h5_path, 'r') as f:
        data = {
            'pdb_id': f['pdb_id'][:].astype(str),
            'protein_chain': f['protein_chain'][:].astype(str),
            'protein_resnum': f['protein_resnum'][:],
            'protein_icode': f['protein_icode'][:].astype(str),
            'protein_resname': f['protein_resname'][:].astype(str),
            'ligand_resname': f['ligand_resname'][:].astype(str),
            'ligand_chain': f['ligand_chain'][:].astype(str),
            'ligand_resnum': f['ligand_resnum'][:],
            'ligand_icode': f['ligand_icode'][:].astype(str),
            'min_distance': f['min_distance'][:],
        }
    return pd.DataFrame(data)


def group_binding_sites(df: pd.DataFrame) -> Dict[Tuple[str, str, str, int], List]:
    """按 (pdb_id, ligand_resname, ligand_chain, ligand_resnum) 分组。"""
    groups = defaultdict(list)
    for _, row in df.iterrows():
        key = (
            str(row["pdb_id"]),
            str(row["ligand_resname"]),
            str(row["ligand_chain"]),
            int(row["ligand_resnum"]),
        )
        groups[key].append(row)
    
    print(f"[Grouping] {len(df)} 条记录分成 {len(groups)} 个 (pdb, ligand) 组合")
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
    """构建蛋白 Cα 图并标记 binding 残基。"""
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
    
    y = torch.zeros(N, dtype=torch.long)
    for row in group_rows:
        # row 是 namedtuple，使用属性访问
        cid = str(row.protein_chain)
        rnum = int(row.protein_resnum)
        idx = index_map.get((cid, rnum))
        if idx is not None:
            y[idx] = 1
    
    # GCPNet featuriser 需要 coords 为 (N, 2, 3) 格式
    # 对于 CA-only 表示，两帧都使用 CA 坐标
    coords_3d = torch.stack([coords, coords], dim=1)  # (N, 2, 3)
    
    data = Data()
    data.pos = coords
    data.coords = coords_3d  # ← 关键：在这里添加 coords
    data.residue_type = res_type_ids
    data.edge_index = edge_index
    data.y = y
    data.seq_pos = torch.arange(N, dtype=torch.long)  # ← 添加 seq_pos
    data.pdb_id = pdb_id
    data.ligand_resname = lig_resname
    data.ligand_chain = lig_chain
    data.ligand_resnum = int(lig_resnum)
    
    return data


def to_batch_for_featuriser(data_list: List[Data]) -> Batch:
    """合并为 Batch 供 ProteinFeaturiser 使用。"""
    # 现在 coords 和 seq_pos 已经在 Data 对象中了
    # Batch.from_data_list 会自动处理它们的切片
    batch = Batch.from_data_list(data_list)
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
    """构建蛋白-配体相互作用图。"""
    prot_coords = protein_data.pos
    lig_coords = ligand_data.pos
    
    N_prot = prot_coords.shape[0]
    N_lig = lig_coords.shape[0]
    
    dist_mat = torch.cdist(prot_coords, lig_coords)
    mask = (dist_mat <= DIST_CUTOFF)
    prot_idx, lig_idx = torch.where(mask)
    
    edge_index = torch.stack([
        torch.cat([prot_idx, lig_idx + N_prot]),
        torch.cat([lig_idx + N_prot, prot_idx])
    ], dim=0)
    
    pos = torch.cat([prot_coords, lig_coords], dim=0)
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
# Part 5: GCPNet 编码器初始化
# ============================================================

# 完整版 GCPNet featuriser 配置（匹配 config_gcpnet_encoder.yaml）
featuriser = ProteinFeaturiser(
    representation="CA",
    scalar_node_features=[
        "amino_acid_one_hot",
        "sequence_positional_encoding",
        "alpha",
        "kappa",
        "dihedrals"
    ],
    vector_node_features=["orientation"],
    edge_types=["knn_16"],
    scalar_edge_features=["edge_distance"],
    vector_edge_features=["edge_vectors"],
)

# 从 YAML 加载完整版 GCPNet 配置
from types import SimpleNamespace

gcpnet_cfg_path = BASE_DIR / "config_gcpnet_encoder.yaml"
gcpnet_configs = OmegaConf.load(str(gcpnet_cfg_path))

# 转换配置为可用格式
enc_kwargs_dict = OmegaConf.to_container(gcpnet_configs.encoder.kwargs, resolve=True)

def dict_to_namespace(d):
    """递归地将字典转换为 SimpleNamespace"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

# 构建 GCPNet encoder 参数
gcpnet_kwargs = {
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

gcpnet_encoder = GCPNetModel(**gcpnet_kwargs)
gcpnet_encoder.eval()

print(f"[OK] GCPNet 编码器初始化成功")
print(f"  - 层数: {gcpnet_kwargs['num_layers']}")
print(f"  - 节点标量维度: {gcpnet_kwargs['node_s_emb_dim']}")
print(f"  - 节点向量维度: {gcpnet_kwargs['node_v_emb_dim']}")


# 配体和相互作用图也使用 GCPNet 架构
# 为配体和相互作用图创建独立的 GCPNet 编码器
ligand_encoder = GCPNetModel(**gcpnet_kwargs)
ligand_encoder.eval()

interaction_encoder = GCPNetModel(**gcpnet_kwargs)
interaction_encoder.eval()

print(f"[OK] 配体编码器和相互作用编码器初始化成功")


def _prepare_gcpnet_batch(batch: Batch) -> Batch:
    """准备 GCPNet 所需的批次格式"""
    # 确保 batch 有 h, chi, e, xi 字段
    if not hasattr(batch, 'h'):
        batch.h = batch.x if hasattr(batch, 'x') else torch.zeros(batch.pos.size(0), gcpnet_kwargs['model_cfg'].h_input_dim)
    if not hasattr(batch, 'chi'):
        batch.chi = getattr(batch, 'x_vector_attr', torch.zeros(batch.pos.size(0), gcpnet_kwargs['model_cfg'].chi_input_dim, 3))
    if not hasattr(batch, 'e'):
        batch.e = getattr(batch, 'edge_attr', torch.zeros(batch.edge_index.size(1), gcpnet_kwargs['model_cfg'].e_input_dim))
    if not hasattr(batch, 'xi'):
        batch.xi = getattr(batch, 'edge_vector_attr', torch.zeros(batch.edge_index.size(1), gcpnet_kwargs['model_cfg'].xi_input_dim, 3))
    return batch


def encode_protein_graph(protein_batch: Batch) -> dict:
    """编码蛋白图，返回字典包含 node_embedding 和 graph_embedding。"""
    with torch.no_grad():
        # 使用 featuriser 提取特征
        protein_batch = featuriser(protein_batch)
        
        # 准备 GCPNet 格式
        protein_batch = _prepare_gcpnet_batch(protein_batch)
        
        # GCPNet 前向传播
        output = gcpnet_encoder(protein_batch)
        
        # GCPNet 返回字典: {'node_embedding': ..., 'graph_embedding': ...}
        h_nodes = output['node_embedding']
        
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
        
        graph_embeddings = torch.stack(h_list, dim=0)
        
        return {
            'node_embedding': h_nodes,
            'graph_embedding': graph_embeddings
        }


def encode_ligand_graph(ligand_data_list: List[Data]) -> torch.Tensor:
    """编码配体图（使用 GCPNet）。"""
    with torch.no_grad():
        h_list = []
        for data in ligand_data_list:
            # 为配体构建简单的节点特征
            atom_type = data.atom_type
            atom_type_onehot = torch.nn.functional.one_hot(
                atom_type, num_classes=UNKNOWN_ATOM_ID + 1
            ).float()
            
            # 准备 GCPNet 输入
            data.h = atom_type_onehot
            data.chi = torch.zeros(atom_type.size(0), gcpnet_kwargs['model_cfg'].chi_input_dim, 3)
            
            # 计算边特征
            edge_index = data.edge_index
            if edge_index.size(1) > 0:
                src, dst = edge_index
                edge_vec = data.pos[dst] - data.pos[src]
                edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)
                data.e = edge_dist
                data.xi = edge_vec.unsqueeze(1)  # [E, 1, 3]
            else:
                data.e = torch.zeros(0, gcpnet_kwargs['model_cfg'].e_input_dim)
                data.xi = torch.zeros(0, gcpnet_kwargs['model_cfg'].xi_input_dim, 3)
            
            data.batch = torch.zeros(atom_type.size(0), dtype=torch.long)
            
            # GCPNet 编码
            output = ligand_encoder(data)
            h_atoms = output['node_embedding']
            h_pooled = h_atoms.mean(dim=0)
            h_list.append(h_pooled)
        
        return torch.stack(h_list, dim=0)


def encode_interaction_graph(inter_data_list: List[Data]) -> torch.Tensor:
    """编码相互作用图（使用 GCPNet）。"""
    with torch.no_grad():
        h_list = []
        for data in inter_data_list:
            # 为相互作用图构建节点特征（基于节点角色）
            node_role = data.node_role
            node_role_onehot = torch.nn.functional.one_hot(node_role, num_classes=2).float()
            
            # 准备 GCPNet 输入
            data.h = node_role_onehot
            data.chi = torch.zeros(node_role.size(0), gcpnet_kwargs['model_cfg'].chi_input_dim, 3)
            
            # 计算边特征
            edge_index = data.edge_index
            if edge_index.size(1) > 0:
                src, dst = edge_index
                edge_vec = data.pos[dst] - data.pos[src]
                edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)
                data.e = edge_dist
                data.xi = edge_vec.unsqueeze(1)  # [E, 1, 3]
            else:
                data.e = torch.zeros(0, gcpnet_kwargs['model_cfg'].e_input_dim)
                data.xi = torch.zeros(0, gcpnet_kwargs['model_cfg'].xi_input_dim, 3)
            
            data.batch = torch.zeros(node_role.size(0), dtype=torch.long)
            
            # GCPNet 编码
            output = interaction_encoder(data)
            h_nodes = output['node_embedding']
            h_pooled = h_nodes.mean(dim=0)
            h_list.append(h_pooled)
        
        return torch.stack(h_list, dim=0)


def encode_interaction_graph_nodes(inter_data_list: List[Data]) -> List[torch.Tensor]:
    """编码相互作用图节点（用于边级特征，使用 GCPNet）。"""
    with torch.no_grad():
        h_nodes_list = []
        for data in inter_data_list:
            # 为相互作用图构建节点特征
            node_role = data.node_role
            node_role_onehot = torch.nn.functional.one_hot(node_role, num_classes=2).float()
            
            # 准备 GCPNet 输入
            data.h = node_role_onehot
            data.chi = torch.zeros(node_role.size(0), gcpnet_kwargs['model_cfg'].chi_input_dim, 3)
            
            # 计算边特征
            edge_index = data.edge_index
            if edge_index.size(1) > 0:
                src, dst = edge_index
                edge_vec = data.pos[dst] - data.pos[src]
                edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)
                data.e = edge_dist
                data.xi = edge_vec.unsqueeze(1)  # [E, 1, 3]
            else:
                data.e = torch.zeros(0, gcpnet_kwargs['model_cfg'].e_input_dim)
                data.xi = torch.zeros(0, gcpnet_kwargs['model_cfg'].xi_input_dim, 3)
            
            data.batch = torch.zeros(node_role.size(0), dtype=torch.long)
            
            # GCPNet 编码
            output = interaction_encoder(data)
            h_nodes = output['node_embedding']
            h_nodes_list.append(h_nodes)
        
        return h_nodes_list


# ============================================================
# Part 6: 三图 embedding 计算并保存为 HDF5
# ============================================================

def compute_and_save_triplet_embeddings(
    pdb_dir: Path,
    binding_groups: dict,
    max_groups: Optional[int] = None,
    batch_size: int = 100,  # ← 分批处理，避免内存溢出
):
    """计算三图 embedding 并保存为 HDF5（分批处理）。"""
    keys = list(binding_groups.keys())
    if max_groups is not None:
        keys = keys[:max_groups]
    
    print(f"\n[Triplet Embeddings] 处理 {len(keys)} 个样本（批大小: {batch_size}）...")
    
    # 用于累积所有批次的结果
    all_h_protein = []
    all_h_ligand = []
    all_h_inter = []
    all_meta = []
    
    # 分批处理
    num_batches = (len(keys) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(keys))
        batch_keys = keys[start_idx:end_idx]
        
        print(f"\n[Batch {batch_idx + 1}/{num_batches}] 处理样本 {start_idx + 1}-{end_idx}...")
        
        protein_data_list = []
        ligand_data_list = []
        inter_data_list = []
        meta_list = []
        
        for i, key in enumerate(batch_keys, 1):
            if i % 20 == 0 or i == len(batch_keys):
                print(f"  构建图: {i}/{len(batch_keys)}", end="\r")
            
            group_rows = binding_groups[key]
            pdb_id, lig_resname, lig_chain, lig_resnum = key
            pdb_path = pdb_dir / f"{pdb_id}.pdb"
            
            protein_data = build_pyg_data_for_group(pdb_dir, key, group_rows)
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
            print("  跳过此批次（无有效样本）")
            continue
        
        # 编码三图（当前批次）
        print("  编码蛋白图...")
        protein_batch = to_batch_for_featuriser(protein_data_list)
        h_protein = encode_protein_graph(protein_batch)['graph_embedding'].cpu().numpy()
        
        print("  编码配体图...")
        h_ligand = encode_ligand_graph(ligand_data_list).cpu().numpy()
        
        print("  编码相互作用图...")
        h_inter = encode_interaction_graph(inter_data_list).cpu().numpy()
        
        # 累积结果
        all_h_protein.append(h_protein)
        all_h_ligand.append(h_ligand)
        all_h_inter.append(h_inter)
        all_meta.extend(meta_list)
        
        # 清理内存
        del protein_data_list, ligand_data_list, inter_data_list
        del protein_batch, h_protein, h_ligand, h_inter
        import gc
        gc.collect()
    
    # 合并所有批次的结果
    print(f"\n[Triplet Embeddings] 合并所有批次的结果...")
    h_protein = np.vstack(all_h_protein)
    h_ligand = np.vstack(all_h_ligand)
    h_inter = np.vstack(all_h_inter)
    meta_list = all_meta
    
    print(f"  总样本数: {len(meta_list)}")
    print(f"  蛋白 embedding shape: {h_protein.shape}")
    print(f"  配体 embedding shape: {h_ligand.shape}")
    print(f"  相互作用 embedding shape: {h_inter.shape}")
    
    # 保存为 HDF5
    meta_arrays = {
        'pdb_id': np.array([k[0] for k in meta_list], dtype='S'),
        'ligand_resname': np.array([k[1] for k in meta_list], dtype='S'),
        'ligand_chain': np.array([k[2] for k in meta_list], dtype='S'),
        'ligand_resnum': np.array([k[3] for k in meta_list], dtype=np.int32),
    }
    
    # 保存蛋白 embedding
    with h5py.File(PROTEIN_EMBEDDINGS_H5, 'w') as f:
        for key, val in meta_arrays.items():
            f.create_dataset(key, data=val)
        f.create_dataset('embeddings', data=h_protein)
        f.attrs['num_samples'] = len(h_protein)
        f.attrs['embedding_dim'] = h_protein.shape[1]
    
    # 保存配体 embedding
    with h5py.File(LIGAND_EMBEDDINGS_H5, 'w') as f:
        for key, val in meta_arrays.items():
            f.create_dataset(key, data=val)
        f.create_dataset('embeddings', data=h_ligand)
        f.attrs['num_samples'] = len(h_ligand)
        f.attrs['embedding_dim'] = h_ligand.shape[1]
    
    # 保存相互作用 embedding
    with h5py.File(INTERACTION_EMBEDDINGS_H5, 'w') as f:
        for key, val in meta_arrays.items():
            f.create_dataset(key, data=val)
        f.create_dataset('embeddings', data=h_inter)
        f.attrs['num_samples'] = len(h_inter)
        f.attrs['embedding_dim'] = h_inter.shape[1]
    
    print(f"\n已保存 {len(h_protein)} 条蛋白 embedding 到 {PROTEIN_EMBEDDINGS_H5}")
    print(f"已保存 {len(h_ligand)} 条配体 embedding 到 {LIGAND_EMBEDDINGS_H5}")
    print(f"已保存 {len(h_inter)} 条相互作用 embedding 到 {INTERACTION_EMBEDDINGS_H5}")


# ============================================================
# Part 7: 边级局部特征提取并保存为 HDF5
# ============================================================

def compute_and_save_edge_features(
    pdb_dir: Path,
    binding_groups: dict,
    max_groups: Optional[int] = None,
):
    """计算边级局部特征并保存为 HDF5。"""
    keys = list(binding_groups.keys())
    if max_groups is not None:
        keys = keys[:max_groups]
    
    print(f"\n[Edge Features] 处理 {len(keys)} 个样本...")
    
    protein_data_list = []
    ligand_data_list = []
    inter_data_list = []
    meta_list = []
    
    for i, key in enumerate(keys, 1):
        if i % 100 == 0 or i == len(keys):
            print(f"  进度: {i}/{len(keys)}", end="\r")
        
        group_rows = binding_groups[key]
        pdb_id, lig_resname, lig_chain, lig_resnum = key
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        
        protein_data = build_pyg_data_for_group(pdb_dir, key, group_rows)
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
    
    print(f"\n  成功构建 {len(inter_data_list)} 个样本")
    
    if not inter_data_list:
        print("没有成功构建的相互作用图样本。")
        return
    
    # 编码相互作用图节点
    print("[Edge Features] 编码相互作用图节点...")
    h_nodes_list = encode_interaction_graph_nodes(inter_data_list)
    
    # 提取边级特征
    print("[Edge Features] 提取边级特征...")
    all_edge_features = []
    all_meta = []
    
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
                all_edge_features.append(feat_vec)
                
                all_meta.append({
                    'pdb_id': pdb_id,
                    'ligand_resname': lig_resname,
                    'ligand_chain': lig_chain,
                    'ligand_resnum': lig_resnum,
                    'graph_index': g,
                    'src_index': src,
                    'dst_index': dst - N_prot,
                })
    
    if not all_edge_features:
        print("未提取到任何边级特征。")
        return
    
    # 保存为 HDF5
    edge_features_array = np.array(all_edge_features, dtype=np.float32)
    
    with h5py.File(EDGE_FEATURES_H5, 'w') as f:
        # 保存特征矩阵
        f.create_dataset('features', data=edge_features_array)
        
        # 保存元信息
        f.create_dataset('pdb_id', data=np.array([m['pdb_id'] for m in all_meta], dtype='S'))
        f.create_dataset('ligand_resname', data=np.array([m['ligand_resname'] for m in all_meta], dtype='S'))
        f.create_dataset('ligand_chain', data=np.array([m['ligand_chain'] for m in all_meta], dtype='S'))
        f.create_dataset('ligand_resnum', data=np.array([m['ligand_resnum'] for m in all_meta], dtype=np.int32))
        f.create_dataset('graph_index', data=np.array([m['graph_index'] for m in all_meta], dtype=np.int32))
        f.create_dataset('src_index', data=np.array([m['src_index'] for m in all_meta], dtype=np.int32))
        f.create_dataset('dst_index', data=np.array([m['dst_index'] for m in all_meta], dtype=np.int32))
        
        f.attrs['num_edges'] = len(all_edge_features)
        f.attrs['feature_dim'] = edge_features_array.shape[1]
    
    print(f"\n已保存 {len(all_edge_features)} 条边级局部特征到 {EDGE_FEATURES_H5}")


# ============================================================
# Part 8: 融合四个 HDF5 文件
# ============================================================

def fuse_edge_and_graph_embeddings():
    """融合边级特征与三图 embedding，保存为 HDF5。"""
    print("\n[Fusion] 融合四个 HDF5 文件...")
    
    if not EDGE_FEATURES_H5.exists():
        print(f"找不到边级特征文件: {EDGE_FEATURES_H5}")
        return
    
    if not PROTEIN_EMBEDDINGS_H5.exists() or \
       not LIGAND_EMBEDDINGS_H5.exists() or \
       not INTERACTION_EMBEDDINGS_H5.exists():
        print("缺少图级 embedding 文件。")
        return
    
    # 读取边级特征
    with h5py.File(EDGE_FEATURES_H5, 'r') as f:
        edge_features = f['features'][:]
        edge_meta = {
            'pdb_id': f['pdb_id'][:].astype(str),
            'ligand_resname': f['ligand_resname'][:].astype(str),
            'ligand_chain': f['ligand_chain'][:].astype(str),
            'ligand_resnum': f['ligand_resnum'][:],
            'graph_index': f['graph_index'][:],
            'src_index': f['src_index'][:],
            'dst_index': f['dst_index'][:],
        }
    
    # 读取三图 embedding
    with h5py.File(PROTEIN_EMBEDDINGS_H5, 'r') as f:
        prot_emb = f['embeddings'][:]
        prot_meta = {
            'pdb_id': f['pdb_id'][:].astype(str),
            'ligand_resname': f['ligand_resname'][:].astype(str),
            'ligand_chain': f['ligand_chain'][:].astype(str),
            'ligand_resnum': f['ligand_resnum'][:],
        }
    
    with h5py.File(LIGAND_EMBEDDINGS_H5, 'r') as f:
        lig_emb = f['embeddings'][:]
    
    with h5py.File(INTERACTION_EMBEDDINGS_H5, 'r') as f:
        inter_emb = f['embeddings'][:]
    
    # 构建样本 key -> index 映射
    sample_key_to_idx = {}
    for i in range(len(prot_meta['pdb_id'])):
        key = (
            prot_meta['pdb_id'][i],
            prot_meta['ligand_resname'][i],
            prot_meta['ligand_chain'][i],
            int(prot_meta['ligand_resnum'][i])
        )
        sample_key_to_idx[key] = i
    
    # 为每条边匹配对应的图级 embedding
    print("[Fusion] 匹配边级特征与图级 embedding...")
    fused_features_list = []
    
    for i in range(len(edge_meta['pdb_id'])):
        key = (
            edge_meta['pdb_id'][i],
            edge_meta['ligand_resname'][i],
            edge_meta['ligand_chain'][i],
            int(edge_meta['ligand_resnum'][i])
        )
        
        if key in sample_key_to_idx:
            idx = sample_key_to_idx[key]
            fused_feat = np.concatenate([
                edge_features[i],
                prot_emb[idx],
                lig_emb[idx],
                inter_emb[idx]
            ])
            fused_features_list.append(fused_feat)
        else:
            # 如果找不到对应样本，用零填充
            zero_pad = np.zeros(prot_emb.shape[1] + lig_emb.shape[1] + inter_emb.shape[1])
            fused_feat = np.concatenate([edge_features[i], zero_pad])
            fused_features_list.append(fused_feat)
    
    fused_features = np.array(fused_features_list, dtype=np.float32)
    
    # 保存融合后的特征
    with h5py.File(FUSED_EDGE_FEATURES_H5, 'w') as f:
        f.create_dataset('features', data=fused_features)
        
        # 保存元信息（字符串需要转换为字节类型）
        for key, val in edge_meta.items():
            if key in ['pdb_id', 'ligand_resname', 'ligand_chain']:
                # 字符串类型转换为字节
                val_bytes = np.array([s.encode('utf-8') for s in val], dtype='S')
                f.create_dataset(key, data=val_bytes)
            else:
                # 数值类型直接保存
                f.create_dataset(key, data=val)
        
        f.attrs['num_edges'] = len(fused_features)
        f.attrs['feature_dim'] = fused_features.shape[1]
        f.attrs['edge_feature_dim'] = edge_features.shape[1]
        f.attrs['protein_emb_dim'] = prot_emb.shape[1]
        f.attrs['ligand_emb_dim'] = lig_emb.shape[1]
        f.attrs['interaction_emb_dim'] = inter_emb.shape[1]
    
    print(f"\n已保存融合后的边级特征到 {FUSED_EDGE_FEATURES_H5}")
    print(f"  总特征维度: {fused_features.shape[1]}")
    print(f"    - 边级局部特征: {edge_features.shape[1]}")
    print(f"    - 蛋白 embedding: {prot_emb.shape[1]}")
    print(f"    - 配体 embedding: {lig_emb.shape[1]}")
    print(f"    - 相互作用 embedding: {inter_emb.shape[1]}")


# ============================================================
# Main 函数
# ============================================================

def main():
    """完整流水线：从 PDB 分析到边级特征融合。"""
    print("=" * 70)
    print("完整的蛋白-配体结合位点分析与特征提取流水线")
    print("=" * 70)
    
    # Step 1: 分析 PDB 复合物
    print("\n" + "=" * 70)
    print("Step 1/5: 分析 PDB 复合物，识别蛋白-配体接触")
    print("=" * 70)
    
    df_binding = analyze_all_pdbs(PDB_DIR)
    if df_binding.empty:
        print("未找到任何接触记录，退出。")
        return
    
    save_binding_sites_to_h5(df_binding, BINDING_SITES_H5)
    
    # Step 2: 分组
    binding_groups = group_binding_sites(df_binding)
    
    # Step 3: 计算三图 embedding
    print("\n" + "=" * 70)
    print("Step 2/5: 计算三图 embedding")
    print("=" * 70)
    compute_and_save_triplet_embeddings(PDB_DIR, binding_groups, max_groups=None)
    
    # Step 4: 计算边级局部特征
    print("\n" + "=" * 70)
    print("Step 3/5: 计算边级局部特征")
    print("=" * 70)
    compute_and_save_edge_features(PDB_DIR, binding_groups, max_groups=None)
    
    # Step 5: 融合四个文件
    print("\n" + "=" * 70)
    print("Step 4/5: 融合边级特征与图级 embedding")
    print("=" * 70)
    fuse_edge_and_graph_embeddings()
    
    print("\n" + "=" * 70)
    print("[SUCCESS] 完整流水线执行完成！")
    print("=" * 70)
    print(f"\n输出文件（HDF5 格式）：")
    print(f"  1. {BINDING_SITES_H5}")
    print(f"  2. {PROTEIN_EMBEDDINGS_H5}")
    print(f"  3. {LIGAND_EMBEDDINGS_H5}")
    print(f"  4. {INTERACTION_EMBEDDINGS_H5}")
    print(f"  5. {EDGE_FEATURES_H5}")
    print(f"  6. {FUSED_EDGE_FEATURES_H5}  ← 最终用于 VQ-VAE 训练")


if __name__ == "__main__":
    main()
