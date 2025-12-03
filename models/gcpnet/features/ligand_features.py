"""Ligand atom feature computation functions."""
from typing import Union

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from ..typecheck import jaxtyped, typechecker


# Common atom types in small molecules
ATOM_TYPES = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H', 'UNK']
ATOM_TYPE_TO_IDX = {atom: idx for idx, atom in enumerate(ATOM_TYPES)}


@jaxtyped(typechecker=typechecker)
def compute_ligand_atom_features(
    x: Union[Batch, Data],
    num_ligand_nodes: int,
    ligand_start_idx: int,
) -> torch.Tensor:
    """
    Compute scalar features for ligand atoms.
    
    For now, we use a simple placeholder approach:
    - Extract ligand node positions
    - Create basic geometric features (distances to centroid, etc.)
    - Pad to match protein feature dimension
    
    Parameters
    ----------
    x : Union[Batch, Data]
        Graph batch containing heterogeneous nodes
    num_ligand_nodes : int
        Number of ligand nodes
    ligand_start_idx : int
        Starting index of ligand nodes in the graph
        
    Returns
    -------
    torch.Tensor
        Ligand atom features of shape (num_ligand_nodes, feature_dim)
    """
    if num_ligand_nodes == 0:
        return torch.zeros((0, 6), dtype=torch.float32, device=x.x.device)
    
    # Extract ligand positions - use the original position attribute
    if hasattr(x, 'pos') and x.pos is not None:
        ligand_pos = x.pos[ligand_start_idx:ligand_start_idx + num_ligand_nodes]  # (Q, 3)
    else:
        # Fallback: assume x.x contains coordinates (first 3 dimensions)
        ligand_features = x.x[ligand_start_idx:ligand_start_idx + num_ligand_nodes]
        if ligand_features.shape[1] >= 3:
            ligand_pos = ligand_features[:, :3]  # (Q, 3)
        else:
            # Can't extract positions, return zeros
            return torch.zeros((num_ligand_nodes, 6), dtype=torch.float32, device=x.x.device)
    
    # Compute centroid
    centroid = ligand_pos.mean(dim=0, keepdim=True)  # (1, 3)
    
    # Distance to centroid
    dist_to_centroid = torch.norm(ligand_pos - centroid, dim=1, keepdim=True)  # (Q, 1)
    
    # Normalized position relative to centroid
    rel_pos = ligand_pos - centroid  # (Q, 3)
    
    # Pairwise distances (mean distance to other ligand atoms)
    if num_ligand_nodes > 1:
        pairwise_dist = torch.cdist(ligand_pos, ligand_pos)  # (Q, Q)
        # Exclude self-distances
        mask = ~torch.eye(num_ligand_nodes, dtype=torch.bool, device=pairwise_dist.device)
        mean_pairwise_dist = (pairwise_dist * mask).sum(dim=1, keepdim=True) / (num_ligand_nodes - 1)
    else:
        mean_pairwise_dist = torch.zeros((num_ligand_nodes, 1), device=ligand_pos.device)
    
    # Concatenate features: [dist_to_centroid(1), rel_pos(3), mean_pairwise_dist(1), padding(1)]
    # Total: 6 features to match typical protein feature dimension
    padding = torch.zeros((num_ligand_nodes, 1), device=ligand_pos.device)
    
    ligand_features = torch.cat([
        dist_to_centroid,
        rel_pos,
        mean_pairwise_dist,
        padding
    ], dim=1)  # (Q, 6)
    
    return ligand_features


@jaxtyped(typechecker=typechecker)
def compute_ligand_vector_features(
    x: Union[Batch, Data],
    num_ligand_nodes: int,
    ligand_start_idx: int,
) -> torch.Tensor:
    """
    Compute vector features for ligand atoms.
    
    Creates directional features based on ligand geometry.
    
    Parameters
    ----------
    x : Union[Batch, Data]
        Graph batch containing heterogeneous nodes
    num_ligand_nodes : int
        Number of ligand nodes
    ligand_start_idx : int
        Starting index of ligand nodes
        
    Returns
    -------
    torch.Tensor
        Ligand vector features of shape (num_ligand_nodes, num_vectors, 3)
    """
    if num_ligand_nodes == 0:
        return torch.zeros((0, 2, 3), dtype=torch.float32, device=x.x.device)
    
    # Extract ligand positions - use the original position attribute
    # x.x might contain features, not coordinates
    if hasattr(x, 'pos') and x.pos is not None:
        # Use pos if available (torch_geometric convention)
        ligand_pos = x.pos[ligand_start_idx:ligand_start_idx + num_ligand_nodes]  # (Q, 3)
    else:
        # Fallback: assume x.x contains coordinates (first 3 dimensions)
        ligand_features = x.x[ligand_start_idx:ligand_start_idx + num_ligand_nodes]
        if ligand_features.shape[1] >= 3:
            ligand_pos = ligand_features[:, :3]  # (Q, 3)
        else:
            # Can't extract positions, return zeros
            return torch.zeros((num_ligand_nodes, 2, 3), dtype=torch.float32, device=x.x.device)
    
    # Compute centroid
    centroid = ligand_pos.mean(dim=0, keepdim=True)  # (1, 3)
    
    # Vector to centroid (normalized)
    vec_to_centroid = centroid - ligand_pos  # (Q, 3)
    vec_to_centroid = F.normalize(vec_to_centroid, p=2, dim=1)
    
    # Vector to nearest neighbor
    if num_ligand_nodes > 1:
        pairwise_dist = torch.cdist(ligand_pos, ligand_pos)  # (Q, Q)
        # Mask self-distances
        pairwise_dist = pairwise_dist + torch.eye(num_ligand_nodes, device=pairwise_dist.device) * 1e6
        nearest_idx = pairwise_dist.argmin(dim=1)  # (Q,)
        nearest_pos = ligand_pos[nearest_idx]  # (Q, 3)
        vec_to_nearest = nearest_pos - ligand_pos  # (Q, 3)
        vec_to_nearest = F.normalize(vec_to_nearest, p=2, dim=1)
    else:
        vec_to_nearest = torch.zeros_like(ligand_pos)
    
    # Stack vectors: [vec_to_centroid, vec_to_nearest]
    ligand_vectors = torch.stack([vec_to_centroid, vec_to_nearest], dim=1)  # (Q, 2, 3)
    
    return ligand_vectors


def get_ligand_feature_dim(protein_feature_dim: int = 6) -> int:
    """
    Get the feature dimension for ligand atoms to match protein features.
    
    Parameters
    ----------
    protein_feature_dim : int
        Dimension of protein node features
        
    Returns
    -------
    int
        Dimension of ligand node features (same as protein)
    """
    return protein_feature_dim
