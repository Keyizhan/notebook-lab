"""Node feature computation functions."""
from typing import List, Union

import torch
import torch.nn.functional as F
from graphein.protein.tensor.angles import alpha, dihedrals, kappa
from graphein.protein.tensor.data import Protein, ProteinBatch
from graphein.protein.tensor.types import AtomTensor, CoordTensor
try:  # Optional dependency for Hydra-style configs
    from omegaconf import ListConfig  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when OmegaConf is unavailable
    ListConfig = list  # type: ignore
from torch_geometric.data import Batch, Data

from ..typecheck import jaxtyped, typechecker
from ..types import OrientationTensor, ScalarNodeFeature

from .sequence_features import amino_acid_one_hot
from .utils import _normalize
from .ligand_features import compute_ligand_atom_features, compute_ligand_vector_features


@jaxtyped(typechecker=typechecker)
def compute_scalar_node_features(
    x: Union[Batch, Data, Protein, ProteinBatch],
    node_features: Union[ListConfig, List[ScalarNodeFeature]],
) -> torch.Tensor:
    """
    Factory function for node features.
    
    Supports heterogeneous graphs with protein and ligand nodes.
    If x has node_type attribute, computes features separately for each type.

    .. seealso::
        :py:class:`models.gcpnet.types.ScalarNodeFeature` for a list of node
        features that can be computed.

    This function operates on a :py:class:`torch_geometric.data.Data` or
    :py:class:`torch_geometric.data.Batch` object and computes the requested
    node features.

    :param x: :py:class:`~torch_geometric.data.Data` or
        :py:class:`~torch_geometric.data.Batch` protein object.
    :type x: Union[Data, Batch]
    :param node_features: List of node features to compute.
    :type node_features: Union[List[str], ListConfig]
    :return: Tensor of node features of shape (``N x F``), where ``N`` is the
        number of nodes and ``F`` is the number of features.
    :rtype: torch.Tensor
    """
    # Check if this is a heterogeneous graph
    has_node_type = hasattr(x, 'node_type') and x.node_type is not None
    has_ligand = False
    if hasattr(x, 'num_ligand_nodes'):
        num_lig = x.num_ligand_nodes
        if isinstance(num_lig, (int, float)):
            has_ligand = num_lig > 0
        elif hasattr(num_lig, 'item'):
            has_ligand = num_lig.item() > 0 if num_lig.numel() == 1 else num_lig.sum().item() > 0
    
    if has_node_type and has_ligand:
        # Heterogeneous graph: compute features separately for protein and ligand
        return compute_heterogeneous_node_features(x, node_features)
    else:
        # Homogeneous protein graph: original logic
        feats = []
        for feature in node_features:
            if feature == "amino_acid_one_hot":
                feats.append(amino_acid_one_hot(x, num_classes=23))
            elif feature == "alpha":
                feats.append(alpha(x.coords, x.batch, rad=True, embed=True))
            elif feature == "kappa":
                feats.append(kappa(x.coords, x.batch, rad=True, embed=True))
            elif feature == "dihedrals":
                feats.append(dihedrals(x.coords, x.batch, rad=True, embed=True))
            elif feature == "sequence_positional_encoding":
                continue
            else:
                raise ValueError(f"Node feature {feature} not recognised.")
        feats = [feat.unsqueeze(1) if feat.ndim == 1 else feat for feat in feats]
        # Return concatenated features or original features if no features were computed
        return torch.cat(feats, dim=1) if feats else x.x


def compute_heterogeneous_node_features(
    x: Union[Batch, Data],
    node_features: Union[ListConfig, List[ScalarNodeFeature]],
) -> torch.Tensor:
    """
    Compute features for heterogeneous graphs with protein and ligand nodes.
    
    Parameters
    ----------
    x : Union[Batch, Data]
        Graph with node_type attribute (0=protein, 1=ligand)
    node_features : Union[ListConfig, List[ScalarNodeFeature]]
        List of features to compute for protein nodes
        
    Returns
    -------
    torch.Tensor
        Combined features for all nodes
    """
    # Handle both single graph and batch
    if hasattr(x.num_pocket_nodes, 'numel') and x.num_pocket_nodes.numel() > 1:
        # This is a batch - sum all pocket/ligand nodes
        num_pocket_nodes = x.num_pocket_nodes.sum().item()
        num_ligand_nodes = x.num_ligand_nodes.sum().item()
    else:
        num_pocket_nodes = x.num_pocket_nodes.item() if hasattr(x.num_pocket_nodes, 'item') else int(x.num_pocket_nodes)
        num_ligand_nodes = x.num_ligand_nodes.item() if hasattr(x.num_ligand_nodes, 'item') else int(x.num_ligand_nodes)
    total_nodes = num_pocket_nodes + num_ligand_nodes
    
    # Compute protein features (only for pocket nodes)
    protein_feats = []
    
    # Create a temporary object with only protein nodes for feature computation
    # We need to extract protein-specific attributes
    protein_coords = x.coords[:num_pocket_nodes] if hasattr(x, 'coords') else None
    protein_batch = x.batch[:num_pocket_nodes] if hasattr(x, 'batch') else None
    
    for feature in node_features:
        if feature == "amino_acid_one_hot":
            # Extract protein residue types
            protein_residue_type = x.residue_type[:num_pocket_nodes] if hasattr(x, 'residue_type') else None
            if protein_residue_type is not None:
                protein_feats.append(torch.nn.functional.one_hot(protein_residue_type, num_classes=23).float())
            else:
                protein_feats.append(torch.zeros((num_pocket_nodes, 23), device=x.x.device))
        elif feature == "alpha" and protein_coords is not None:
            protein_feats.append(alpha(protein_coords, protein_batch, rad=True, embed=True))
        elif feature == "kappa" and protein_coords is not None:
            protein_feats.append(kappa(protein_coords, protein_batch, rad=True, embed=True))
        elif feature == "dihedrals" and protein_coords is not None:
            protein_feats.append(dihedrals(protein_coords, protein_batch, rad=True, embed=True))
        elif feature == "sequence_positional_encoding":
            continue
    
    # Concatenate protein features
    if protein_feats:
        protein_feats = [feat.unsqueeze(1) if feat.ndim == 1 else feat for feat in protein_feats]
        protein_features = torch.cat(protein_feats, dim=1)
    else:
        # Fallback: use simple geometric features
        protein_features = torch.zeros((num_pocket_nodes, 6), device=x.x.device)
    
    # Compute ligand features
    ligand_features = compute_ligand_atom_features(
        x, 
        num_ligand_nodes=num_ligand_nodes,
        ligand_start_idx=num_pocket_nodes
    )
    
    # Ensure feature dimensions match
    protein_dim = protein_features.shape[1]
    ligand_dim = ligand_features.shape[1]
    
    # Target dimension should match protein features for GCPNet compatibility
    target_dim = max(protein_dim, ligand_dim, 49)  # GCPNet expects at least 49 dims
    
    # Pad both to target dimension
    if protein_dim < target_dim:
        padding = torch.zeros((num_pocket_nodes, target_dim - protein_dim), device=protein_features.device)
        protein_features = torch.cat([protein_features, padding], dim=1)
    
    if ligand_dim < target_dim:
        padding = torch.zeros((num_ligand_nodes, target_dim - ligand_dim), device=ligand_features.device)
        ligand_features = torch.cat([ligand_features, padding], dim=1)
    
    # Concatenate protein and ligand features
    combined_features = torch.cat([protein_features, ligand_features], dim=0)
    
    return combined_features


@jaxtyped(typechecker=typechecker)
def compute_vector_node_features(
    x: Union[Batch, Data, Protein, ProteinBatch],
    vector_features: Union[ListConfig, List[str]],
) -> Union[Batch, Data, Protein, ProteinBatch]:
    """Factory function for vector features.
    
    Supports heterogeneous graphs with protein and ligand nodes.

    Currently implemented vector features are:

        - ``orientation``: Orientation of each node in the protein backbone
        - ``virtual_cb_vector``: Virtual CB vector for each node in the protein
        backbone


    """
    # Check if this is a heterogeneous graph
    has_node_type = hasattr(x, 'node_type') and x.node_type is not None
    has_ligand = False
    if hasattr(x, 'num_ligand_nodes'):
        num_lig = x.num_ligand_nodes
        if isinstance(num_lig, (int, float)):
            has_ligand = num_lig > 0
        elif hasattr(num_lig, 'item'):
            has_ligand = num_lig.item() > 0 if num_lig.numel() == 1 else num_lig.sum().item() > 0
    
    if has_node_type and has_ligand:
        # Heterogeneous graph: compute features separately
        # Handle both single graph and batch
        if hasattr(x.num_pocket_nodes, 'numel') and x.num_pocket_nodes.numel() > 1:
            num_pocket_nodes = x.num_pocket_nodes.sum().item()
            num_ligand_nodes = x.num_ligand_nodes.sum().item()
        else:
            num_pocket_nodes = x.num_pocket_nodes.item() if hasattr(x.num_pocket_nodes, 'item') else int(x.num_pocket_nodes)
            num_ligand_nodes = x.num_ligand_nodes.item() if hasattr(x.num_ligand_nodes, 'item') else int(x.num_ligand_nodes)
        
        # Compute protein vector features
        # For heterogeneous graphs, we need to handle coords carefully
        # x.coords might be (N, 4, 3) for backbone atoms
        if hasattr(x, 'coords') and x.coords.ndim == 3:
            # Extract CA coordinates for protein nodes
            protein_ca_coords = x.coords[:num_pocket_nodes, 1, :]  # (num_pocket, 3) - CA only
        elif hasattr(x, 'x'):
            # Fallback: use x as coordinates
            protein_ca_coords = x.x[:num_pocket_nodes]  # (num_pocket, 3)
        else:
            protein_ca_coords = torch.zeros((num_pocket_nodes, 3), device=x.x.device)
        
        # Compute orientations manually for protein nodes
        if num_pocket_nodes > 1:
            forward_vec = protein_ca_coords[1:] - protein_ca_coords[:-1]  # (num_pocket-1, 3)
            backward_vec = protein_ca_coords[:-1] - protein_ca_coords[1:]  # (num_pocket-1, 3)
            
            # Normalize
            forward_vec = torch.nn.functional.normalize(forward_vec, p=2, dim=1)
            backward_vec = torch.nn.functional.normalize(backward_vec, p=2, dim=1)
            
            # Pad to match num_pocket_nodes
            forward_vec = torch.nn.functional.pad(forward_vec, [0, 0, 0, 1])  # (num_pocket, 3)
            backward_vec = torch.nn.functional.pad(backward_vec, [0, 0, 1, 0])  # (num_pocket, 3)
            
            protein_vectors = torch.stack([forward_vec, backward_vec], dim=1)  # (num_pocket, 2, 3)
        else:
            protein_vectors = torch.zeros((num_pocket_nodes, 2, 3), device=x.x.device)
        
        # Compute ligand vector features
        ligand_vectors = compute_ligand_vector_features(
            x,
            num_ligand_nodes=num_ligand_nodes,
            ligand_start_idx=num_pocket_nodes
        )
        
        # Debug: print shapes
        # print(f"DEBUG: protein_vectors shape: {protein_vectors.shape}")
        # print(f"DEBUG: ligand_vectors shape: {ligand_vectors.shape}")
        
        # Concatenate along node dimension (dim=0)
        x.x_vector_attr = torch.cat([protein_vectors, ligand_vectors], dim=0)
    else:
        # Homogeneous protein graph: original logic
        vector_node_features = []
        for feature in vector_features:
            if feature == "orientation":
                vector_node_features.append(orientations(x.coords, x._slice_dict["coords"]))
            elif feature == "virtual_cb_vector":
                raise NotImplementedError("Virtual CB vector not implemented yet.")
            else:
                raise ValueError(f"Vector feature {feature} not recognised.")
        x.x_vector_attr = torch.cat(vector_node_features, dim=0)
    
    return x


@jaxtyped(typechecker=typechecker)
def orientations(
    X: Union[CoordTensor, AtomTensor], coords_slice_index: torch.Tensor, ca_idx: int = 1
) -> OrientationTensor:
    if X.ndim == 3:
        X = X[:, ca_idx, :]

    # NOTE: the first item in the coordinates slice index is always 0,
    # and the last item is always the node count of the batch
    batch_num_nodes = X.shape[0]
    slice_index = coords_slice_index[1:] - 1
    last_node_index = slice_index[:-1]
    first_node_index = slice_index[:-1] + 1

    # NOTE: all of the last (first) nodes in a subgraph have their
    # forward (backward) vectors set to a padding value (i.e., 0.0)
    # to mimic feature construction behavior with single input graphs
    forward_slice = X[1:] - X[:-1]
    backward_slice = X[:-1] - X[1:]

    if forward_slice.numel() > 0 and last_node_index.numel() > 0:
        max_forward_idx = forward_slice.size(0) - 1
        # zero the forward vectors for last nodes in each subgraph without boolean masks (torch.compile friendly)
        valid_forward_idx = last_node_index.clamp_min(0).clamp_max(max_forward_idx).to(X.device)
        forward_slice.index_fill_(0, valid_forward_idx, 0.0)

    if backward_slice.numel() > 0 and first_node_index.numel() > 0:
        max_backward_idx = backward_slice.size(0) - 1
        # zero the backward vectors for first nodes in each subgraph
        valid_backward_idx = (first_node_index - 1).clamp_min(0).clamp_max(max_backward_idx).to(X.device)
        backward_slice.index_fill_(0, valid_backward_idx, 0.0)

    # NOTE: padding first and last nodes with zero vectors does not impact feature normalization
    forward = _normalize(forward_slice)
    backward = _normalize(backward_slice)
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    orientations = torch.cat((forward.unsqueeze(-2), backward.unsqueeze(-2)), dim=-2)

    # optionally debug/verify the orientations
    # last_node_indices = torch.cat((last_node_index, torch.tensor([batch_num_nodes - 1])), dim=0)
    # first_node_indices = torch.cat((torch.tensor([0]), first_node_index), dim=0)
    # intermediate_node_indices_mask = torch.ones(batch_num_nodes, device=X.device, dtype=torch.bool)
    # intermediate_node_indices_mask[last_node_indices] = False
    # intermediate_node_indices_mask[first_node_indices] = False
    # assert not orientations[last_node_indices][:, 0].any() and orientations[last_node_indices][:, 1].any()
    # assert orientations[first_node_indices][:, 0].any() and not orientations[first_node_indices][:, 1].any()
    # assert orientations[intermediate_node_indices_mask][:, 0].any() and orientations[intermediate_node_indices_mask][:, 1].any()

    return orientations
