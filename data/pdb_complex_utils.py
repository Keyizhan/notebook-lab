import math
import os
from typing import Dict, List, Tuple

import numpy as np


STANDARD_AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "SEC", "PYL",
}

# Common solvent/ions that should not be treated as ligands
NON_LIGAND_HET = {
    "HOH", "WAT", "DOD", "NA", "K", "CL", "CA", "MG", "ZN", "SO4", "PO4",
    "GOL", "EDO", "PEG", "MPD",
}


def _is_protein_resname(resname: str) -> bool:
    return resname.strip().upper() in STANDARD_AA3


def _is_non_ligand_het(resname: str) -> bool:
    return resname.strip().upper() in NON_LIGAND_HET


def parse_protein_ligand_from_pdb(pdb_path: str) -> Dict[str, object]:
    """Lightweight PDB parser to separate protein residues and ligand atoms.

    Returns a dict with:
      - protein_residues: List[dict] with keys {chain_id, resseq, icode, resname,
        atoms: {atom_name -> (x,y,z)}, is_protein: bool}
      - ligand_atoms: List[dict] with keys {chain_id, resseq, icode, resname,
        atom_name, coord(np.array shape (3,))}
    """
    protein_residues: Dict[Tuple[str, int, str], Dict[str, object]] = {}
    ligand_atoms: List[Dict[str, object]] = []

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(pdb_path)

    with open(pdb_path, "r") as fh:
        for line in fh:
            record = line[0:6]
            if record not in ("ATOM  ", "HETATM"):
                continue

            resname = line[17:20].strip()
            chain_id = line[21].strip() or "A"
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26].strip()
            atom_name = line[12:16].strip()

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            key = (chain_id, resseq, icode)
            is_protein = _is_protein_resname(resname)

            if is_protein:
                if key not in protein_residues:
                    protein_residues[key] = {
                        "chain_id": chain_id,
                        "resseq": resseq,
                        "icode": icode,
                        "resname": resname,
                        "atoms": {},
                    }
                protein_residues[key]["atoms"][atom_name] = np.array([x, y, z], dtype=np.float32)
            else:
                if _is_non_ligand_het(resname):
                    continue
                ligand_atoms.append(
                    {
                        "chain_id": chain_id,
                        "resseq": resseq,
                        "icode": icode,
                        "resname": resname,
                        "atom_name": atom_name,
                        "coord": np.array([x, y, z], dtype=np.float32),
                    }
                )

    residues_list: List[Dict[str, object]] = [protein_residues[k] for k in sorted(protein_residues.keys())]
    return {"protein_residues": residues_list, "ligand_atoms": ligand_atoms}


def build_backbone_from_residues(residues: List[Dict[str, object]]) -> Tuple[str, np.ndarray]:
    """Build sequence and backbone coordinates (N, CA, C, O) from parsed residues.

    Returns
    -------
    seq : str
        One-letter amino acid sequence (unknown residues as 'X').
    coords : np.ndarray
        Shape (L, 4, 3) backbone coordinates; missing atoms are set to NaN.
    """
    from graphein.protein.resi_atoms import STANDARD_AMINO_ACID_MAPPING_1_TO_3

    three_to_one = {v: k for k, v in STANDARD_AMINO_ACID_MAPPING_1_TO_3.items()}

    seq_chars: List[str] = []
    coords = []
    for res in residues:
        resname = res["resname"].upper()
        atoms = res["atoms"]
        one = three_to_one.get(resname, "X")
        seq_chars.append(one)
        res_coords = []
        for atom_name in ("N", "CA", "C", "O"):
            if atom_name in atoms:
                res_coords.append(atoms[atom_name])
            else:
                res_coords.append(np.full(3, np.nan, dtype=np.float32))
        coords.append(res_coords)

    coords_arr = np.asarray(coords, dtype=np.float32)
    return "".join(seq_chars), coords_arr


def detect_pocket_and_interactions(
    residues: List[Dict[str, object]],
    ligand_atoms: List[Dict[str, object]],
    cutoff: float = 5.0,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Detect pocket residues and protein-ligand contacts within cutoff.

    Parameters
    ----------
    residues : list
        Parsed protein residues.
    ligand_atoms : list
        Parsed ligand atoms.
    cutoff : float
        Distance cutoff in Å.

    Returns
    -------
    pocket_mask : np.ndarray
        Boolean array of shape (L,) marking pocket residues.
    contacts : list of (res_idx, lig_atom_idx)
        Indices of contacting residue and ligand atom.
    """
    L = len(residues)
    pocket_mask = np.zeros(L, dtype=bool)
    contacts: List[Tuple[int, int]] = []

    if L == 0 or not ligand_atoms:
        return pocket_mask, contacts

    prot_coords = []
    res_indices = []
    for i, res in enumerate(residues):
        for atom_name, coord in res["atoms"].items():
            if atom_name.upper().startswith("H"):
                continue
            prot_coords.append(coord)
            res_indices.append(i)

    if not prot_coords:
        return pocket_mask, contacts

    prot_arr = np.asarray(prot_coords, dtype=np.float32)  # (P,3)
    lig_arr = np.asarray([a["coord"] for a in ligand_atoms], dtype=np.float32)  # (Q,3)

    diff = prot_arr[:, None, :] - lig_arr[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    cut2 = cutoff * cutoff

    P, Q = dist2.shape
    for p in range(P):
        for q in range(Q):
            if dist2[p, q] <= cut2:
                res_idx = res_indices[p]
                pocket_mask[res_idx] = True
                contacts.append((res_idx, q))

    return pocket_mask, contacts


def build_ligand_graph(ligand_atoms: List[Dict[str, object]], cutoff: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Build ligand internal graph using distance-based connectivity.

    Parameters
    ----------
    ligand_atoms : list
        Parsed ligand atoms.
    cutoff : float
        Distance cutoff in Å for ligand internal edges (default 2.0Å for covalent-like).

    Returns
    -------
    ligand_coords : np.ndarray
        Shape (Q, 3) ligand atom coordinates.
    ligand_edges : np.ndarray
        Shape (2, E_lig) edge indices for ligand internal connectivity.
    """
    if not ligand_atoms:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((2, 0), dtype=np.int64)

    coords = np.asarray([a["coord"] for a in ligand_atoms], dtype=np.float32)
    Q = coords.shape[0]

    # Compute pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))

    # Build edges: i < j and dist <= cutoff
    edges = []
    for i in range(Q):
        for j in range(i + 1, Q):
            if dist[i, j] <= cutoff:
                edges.append([i, j])
                edges.append([j, i])  # undirected

    if not edges:
        # Fallback: use kNN if no edges found
        k = min(4, Q - 1)
        if k > 0:
            for i in range(Q):
                dists_i = dist[i].copy()
                dists_i[i] = np.inf
                nearest = np.argsort(dists_i)[:k]
                for j in nearest:
                    edges.append([i, int(j)])

    if edges:
        edge_index = np.asarray(edges, dtype=np.int64).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return coords, edge_index
