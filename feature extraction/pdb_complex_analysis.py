from pathlib import Path
import math
import csv

from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa


# 距离阈值（Å），用于定义“接触/关键位点”
DIST_CUTOFF = 4.0

# 忽略的 HET 残基名（溶剂、简单离子等）
IGNORED_HET = {
    "HOH", "WAT",  # 水
    "NA", "K", "CL", "MG", "CA", "ZN", "FE", "MN", "CU",
    "SO4", "PO4", "IOD", "GOL", "PEG",
}

# 如果你已有确定的受体链，可以在这里设置，例如 ["A"]
# 如果 None 或空列表，则分析所有蛋白链
RECEPTOR_CHAINS: list[str] = []  # 例如 ["A"] 或 ["A", "B"]


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


def compute_contacts_for_structure(pdb_path: Path) -> list[dict]:
    """对单个 PDB 文件计算蛋白残基与配体残基的最小原子-原子距离。"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))

    protein_residues, ligand_residues = split_protein_and_ligands(structure)

    if not protein_residues or not ligand_residues:
        return []

    all_atoms = [atom for atom in structure.get_atoms()]
    _ = NeighborSearch(all_atoms)  # 保留以便后续如需优化

    records: list[dict] = []

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


def analyze_all_pdbs(pdb_dir: Path, output_csv: Path) -> None:
    """遍历目录下所有 .pdb 文件并导出 binding_sites.csv。"""
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    print(f"Found {len(pdb_files)} PDB files under {pdb_dir}")

    all_records: list[dict] = []
    for i, pdb_path in enumerate(pdb_files, 1):
        print(f"[{i}/{len(pdb_files)}] Processing {pdb_path.name} ...", end="\r")
        try:
            recs = compute_contacts_for_structure(pdb_path)
            all_records.extend(recs)
        except Exception as e:  # noqa: BLE001
            print(f"\nError processing {pdb_path.name}: {e}")

    print(f"\nTotal contact records: {len(all_records)}")

    if not all_records:
        print("No contacts found. Check parameters (DIST_CUTOFF, IGNORED_HET, RECEPTOR_CHAINS).")
        return

    fieldnames = list(all_records[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"Saved results to: {output_csv}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    pdb_dir = base_dir / "complex-20251129T063258Z-1-001" / "complex"
    output_csv = base_dir / "binding_sites.csv"
    analyze_all_pdbs(pdb_dir, output_csv)


if __name__ == "__main__":
    main()
