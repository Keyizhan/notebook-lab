"""
PDB蛋白质-配体复合物分析工具
===================================
用于分析YASARA生成的蛋白质-配体对接结果PDB文件

作者: PDB分析工具
日期: 2024
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Atom:
    """原子数据类"""
    serial: int          # 原子序号
    name: str           # 原子名称
    resname: str        # 残基名称
    chain: str          # 链标识
    resseq: int         # 残基序号
    x: float            # X坐标
    y: float            # Y坐标
    z: float            # Z坐标
    occupancy: float    # 占有率
    tempfactor: float   # 温度因子
    element: str        # 元素符号
    record_type: str    # 记录类型 (ATOM或HETATM)


@dataclass
class Bond:
    """化学键数据类"""
    atom1: int
    atom2: int
    bond_order: int = 1  # 键级 (1=单键, 2=双键, 3=三键)


class PDBComplexAnalyzer:
    """PDB复合物分析器"""
    
    def __init__(self, pdb_file: str):
        """
        初始化分析器
        
        参数:
            pdb_file: PDB文件路径
        """
        self.pdb_file = pdb_file
        self.filename = Path(pdb_file).name
        self.protein_atoms: List[Atom] = []
        self.ligand_atoms: List[Atom] = []
        self.bonds: List[Bond] = []
        self.metadata: Dict = {}
        self.sequence: Dict[str, List[str]] = defaultdict(list)
        
    def parse(self):
        """解析PDB文件"""
        with open(self.pdb_file, 'r', encoding='utf-8') as f:
            for line in f:
                record_type = line[0:6].strip()
                
                if record_type == 'REMARK':
                    self._parse_remark(line)
                elif record_type == 'SEQRES':
                    self._parse_seqres(line)
                elif record_type == 'ATOM':
                    self.protein_atoms.append(self._parse_atom(line, 'ATOM'))
                elif record_type == 'HETATM':
                    self.ligand_atoms.append(self._parse_atom(line, 'HETATM'))
                elif record_type == 'CONECT':
                    self._parse_conect(line)
                elif record_type == 'COMPND':
                    self.metadata['compound'] = line[10:].strip()
                    
    def _parse_remark(self, line: str):
        """解析REMARK记录"""
        if 'Number of atoms' in line:
            match = re.search(r'(\d+)', line)
            if match:
                self.metadata['n_atoms'] = int(match.group(1))
        elif 'active torsions' in line:
            match = re.search(r'(\d+)', line)
            if match:
                self.metadata['n_torsions'] = int(match.group(1))
                
    def _parse_seqres(self, line: str):
        """解析SEQRES序列记录"""
        chain = line[11].strip()
        residues = line[19:].split()
        self.sequence[chain].extend(residues)
        
    def _parse_atom(self, line: str, record_type: str) -> Atom:
        """
        解析ATOM/HETATM记录
        
        PDB格式说明:
        列1-6:   记录类型 (ATOM/HETATM)
        列7-11:  原子序号
        列13-16: 原子名称
        列18-20: 残基名称
        列22:    链标识
        列23-26: 残基序号
        列31-38: X坐标
        列39-46: Y坐标
        列47-54: Z坐标
        列55-60: 占有率
        列61-66: 温度因子
        列77-78: 元素符号
        """
        return Atom(
            serial=int(line[6:11].strip()),
            name=line[12:16].strip(),
            resname=line[17:20].strip(),
            chain=line[21].strip(),
            resseq=int(line[22:26].strip()) if line[22:26].strip() else 0,
            x=float(line[30:38].strip()),
            y=float(line[38:46].strip()),
            z=float(line[46:54].strip()),
            occupancy=float(line[54:60].strip()),
            tempfactor=float(line[60:66].strip()),
            element=line[76:78].strip() if len(line) > 76 else '',
            record_type=record_type
        )
        
    def _parse_conect(self, line: str):
        """解析CONECT连接记录"""
        atoms = [int(x) for x in line[6:].split()]
        if len(atoms) >= 2:
            atom1 = atoms[0]
            for atom2 in atoms[1:]:
                # 计算键级（出现次数）
                bond_order = atoms[1:].count(atom2)
                # 避免重复添加
                if not any(b.atom1 == atom1 and b.atom2 == atom2 for b in self.bonds):
                    self.bonds.append(Bond(atom1, atom2, bond_order))
    
    def calculate_distance(self, atom1: Atom, atom2: Atom) -> float:
        """计算两个原子之间的距离"""
        return np.sqrt(
            (atom1.x - atom2.x)**2 + 
            (atom1.y - atom2.y)**2 + 
            (atom1.z - atom2.z)**2
        )
    
    def find_interactions(self, distance_cutoff: float = 4.0) -> List[Tuple[Atom, Atom, float]]:
        """
        查找蛋白质与配体之间的相互作用
        
        参数:
            distance_cutoff: 距离阈值（埃）
            
        返回:
            相互作用列表 [(蛋白原子, 配体原子, 距离)]
        """
        interactions = []
        for p_atom in self.protein_atoms:
            for l_atom in self.ligand_atoms:
                dist = self.calculate_distance(p_atom, l_atom)
                if dist <= distance_cutoff:
                    interactions.append((p_atom, l_atom, dist))
        return sorted(interactions, key=lambda x: x[2])
    
    def get_binding_residues(self, distance_cutoff: float = 4.0) -> Dict[int, Dict]:
        """
        获取与配体结合的蛋白质残基
        
        参数:
            distance_cutoff: 距离阈值（埃）
            
        返回:
            结合残基字典 {残基序号: {信息}}
        """
        interactions = self.find_interactions(distance_cutoff)
        binding_residues = {}
        
        for p_atom, l_atom, dist in interactions:
            if p_atom.resseq not in binding_residues:
                binding_residues[p_atom.resseq] = {
                    'resname': p_atom.resname,
                    'chain': p_atom.chain,
                    'min_distance': dist,
                    'contacts': []
                }
            else:
                binding_residues[p_atom.resseq]['min_distance'] = min(
                    binding_residues[p_atom.resseq]['min_distance'], dist
                )
            
            binding_residues[p_atom.resseq]['contacts'].append({
                'protein_atom': p_atom.name,
                'ligand_atom': l_atom.name,
                'distance': dist
            })
        
        return binding_residues
    
    def calculate_center_of_mass(self, atoms: List[Atom]) -> Tuple[float, float, float]:
        """计算原子的质心"""
        if not atoms:
            return (0.0, 0.0, 0.0)
        
        x = sum(atom.x for atom in atoms) / len(atoms)
        y = sum(atom.y for atom in atoms) / len(atoms)
        z = sum(atom.z for atom in atoms) / len(atoms)
        return (x, y, z)
    
    def analyze_ligand_composition(self) -> Dict[str, int]:
        """分析配体的元素组成"""
        composition = defaultdict(int)
        for atom in self.ligand_atoms:
            element = atom.element if atom.element else atom.name[0]
            composition[element] += 1
        return dict(composition)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        protein_com = self.calculate_center_of_mass(self.protein_atoms)
        ligand_com = self.calculate_center_of_mass(self.ligand_atoms)
        
        # 计算蛋白质和配体质心之间的距离
        com_distance = np.sqrt(
            (protein_com[0] - ligand_com[0])**2 +
            (protein_com[1] - ligand_com[1])**2 +
            (protein_com[2] - ligand_com[2])**2
        )
        
        return {
            'filename': self.filename,
            'n_protein_atoms': len(self.protein_atoms),
            'n_ligand_atoms': len(self.ligand_atoms),
            'n_bonds': len(self.bonds),
            'protein_com': protein_com,
            'ligand_com': ligand_com,
            'com_distance': com_distance,
            'sequence_length': sum(len(seq) for seq in self.sequence.values()),
            'ligand_composition': self.analyze_ligand_composition(),
            **self.metadata
        }
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        生成详细的分析报告
        
        参数:
            output_file: 输出文件路径（可选）
        """
        stats = self.get_statistics()
        binding_residues = self.get_binding_residues()
        
        report_lines = [
            "=" * 80,
            f"PDB复合物分析报告: {stats['filename']}",
            "=" * 80,
            "",
            "【基本信息】",
            f"  文件名: {stats['filename']}",
            f"  蛋白质原子数: {stats['n_protein_atoms']}",
            f"  配体原子数: {stats['n_ligand_atoms']}",
            f"  化学键数: {stats['n_bonds']}",
            f"  序列长度: {stats['sequence_length']} 残基",
            "",
            "【配体信息】",
            f"  元素组成: {stats['ligand_composition']}",
            f"  配体质心坐标: ({stats['ligand_com'][0]:.3f}, {stats['ligand_com'][1]:.3f}, {stats['ligand_com'][2]:.3f})",
            "",
            "【蛋白质信息】",
            f"  蛋白质质心坐标: ({stats['protein_com'][0]:.3f}, {stats['protein_com'][1]:.3f}, {stats['protein_com'][2]:.3f})",
            f"  蛋白-配体质心距离: {stats['com_distance']:.3f} Å",
            "",
            "【结合位点分析】",
            f"  结合残基数量: {len(binding_residues)}",
            ""
        ]
        
        if binding_residues:
            report_lines.append("  结合残基列表（按距离排序）:")
            sorted_residues = sorted(
                binding_residues.items(),
                key=lambda x: x[1]['min_distance']
            )
            
            for resseq, info in sorted_residues[:20]:  # 显示前20个
                report_lines.append(
                    f"    {info['resname']:>3} {resseq:>4} (链 {info['chain']}) - "
                    f"最近距离: {info['min_distance']:.2f} Å, "
                    f"接触数: {len(info['contacts'])}"
                )
        
        report_lines.extend([
            "",
            "【序列信息】"
        ])
        
        for chain, seq in self.sequence.items():
            report_lines.append(f"  链 {chain}: {' '.join(seq[:10])}... (共{len(seq)}个残基)")
        
        report_lines.extend([
            "",
            "=" * 80,
            ""
        ])
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


class BatchPDBAnalyzer:
    """批量PDB文件分析器"""
    
    def __init__(self, directory: str):
        """
        初始化批量分析器
        
        参数:
            directory: 包含PDB文件的目录
        """
        self.directory = Path(directory)
        self.analyzers: List[PDBComplexAnalyzer] = []
        
    def load_all_pdbs(self, pattern: str = "*.pdb"):
        """加载所有PDB文件"""
        pdb_files = sorted(self.directory.glob(pattern))
        print(f"找到 {len(pdb_files)} 个PDB文件")
        
        for pdb_file in pdb_files:
            print(f"正在加载: {pdb_file.name}")
            analyzer = PDBComplexAnalyzer(str(pdb_file))
            analyzer.parse()
            self.analyzers.append(analyzer)
        
        print(f"成功加载 {len(self.analyzers)} 个PDB文件")
        
    def compare_binding_sites(self) -> Dict:
        """比较所有复合物的结合位点"""
        all_binding_residues = {}
        
        for analyzer in self.analyzers:
            binding_res = analyzer.get_binding_residues()
            all_binding_residues[analyzer.filename] = set(binding_res.keys())
        
        # 找出共同的结合残基
        if all_binding_residues:
            common_residues = set.intersection(*all_binding_residues.values())
        else:
            common_residues = set()
        
        return {
            'all_binding_residues': all_binding_residues,
            'common_residues': common_residues,
            'n_files': len(self.analyzers)
        }
    
    def generate_summary_table(self, output_file: str = "summary.txt"):
        """生成汇总表"""
        with open(output_file, 'w', encoding='utf-8') as f:
            # 表头
            f.write(f"{'文件名':<30} {'蛋白原子':<12} {'配体原子':<12} {'结合残基':<12} {'质心距离':<12}\n")
            f.write("-" * 90 + "\n")
            
            # 数据行
            for analyzer in self.analyzers:
                stats = analyzer.get_statistics()
                binding_res = analyzer.get_binding_residues()
                
                f.write(
                    f"{stats['filename']:<30} "
                    f"{stats['n_protein_atoms']:<12} "
                    f"{stats['n_ligand_atoms']:<12} "
                    f"{len(binding_res):<12} "
                    f"{stats['com_distance']:<12.2f}\n"
                )
    
    def plot_statistics(self, output_file: str = "statistics.png"):
        """绘制统计图表"""
        if not self.analyzers:
            print("没有数据可以绘制")
            return
        
        # 收集数据
        filenames = []
        com_distances = []
        n_binding_residues = []
        
        for analyzer in self.analyzers[:30]:  # 最多显示30个
            stats = analyzer.get_statistics()
            binding_res = analyzer.get_binding_residues()
            
            filenames.append(stats['filename'].replace('.pdb', ''))
            com_distances.append(stats['com_distance'])
            n_binding_residues.append(len(binding_res))
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 子图1: 质心距离
        ax1.bar(range(len(filenames)), com_distances, color='steelblue', alpha=0.7)
        ax1.set_xlabel('PDB文件', fontsize=12)
        ax1.set_ylabel('质心距离 (Å)', fontsize=12)
        ax1.set_title('蛋白-配体质心距离分布', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(filenames)))
        ax1.set_xticklabels(filenames, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # 子图2: 结合残基数
        ax2.bar(range(len(filenames)), n_binding_residues, color='coral', alpha=0.7)
        ax2.set_xlabel('PDB文件', fontsize=12)
        ax2.set_ylabel('结合残基数', fontsize=12)
        ax2.set_title('结合位点残基数量', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(filenames)))
        ax2.set_xticklabels(filenames, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"统计图表已保存至: {output_file}")
        plt.close()


def main():
    """主函数示例"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PDB蛋白质-配体复合物分析工具')
    parser.add_argument('input', help='PDB文件或包含PDB文件的目录')
    parser.add_argument('-o', '--output', help='输出报告文件路径')
    parser.add_argument('-b', '--batch', action='store_true', help='批量分析模式')
    parser.add_argument('-d', '--distance', type=float, default=4.0, 
                       help='相互作用距离阈值（默认: 4.0 Å）')
    
    args = parser.parse_args()
    
    if args.batch:
        # 批量分析模式
        print(f"批量分析目录: {args.input}")
        batch_analyzer = BatchPDBAnalyzer(args.input)
        batch_analyzer.load_all_pdbs()
        
        # 生成汇总表
        summary_file = args.output or "summary.txt"
        batch_analyzer.generate_summary_table(summary_file)
        print(f"汇总表已保存至: {summary_file}")
        
        # 生成统计图表
        batch_analyzer.plot_statistics("statistics.png")
        
        # 比较结合位点
        comparison = batch_analyzer.compare_binding_sites()
        print(f"\n共同结合残基: {sorted(comparison['common_residues'])}")
        
    else:
        # 单文件分析模式
        print(f"分析文件: {args.input}")
        analyzer = PDBComplexAnalyzer(args.input)
        analyzer.parse()
        
        # 生成报告
        report = analyzer.generate_report(args.output)
        print(report)
        
        # 显示相互作用
        interactions = analyzer.find_interactions(args.distance)
        print(f"\n发现 {len(interactions)} 个蛋白-配体相互作用（距离 < {args.distance} Å）")


if __name__ == "__main__":
    main()
